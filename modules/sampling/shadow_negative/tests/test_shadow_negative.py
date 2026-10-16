"""Shadow negative on a real H3 forward.

The sampling run is ComfyUI's CFGGuider in real life; here the OUTER_SAMPLE
wrapper is called the way comfy/samplers.py calls it -- an executor whose
class_obj is the guider, with `conds` already copied -- around a real forward.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


class _Guider:
    def __init__(self, negative):
        self.conds = {"positive": [{"cross_attn": torch.zeros(1, 5, 48)}],
                      "negative": [] if negative is None else [{"cross_attn": negative}]}


class _Executor:
    def __init__(self, guider, fn):
        self.class_obj, self._fn = guider, fn

    def __call__(self, *a, **k):
        return self._fn()


def _sample(tiny, patched, negative, **run):
    from comfy.patcher_extension import WrappersMP
    result = {}

    def go():
        result["out"] = tiny.run(patched, **run)

    wrappers = [w for ws in patched.wrappers.get(WrappersMP.OUTER_SAMPLE, {}).values() for w in ws]
    call = go
    for w in wrappers:
        call = (lambda w, inner: (lambda: w(_Executor(_Guider(negative), inner))))(w, call)
    call()
    return result["out"]


def _load(tiny, **values):
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    return FunPackLoadModifiers.execute(
        tiny.patcher, {"shadow_negative": {"enabled": True, "end": 1.0, **values}}).result


def test_it_pushes_the_picture_when_there_is_a_negative(tiny_h3):
    base = tiny_h3.run()
    patched, status = _load(tiny_h3, video_scale=4.0, alpha=1.0)
    assert "shadow_negative: picture 4" in status
    out = _sample(tiny_h3, patched, torch.randn(1, 4, 48))
    assert not torch.allclose(out[0], base[0]), "installed and changed nothing (v4's bug)"
    # sound push is 1 = off, but attention mixes modalities through later blocks
    assert torch.isfinite(out[0]).all() and torch.isfinite(out[1]).all()


def test_with_no_negative_prompt_the_output_is_untouched(tiny_h3):
    base = tiny_h3.run()
    patched, _ = _load(tiny_h3, video_scale=4.0, alpha=1.0)
    out = _sample(tiny_h3, patched, None)
    assert torch.allclose(out[0], base[0])


def test_a_different_negative_gives_a_different_result(tiny_h3):
    patched, _ = _load(tiny_h3, video_scale=4.0, alpha=1.0)
    a = _sample(tiny_h3, patched, torch.randn(1, 4, 48))
    b = _sample(tiny_h3, patched, torch.randn(1, 4, 48))
    assert not torch.allclose(a[0], b[0]), "the negative text does not reach the push"


def test_outside_the_step_window_it_does_nothing(tiny_h3):
    base_late = tiny_h3.run(sigma=0.3, sigmas=torch.tensor([1.0, 0.7, 0.3, 0.0]))
    patched, _ = _load(tiny_h3, video_scale=4.0, alpha=1.0, start=0.0, end=0.4)
    late = _sample(tiny_h3, patched, torch.randn(1, 4, 48),
                   sigma=0.3, sigmas=torch.tensor([1.0, 0.7, 0.3, 0.0]))
    assert torch.allclose(late[0], base_late[0])


def test_compose_leaves_a_block_another_feature_holds(tiny_h3):
    from core import dit_hooks
    held = tiny_h3.patcher.clone()
    ran = []
    dit_hooks.add_block_hook(held, "other", 2,
                             lambda a, e: (ran.append(1), e["original_block"](a))[1])
    from modules.sampling import shadow_negative as m
    note = m.install(held, {**{k: v["default"] for k, v in m.SETTINGS.items()},
                            "enabled": True, "compose": True, "end": 1.0}, key="funpack.sn")
    assert "leaves block(s) 2" in note
    _sample(tiny_h3, held, torch.randn(1, 4, 48))
    assert ran == [1]


def test_it_runs_on_an_i2v_scene_with_a_pinned_first_frame(tiny_h3):
    mask = tiny_h3.pinned_first_frame()
    base = tiny_h3.run(denoise_mask=mask)
    patched, _ = _load(tiny_h3, video_scale=4.0, alpha=1.0)
    out = _sample(tiny_h3, patched, torch.randn(1, 4, 48), denoise_mask=mask)
    assert not torch.allclose(out[0], base[0]), "per-token mod rows made it stand down"
