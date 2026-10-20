import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def _load(tiny, **v):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    patched, status = FunPackLoadModifiers.execute(
        tiny.patcher, {"sharpness": {"enabled": True, **v}}).result
    wrap = [w for ws in patched.wrappers[WrappersMP.APPLY_MODEL].values() for w in ws][-1]
    return patched, wrap, status


def _to(i, steps=4):
    s = torch.linspace(1.0, 0.0, steps + 1)
    return {"sample_sigmas": s, "sigmas": s[i:i + 1]}


def test_off_installs_nothing(tiny_h3):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    patched, _ = FunPackLoadModifiers.execute(tiny_h3.patcher, {"sharpness": {"enabled": False}}).result
    assert not patched.wrappers.get(WrappersMP.APPLY_MODEL)


def test_a_plain_latent_is_pushed_away_from_the_previous_prediction_in_the_last_part_only(tiny_h3):
    _p, wrap, status = _load(tiny_h3, amount=0.5, last_part=0.5)
    assert "sharpness" in status
    x = torch.randn(1, 4, 3, 8, 8)
    outs = [x * f for f in (0.9, 0.5, 0.2, 0.1)]
    ts = torch.linspace(1.0, 0.0, 5)
    got = [wrap(lambda *a, o=outs[i], **kw: o, x, ts[i:i + 1], None, None, None, _to(i)) for i in range(4)]
    assert torch.equal(got[0], outs[0]) and torch.equal(got[1], outs[1])      # outside the last half
    kept = list(got)                                       # the boost is measured against the sharpened answer before
    for i in (2, 3):
        assert torch.allclose(got[i], outs[i] + 0.25 * (outs[i] - kept[i - 1]), atol=1e-5)


def test_a_packed_model_is_edited_on_its_answer_including_the_last_step_and_never_the_sound(tiny_h3):
    from conftest import packed_av, unpacked
    _p, wrap, _ = _load(tiny_h3, amount=1.0, last_part=1.0)
    x0, shapes = packed_av(torch.randn(1, 4, 3, 8, 8), torch.randn(1, 8, 5))
    answers = [x0 * f for f in (0.9, 0.5, 0.2, 0.1)]
    ts = torch.linspace(1.0, 0.0, 5)
    got = [wrap(lambda *a, o=answers[i], **k: o, x0, ts[i:i + 1], None, None, None, _to(i), latent_shapes=shapes)
           for i in range(4)]
    assert torch.equal(got[0], answers[0])                 # nothing to compare on step 0
    for i in (1, 3):
        assert not torch.allclose(unpacked(got[i], shapes)[0], unpacked(answers[i], shapes)[0])
        assert torch.equal(unpacked(got[i], shapes)[1], unpacked(answers[i], shapes)[1])   # sound untouched
