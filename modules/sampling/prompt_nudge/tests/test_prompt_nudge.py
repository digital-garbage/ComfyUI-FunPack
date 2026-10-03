"""Taste prompt nudge: words shift along the learned direction, late, with no extra pass."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    from modules.system.taste import store
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")


def _to(i, steps=4):
    s = torch.linspace(1.0, 0.0, steps + 1)
    return {"sample_sigmas": s, "sigmas": s[i:i + 1]}


def test_words_move_along_the_direction_by_a_share_of_their_own_size():
    from modules.sampling.prompt_nudge import nudged
    c = torch.ones(1, 3, 4)
    d = torch.tensor([1.0, 0, 0, 0])
    words = torch.tensor([True, True, False])
    out = nudged(c, words, d, 0.1)
    size = float(c[0, 0].norm())
    assert torch.allclose(out[0, 0, 0], c[0, 0, 0] + 0.1 * size)
    assert torch.equal(out[0, 2], c[0, 2])                  # a reference image's row is left alone
    assert torch.equal(out[0, :, 1:], c[0, :, 1:])


def _load(tiny, **v):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    patched, status = FunPackLoadModifiers.execute(tiny.patcher, {
        "taste": {"key": "fox"}, "prompt_nudge": {"enabled": True, **v}}).result
    wrap = [w for ws in patched.wrappers[WrappersMP.APPLY_MODEL].values() for w in ws][-1]
    for ws in patched.wrappers[WrappersMP.OUTER_SAMPLE].values():
        for o in ws:
            o(lambda: None)
    return wrap, status


def _teach(dim):
    from modules.system.taste import store
    liked = torch.zeros(dim)
    liked[0] = 1.0
    for i in range(6):
        store.capture("fox", "prompt_taste", {"pooled": liked if i % 2 else -liked}, prompt_id=f"t{i}")
        store.rate(f"t{i}", "liked" if i % 2 else "disliked")


def test_late_steps_nudge_the_words_in_one_pass_and_early_ones_do_not(tiny_h3):
    dim = 8
    _teach(dim)
    wrap, _ = _load(tiny_h3, strength=0.1)
    c = torch.ones(1, 4, dim)
    seen = []

    def executor(x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kw):
        seen.append(c_crossattn.clone())
        return x

    x = torch.zeros(1, 1, 4)
    ts = torch.linspace(1.0, 0.0, 5)
    for i in (0, 3):
        wrap(executor, x, ts[i:i + 1], None, c, None, _to(i))
    assert len(seen) == 2                                    # no extra passes
    assert torch.equal(seen[0], c)                           # first half: untouched
    assert (seen[1][0, :, 0] > c[0, :, 0]).all() and torch.equal(seen[1][..., 1:], c[..., 1:])


def test_a_direction_from_another_text_width_is_said_not_silently_dropped(tiny_h3):
    from core import log
    _teach(8)
    wrap, _ = _load(tiny_h3, strength=0.1)
    log.new_run()
    c = torch.ones(1, 4, 6)
    seen = []
    wrap(lambda x, t, *a, **k: seen.append(a[1]) or x, torch.zeros(1, 1, 4), torch.linspace(1.0, 0.0, 5)[3:4],
         None, c, None, _to(3))
    assert torch.equal(seen[0], c)
    assert any("different model" in e["message"] for e in log.history())


def test_off_installs_nothing(tiny_h3):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    patched, _ = FunPackLoadModifiers.execute(tiny_h3.patcher, {"prompt_nudge": {"enabled": False}}).result
    assert not patched.wrappers.get(WrappersMP.APPLY_MODEL)
