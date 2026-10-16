"""Taste guidance through the loader, on H3's latent."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    from modules.system.taste import store
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")


def _to(index, steps=4):
    sched = torch.linspace(1.0, 0.0, steps + 1)
    return {"sample_sigmas": sched, "sigmas": sched[index:index + 1]}


def _load(tiny, module):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    patched, _ = FunPackLoadModifiers.execute(tiny.patcher, {
        "taste": {"key": "fox"}, module: {"enabled": True, "strength": 0.05}}).result
    wrap = [w for ws in patched.wrappers[WrappersMP.APPLY_MODEL].values() for w in ws][-1]
    outer = [w for ws in patched.wrappers[WrappersMP.OUTER_SAMPLE].values() for w in ws]
    return wrap, outer


def _teach(kind, names):
    from modules.system.taste import store, value
    torch.manual_seed(0)
    liked = torch.randn(value.DIM)
    for i in range(12):
        sign = 1.0 if i % 2 else -1.0
        store.capture("fox", kind, {n: sign * liked + 0.3 * torch.randn(value.DIM) for n in names},
                      prompt_id=f"t{i}")
        store.rate(f"t{i}", "liked" if sign > 0 else "disliked")


def _t(index, steps=4):
    return torch.linspace(1.0, 0.0, steps + 1)[index:index + 1]


def _x0():
    from conftest import packed_av
    torch.manual_seed(3)
    return packed_av(torch.randn(1, 4, 3, 8, 8), torch.randn(1, 8, 5))


def _start(outer):
    for o in outer:
        o(lambda: None)


def test_late_guidance_banks_the_final_picture_and_steers_only_late(tiny_h3, monkeypatch):
    from modules.system.taste import store
    monkeypatch.setattr(store, "current_prompt_id", lambda: "run-1")
    x0, shapes = _x0()
    wrap, outer = _load(tiny_h3, "output_guidance")
    for o in outer:
        o(lambda: wrap(lambda *a, **k: x0, x0, None, None, None, None, _to(3), latent_shapes=shapes))
    assert store.rate("run-1", "liked")["recorded"] == ["x0_final"]

    _teach("x0_final", ["final"])
    wrap, outer = _load(tiny_h3, "output_guidance")
    _start(outer)
    from conftest import unpacked
    seen = []

    def model(x, *a, **k):
        seen.append(x)
        return x0

    def call(i):
        return wrap(model, x0, _t(i), None, None, None, _to(i), latent_shapes=shapes)

    assert call(1) is x0 and call(2) is x0                  # the model's own answer, never the edit
    assert torch.equal(seen[0], x0)                         # step 1: no push yet
    assert call(3) is x0
    push = seen[-1] - x0                                    # the late step's input carries it
    assert not torch.allclose(unpacked(push, shapes)[0], torch.zeros_like(unpacked(x0, shapes)[0]))
    assert torch.equal(unpacked(push, shapes)[1], torch.zeros_like(unpacked(x0, shapes)[1]))   # sound untouched
    assert len(seen) == 3 and torch.equal(seen[1], x0)      # step 2's input had no push
