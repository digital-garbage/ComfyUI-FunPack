"""DynaShift: the step gate, the shift itself, and banking through a taste key."""

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


def _t(index, steps=4):
    return torch.linspace(1.0, 0.0, steps + 1)[index:index + 1]


def test_the_gate_opens_over_the_last_half_only():
    from core.dit_hooks import late_half as gate
    assert [gate(_to(i)) for i in range(4)] == [0.0, 0.0, 0.0, 0.5]
    assert gate({}) == 0.0


def _bank(neg, pos=()):
    from modules.sampling.dynashift import Bank
    rows = [{"reward": -1.0, "rows": {"latent": n}} for n in neg]
    rows += [{"reward": 1.0, "rows": {"latent": p}} for p in pos]
    return Bank(rows)


def test_a_frame_like_a_disliked_one_loses_that_likeness_and_others_are_left_alone():
    from modules.sampling.dynashift import shift
    torch.manual_seed(0)
    bad = torch.randn(4, 2, 8, 8)
    other = torch.randn(4, 1, 8, 8)
    video = torch.cat([bad[:, :1] + 0.1 * torch.randn(4, 1, 8, 8), other], dim=1)[None]
    out = shift(video, _bank([bad]), strength=1.0, threshold=0.6)
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    assert cos(out[0, :, 0], bad[:, 0]) < 0.5 * cos(video[0, :, 0], bad[:, 0])
    assert torch.equal(out[0, :, 1], video[0, :, 1])        # never matched: untouched


def test_nothing_banked_at_this_resolution_changes_nothing():
    from modules.sampling.dynashift import shift
    assert shift(torch.randn(1, 4, 2, 8, 8), _bank([torch.randn(4, 2, 16, 16)]), 1.0, 0.6) is None


def test_it_banks_the_last_step_and_steers_once_rated(tiny_h3, monkeypatch):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    from modules.system.taste import store

    def load():
        patched, status = FunPackLoadModifiers.execute(tiny_h3.patcher, {
            "taste": {"key": "fox"}, "dynashift": {"enabled": True, "strength": 1.0}}).result
        wrap = [w for ws in patched.wrappers[WrappersMP.APPLY_MODEL].values() for w in ws][-1]
        outer = [w for ws in patched.wrappers[WrappersMP.OUTER_SAMPLE].values() for w in ws]
        return wrap, outer, status

    torch.manual_seed(1)
    bad = torch.randn(1, 4, 3, 8, 8)
    audio = torch.randn(1, 8, 5)
    from conftest import packed_av, unpacked
    x0, shapes = packed_av(bad, audio)
    for i, rating in enumerate(["disliked", "disliked"]):
        monkeypatch.setattr(store, "current_prompt_id", lambda i=i: f"run-{i}")
        wrap, outer, status = load()
        for o in outer:
            o(lambda: wrap(lambda *a, **k: x0, x0, None, None, None, None, _to(3), latent_shapes=shapes))
        assert store.rate(f"run-{i}", rating)["recorded"] == ["dynashift"]

    wrap, outer, _status = load()
    from core import log
    log.new_run()
    for o in outer:
        o(lambda: None)                                     # a run starts: the bank is read
    assert any("2 disliked, 0 liked banked" in e["message"] for e in log.history())
    seen = []

    def model(x, *a, **k):
        seen.append(x)
        return x0

    def call(i):
        return wrap(model, x0, _t(i), None, None, None, _to(i), latent_shapes=shapes)

    assert call(1) is x0                                    # first half: untouched
    assert call(2) is x0                                    # the model's own answer, never the edit
    assert call(3) is x0
    push = unpacked(seen[-1] - x0, shapes)                  # the edit rode into the last step's input
    assert not torch.allclose(push[0], torch.zeros_like(push[0]))
    assert torch.equal(push[1], torch.zeros_like(push[1]))   # sound untouched
    assert torch.equal(seen[1], x0)                         # nothing was pushed into step 2
