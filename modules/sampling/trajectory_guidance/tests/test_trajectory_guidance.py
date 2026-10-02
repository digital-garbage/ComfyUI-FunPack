"""Early taste guidance through the loader, on H3's latent."""

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


def test_early_guidance_steers_in_the_first_quarter_and_banks_every_quarter(tiny_h3, monkeypatch):
    from modules.system.taste import store
    _teach("x0_quarters", ["q0", "q1", "q2", "q3"])
    monkeypatch.setattr(store, "current_prompt_id", lambda: "run-2")
    x0, shapes = _x0()
    wrap, outer = _load(tiny_h3, "trajectory_guidance")

    seen = []

    def model(x, *a, **k):
        seen.append(x)
        return x0

    def run():
        return [wrap(model, x0, _t(i), None, None, None, _to(i), latent_shapes=shapes) for i in range(4)]

    call = run
    for o in outer:                                         # nested, as one sampling call is
        call = (lambda w, inner: (lambda: w(lambda: inner())))(o, call)
    results = {"steps": call()}
    assert all(r is x0 for r in results["steps"])           # the model's own answer every step
    assert torch.equal(seen[0], x0)
    push = seen[1] - x0                                     # step 1's edit rode into step 2
    video = 4 * 3 * 8 * 8
    assert not torch.allclose(push[..., :video], torch.zeros_like(push[..., :video]))
    assert torch.equal(push[..., video:], torch.zeros_like(push[..., video:]))
    assert torch.equal(seen[3] - x0, torch.zeros_like(x0)) is False   # steps keep carrying
    pending = torch.load(store.ROOT / "fox" / "x0_quarters.pending.pt")["rows"]
    assert sorted(pending) == ["q0", "q1", "q2", "q3"]


def test_quarters_split_the_schedule_by_position():
    from modules.sampling.trajectory_guidance import quarter
    assert [quarter(_to(i, 8)) for i in range(8)] == [0, 0, 1, 1, 2, 2, 3, 3]
