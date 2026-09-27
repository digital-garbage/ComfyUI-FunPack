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


def _x0():
    from comfy.nested_tensor import NestedTensor
    torch.manual_seed(3)
    return NestedTensor([torch.randn(1, 4, 3, 8, 8), torch.randn(1, 8, 5)])


def _start(outer):
    for o in outer:
        o(lambda: None)


def test_early_guidance_steers_in_the_first_quarter_and_banks_every_quarter(tiny_h3, monkeypatch):
    from modules.system.taste import store
    _teach("x0_quarters", ["q0", "q1", "q2", "q3"])
    monkeypatch.setattr(store, "current_prompt_id", lambda: "run-2")
    x0 = _x0()
    wrap, outer = _load(tiny_h3, "trajectory_guidance")

    def run():
        return [wrap(lambda *a, **k: x0, x0, None, transformer_options=_to(i)) for i in range(4)]

    results = {}
    for o in outer:
        o(lambda: results.setdefault("steps", run()))
    first = results["steps"][0]
    assert not torch.allclose(first.tensors[0], x0.tensors[0])
    pending = torch.load(store.ROOT / "fox" / "x0_quarters.pending.pt")["rows"]
    assert sorted(pending) == ["q0", "q1", "q2", "q3"]


def test_quarters_split_the_schedule_by_position():
    from modules.sampling.trajectory_guidance import quarter
    assert [quarter(_to(i, 8)) for i in range(8)] == [0, 0, 1, 1, 2, 2, 3, 3]
