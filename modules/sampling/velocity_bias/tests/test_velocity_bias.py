"""Velocity bias: the structure step, the reference, the size-keeping rotation, learning through ratings."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    from modules.system.taste import store
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")


def _to(i, sched):
    return {"sample_sigmas": sched, "sigmas": sched[i:i + 1]}


def test_only_the_step_nearest_ninety_percent_of_the_noise_is_the_structure_step():
    from modules.sampling.velocity_bias import structure_step
    sched = torch.tensor([1.0, 0.909375, 0.725, 0.421875, 0.0])           # LTX distilled
    assert [structure_step(_to(i, sched)) for i in range(4)] == [False, True, False, False]
    far = torch.tensor([1.0, 0.6, 0.3, 0.0])                              # nothing within the window
    assert not any(structure_step(_to(i, far)) for i in range(3))
    assert not structure_step({})


def test_the_reference_is_the_average_or_the_nearest_liked_prompt_at_this_shape():
    from modules.sampling.velocity_bias import reference
    a, b = torch.ones(2, 2), -torch.ones(2, 2)
    sa, sb = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    rows = [{"reward": 1.0, "rows": {"v": a.half(), "sig": sa}},
            {"reward": 1.0, "rows": {"v": b.half(), "sig": sb}},
            {"reward": -1.0, "rows": {"v": a.half() * 5, "sig": sa}},       # a disliked clip never counts
            {"reward": 1.0, "rows": {"v": torch.ones(3, 3).half(), "sig": sa}}]   # another shape never counts
    mean, how = reference(rows, (2, 2), sa, nearest=False)
    assert torch.allclose(mean, torch.zeros(2, 2)) and "average of 2" in how
    near, how = reference(rows, (2, 2), sa, nearest=True)
    assert torch.allclose(near, a) and "nearest" in how
    assert reference(rows, (5, 5), sa, False)[0] is None and "size" in reference(rows, (5, 5), sa, False)[1]
    assert reference([], (2, 2), sa, False)[0] is None


def test_the_rotation_keeps_the_size_is_capped_and_fades():
    from modules.sampling.velocity_bias import rotate
    torch.manual_seed(0)
    x, d = torch.randn(1, 4, 3, 8, 8), torch.randn(1, 4, 3, 8, 8)
    out = rotate(x, d, strength=3.0, ratio=0.9)
    assert torch.allclose(out.norm(), x.norm(), rtol=1e-4)                # no energy added
    assert float((out - x).norm() / x.norm()) < 1.0                       # never wipes the clip
    small = rotate(x, d, strength=0.05, ratio=0.9)
    assert float((small - x).norm()) < float((out - x).norm())
    assert torch.equal(rotate(x, d, strength=1.0, ratio=0.0), x)         # no noise left: nothing to turn


def _load(tiny, **v):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    patched, status = FunPackLoadModifiers.execute(tiny.patcher, {
        "taste": {"key": "fox"}, "velocity_bias": {"enabled": True, **v}}).result
    wrap = [w for ws in patched.wrappers[WrappersMP.APPLY_MODEL].values() for w in ws][-1]
    outer = [w for ws in patched.wrappers[WrappersMP.OUTER_SAMPLE].values() for w in ws]
    return wrap, outer


def test_a_run_banks_its_motion_and_once_liked_the_next_run_is_turned_toward_it(tiny_h3):
    from conftest import packed_av
    from modules.system.taste import store
    sched = torch.tensor([1.0, 0.909375, 0.725, 0.421875, 0.0])
    torch.manual_seed(0)
    video, audio = torch.randn(1, 4, 3, 8, 8), torch.randn(1, 8, 5)
    x0, shapes = packed_av(video, audio)
    c = torch.ones(1, 4, 8)
    answer = x0 * 0.6 + torch.randn_like(x0)          # a direction that is not just x itself

    def run(pid, rate=None):
        wrap, outer = _load(tiny_h3, strength=1.0)
        seen = []
        import core.log as log
        log.new_run()

        def body():
            for i in range(4):
                wrap(lambda x, *a, **k: seen.append(x) or answer, x0, sched[i:i + 1], None, c, None,
                     _to(i, sched), latent_shapes=shapes)
        store.current_prompt_id = lambda: pid
        def nest(i):                              # the wrappers wrap each other, the body runs once inside them all
            return body() if i == len(outer) else outer[i](lambda: nest(i + 1))
        nest(0)
        if rate:
            store.rate(pid, rate)
        return seen

    first = run("r1", "liked")
    assert all(torch.equal(s, x0) for s in first)                          # nothing banked yet: untouched
    second = run("r2")
    assert torch.equal(second[0], x0) and torch.equal(second[2], x0) and torch.equal(second[3], x0)
    assert not torch.equal(second[1], x0)                                  # only the structure step is turned
    from conftest import unpacked
    assert torch.equal(unpacked(second[1], shapes)[1], unpacked(x0, shapes)[1])     # sound untouched


def test_strength_zero_only_learns(tiny_h3):
    wrap, _ = _load(tiny_h3, strength=0.0)
    from conftest import packed_av
    x0, shapes = packed_av(torch.randn(1, 4, 3, 8, 8), torch.randn(1, 8, 5))
    sched = torch.tensor([1.0, 0.909375, 0.725, 0.421875, 0.0])
    seen = []
    wrap(lambda x, *a, **k: seen.append(x) or x0, x0, sched[1:2], None, None, None, _to(1, sched), latent_shapes=shapes)
    assert torch.equal(seen[0], x0)
