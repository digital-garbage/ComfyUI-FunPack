"""Decisiveness: the maths, the learner, and the wrapper on a packed H3 latent."""

import math

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    from modules.system.taste import store
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")
    monkeypatch.setattr(store, "current_prompt_id", lambda: "run-1")


def _sched(steps):
    return torch.linspace(1.0, 0.0, steps + 1)


def _to(i, steps=8):
    s = _sched(steps)
    return {"sample_sigmas": s, "sigmas": s[i:i + 1]}


def test_k_of_one_is_exactly_off_and_the_rescale_scales_only_the_noise_part():
    from modules.sampling.decisiveness import factor, rescale_x0
    x, x0 = torch.randn(1, 4, 2, 4, 4), torch.randn(1, 4, 2, 4, 4)
    assert factor(0.5, 1.0) == pytest.approx(1.0)
    assert rescale_x0(x, x0, 0.5, 1.0) is x0
    out = rescale_x0(x, x0, 0.5, 1.3)
    eps = (x - 0.5 * x0) / 0.5
    assert torch.allclose(x - 0.5 * factor(0.5, 1.3) * eps, 0.5 * out, atol=1e-5)
    assert rescale_x0(x, x0, 1.0, 1.3) is x0          # pure noise: nothing to rescale


def test_the_learner_moves_toward_liked_and_away_from_disliked_and_is_bounded():
    from modules.sampling.decisiveness import LOGK_MAX, learned_logk
    assert learned_logk([]) == 0.0
    assert learned_logk([(0.2, 1.0)]) > 0.0
    assert learned_logk([(0.2, -1.0)]) < 0.0 + 1e-9 and learned_logk([(0.2, -1.0)]) == pytest.approx(-0.1)
    assert learned_logk([(5.0, 1.0)] * 20) == LOGK_MAX


def test_simple_at_four_steps_is_inert_and_longer_schedules_are_not():
    from modules.sampling.decisiveness import INERT_SHARE, reach
    simple4 = [1.0, 0.95, 0.89, 0.80, 0.0]            # a few-step schedule whose last step starts near 0.8
    assert reach(simple4, 0.9) < INERT_SHARE < reach(simple4, 1.3)
    assert reach(_sched(12).tolist(), 1.3) > INERT_SHARE
    assert reach(_sched(4).tolist(), 1.0) == 0.0


def _load(tiny, **values):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    settings = {"taste": {"key": "fox"}, "decisiveness": {"enabled": True, **values}}
    patched, status = FunPackLoadModifiers.execute(tiny.patcher, settings).result
    wrap = [w for ws in patched.wrappers[WrappersMP.APPLY_MODEL].values() for w in ws][-1]
    outer = [w for ws in patched.wrappers[WrappersMP.OUTER_SAMPLE].values() for w in ws]
    return wrap, outer, status


def _run(outer, fn):
    call = fn
    for o in reversed(outer):
        call = (lambda w, inner: (lambda: w(lambda: inner())))(o, call)
    return call()


def _packed():
    from conftest import packed_av
    torch.manual_seed(2)
    return packed_av(torch.randn(1, 4, 3, 8, 8), torch.randn(1, 8, 5))


def test_manual_k_steers_the_picture_into_the_next_step_and_never_the_answer_or_the_sound(tiny_h3):
    from conftest import unpacked
    wrap, outer, status = _load(tiny_h3, mode="manual", k=1.4)
    assert "manual k" in status
    x0, shapes = _packed()
    seen = []

    def model(x, *a, **k):
        seen.append(x)
        return x0

    def call(i):
        t = _sched(8)[i:i + 1]
        return wrap(model, x0.clone(), t, None, None, None, _to(i), latent_shapes=shapes)

    answers = _run(outer, lambda: [call(i) for i in range(8)])
    assert all(a is x0 for a in answers)                # the model's own answer, every step
    assert torch.equal(seen[0], x0)
    push = unpacked(seen[3] - x0, shapes)
    assert not torch.allclose(push[0], torch.zeros_like(push[0]))
    assert torch.equal(push[1], torch.zeros_like(push[1]))   # sound untouched


def test_learned_mode_banks_k_only_when_it_acts_and_a_rating_teaches_it(tiny_h3, monkeypatch):
    import random
    from modules.system.taste import store
    monkeypatch.setattr(random, "gauss", lambda *a: 0.2)     # the exploration draw, fixed
    wrap, outer, _ = _load(tiny_h3)
    x0, shapes = _packed()
    _run(outer, lambda: [wrap(lambda *a, **k: x0, x0, _sched(8)[i:i + 1], None, None, None, _to(i),
                              latent_shapes=shapes) for i in range(8)])
    assert store.rate("run-1", "liked")["recorded"] == ["decisiveness"]
    row = store.load("fox", "decisiveness")["rows"][0]
    assert abs(float(row["rows"]["logk"])) <= 0.3 + 1e-6 and row["reward"] == 1.0


def test_a_schedule_too_short_to_feel_it_says_so_and_teaches_nothing(tiny_h3):
    from core import log
    from modules.system.taste import store
    wrap, outer, _ = _load(tiny_h3)
    x0, shapes = _packed()
    log.new_run()
    _run(outer, lambda: [wrap(lambda *a, **k: x0, x0, _sched(2)[i:i + 1], None, None, None, _to(i, 2),
                              latent_shapes=shapes) for i in range(2)])
    assert any("barely acts" in e["message"] for e in log.history())
    assert not (store.ROOT / "fox" / "decisiveness.pending.pt").exists()


def test_learning_without_a_taste_key_is_off_and_says_so(tiny_h3):
    from comfy.patcher_extension import WrappersMP
    from core import log
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    log.new_run()
    patched, _ = FunPackLoadModifiers.execute(tiny_h3.patcher, {"decisiveness": {"enabled": True}}).result
    assert any("needs a Taste key" in e["message"] for e in log.history())
    assert not patched.wrappers.get(WrappersMP.APPLY_MODEL)


def test_a_run_whose_push_never_arrived_teaches_nothing(tiny_h3, monkeypatch):
    import random
    from modules.system.taste import store
    monkeypatch.setattr(random, "gauss", lambda *a: 0.2)
    wrap, outer, _ = _load(tiny_h3)
    x0, shapes = _packed()

    def twice():                                       # a second-order sampler: every step called twice
        for i in range(8):
            for _ in range(2):
                wrap(lambda *a, **k: x0, x0, _sched(8)[i:i + 1], None, None, None, _to(i), latent_shapes=shapes)

    _run(outer, twice)
    assert store.rate("run-1", "liked")["why"]


def test_ratings_from_another_schedule_length_are_left_out_of_the_centre(tiny_h3, monkeypatch):
    import random
    from core import log
    from modules.system.taste import store
    monkeypatch.setattr(random, "gauss", lambda *a: 0.0)
    for i in range(3):
        store.capture("fox", "decisiveness", {"logk": torch.tensor(0.3), "steps": torch.tensor(4)}, prompt_id=f"t{i}")
        store.rate(f"t{i}", "liked")
    wrap, outer, _ = _load(tiny_h3)
    x0, shapes = _packed()
    log.new_run()
    _run(outer, lambda: wrap(lambda *a, **k: x0, x0, _sched(8)[0:1], None, None, None, _to(0), latent_shapes=shapes))
    assert any("from 0 rating(s) at 8 steps" in e["message"] for e in log.history())
