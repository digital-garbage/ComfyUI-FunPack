"""Input steering: an edit made at step i rides into step i+1's input, scaled by that step's
noise, and the model's own answer is what comes back."""

import torch

from core import input_steer, log


def _to(i, steps=4, **extra):
    sched = torch.linspace(1.0, 0.0, steps + 1)
    return {"sample_sigmas": sched, "sigmas": sched[i:i + 1], **extra}


def _t(i, steps=4):
    return torch.linspace(1.0, 0.0, steps + 1)[i:i + 1]


def _x():
    torch.manual_seed(0)
    return torch.randn(1, 1, 12)


def _edit(steer, i, x, out, delta, steps=4):
    """What a wrapper does at step i: begin, 'run', and keep out + delta."""
    step = steer.begin(x, _t(i, steps), _to(i, steps))
    kept = step.keep(out, out + delta) if step.gate > 0 else out
    return step, kept


def test_the_edit_lands_in_the_next_input_scaled_by_its_noise_and_the_answer_stays_the_models():
    s, x, out, d = input_steer.Steer("t"), _x(), _x() * 2, torch.ones(1, 1, 12)
    step, kept = _edit(s, 2, x, out, d)                  # 4 steps: the one gated push, 2 -> 3
    assert kept is out
    nxt = s.begin(x, _t(3), _to(3))
    sigma = float(_t(3))
    assert torch.allclose(nxt.x, x + (1 - sigma) * d)


def test_step_one_gets_no_push_and_the_last_step_computes_nothing():
    s, x = input_steer.Steer("t"), _x()
    assert torch.equal(s.begin(x, _t(0), _to(0)).x, x)
    last = s.begin(x, _t(3), _to(3))
    assert last.final and last.gate == 0.0


def test_the_gate_is_the_one_of_the_step_the_push_lands_on():
    s, x = input_steer.Steer("t"), _x()
    assert [s.begin(x, _t(i), _to(i)).gate for i in range(4)] == [0.0, 0.0, 0.5, 0.0]


def test_a_second_call_in_the_same_step_does_not_get_that_steps_push():
    s, x, out, d = input_steer.Steer("t"), _x(), _x(), torch.ones(1, 1, 12)
    _edit(s, 2, x, out, d)
    second = s.begin(x, _t(2), _to(2))                   # a split batch, same step
    assert torch.equal(second.x, x)


def test_a_new_pass_drops_everything_held():
    s, x, out, d = input_steer.Steer("t"), _x(), _x(), torch.ones(1, 1, 12)
    _edit(s, 2, x, out, d)
    s.begin(x, _t(3), _to(3))
    assert torch.equal(s.begin(x, _t(1), _to(1)).x, x)   # the schedule restarted
    assert torch.equal(s.begin(x, _t(2), _to(2)).x, x)   # and step 1 made no push


def test_windows_and_cond_rows_keep_separate_pushes():
    s, x, out, d = input_steer.Steer("t"), _x(), _x(), torch.ones(1, 1, 12)
    step = s.begin(x, _t(2), _to(2, cond_or_uncond=[0]))
    step.keep(out, out + d)
    assert torch.equal(s.begin(x, _t(3), _to(3, cond_or_uncond=[1])).x, x)
    assert not torch.equal(s.begin(x, _t(3), _to(3, cond_or_uncond=[0])).x, x)


def test_a_probe_and_an_off_schedule_call_are_inert_and_the_latter_says_so():
    s, x, out = input_steer.Steer("t"), _x(), _x()
    probe = s.begin(x, _t(2), _to(2, funpack_probe=True))
    assert probe.gate == 0.0 and probe.keep(out, out + 1) is out
    log._reset()
    off = s.begin(x, torch.tensor([0.123]), _to(2) | {"sigmas": torch.tensor([0.123])})
    assert off.gate == 0.0 and off.keep(out, out + 1) is out
    assert any("not a step of the schedule" in r["message"] for r in log.history())


def test_a_plain_video_latent_keeps_the_old_behaviour():
    s, x, out = input_steer.Steer("t"), torch.randn(1, 4, 2, 4, 4), torch.zeros(1, 4, 2, 4, 4)
    step = s.begin(x, _t(3), _to(3))
    assert step.x is x and step.gate == 0.5 and not step.final
    assert torch.equal(step.keep(out, out + 1), out + 1)


def test_the_run_says_how_big_the_edits_were_or_that_there_were_none():
    log._reset()
    s, x, out, d = input_steer.Steer("DynaShift"), _x(), _x() + 1, torch.ones(1, 1, 12) * 0.1
    _edit(s, 2, x, out, d)
    s.begin(x, _t(3), _to(3))
    assert any("DynaShift" in r["source"] and "Active |" in r["message"] for r in log.history())
    log._reset()
    quiet = input_steer.Steer("Quiet")
    quiet.begin(x, _t(3), _to(3))
    assert any("Inactive | no step made an edit" in r["message"] for r in log.history())


def test_a_one_step_schedule_says_there_is_nothing_to_carry_into():
    log._reset()
    s, x = input_steer.Steer("t"), _x()
    step = s.begin(x, _t(0, 1), _to(0, 1))
    assert step.final and any("1-step" in r["message"] for r in log.history())


def _said(text):
    return any(text in r["message"] for r in log.history())


def test_a_sliver_of_a_push_is_reported_as_barely_acting_and_a_real_one_as_active():
    x, out = _x(), _x()
    for size, word in ((1e-4, "barely acts"), (1.0, "Active |")):
        log._reset()
        s = input_steer.Steer("t")
        _edit(s, 2, x, out, torch.ones(1, 1, 12) * size)
        s.begin(x, _t(3), _to(3))
        assert _said(word), (size, [r["message"] for r in log.history()])


def test_a_second_run_on_the_same_object_does_not_report_the_first_runs_edits():
    log._reset()
    s, x, out = input_steer.Steer("t"), _x(), _x()
    _edit(s, 2, x, out, torch.ones(1, 1, 12))
    s.begin(x, _t(3), _to(3))
    s.reset()                                           # what the sampling wrapper does per run
    log._reset()
    s.begin(x, _t(3), _to(3))
    assert _said("Inactive | no step made an edit") and not _said("Active |")


def test_a_repeated_sigma_cannot_count_an_edit_that_never_arrived():
    s, x, out = input_steer.Steer("t"), _x(), _x()
    sched = torch.tensor([1.0, 0.8, 0.8, 0.5, 0.0])
    to = lambda i: {"sample_sigmas": sched, "sigmas": sched[i:i + 1]}
    s.begin(x, sched[2:3], to(2)).keep(out, out + 1)    # index reads 1, the first match: gate 0, no edit
    log._reset()
    s.begin(x, sched[3:4], to(3))
    assert not _said("Active |")


class _Patcher:
    def __init__(self):
        self.wrappers = []

    def add_wrapper_with_key(self, _what, _key, fn):
        self.wrappers.append(fn)


def test_the_sampling_wrapper_resets_each_run_and_drops_pushes_even_on_an_interrupt():
    s, x, out, p = input_steer.Steer("t"), _x(), _x(), _Patcher()
    s.attach(p, "k")
    _edit(s, 2, x, out, torch.ones(1, 1, 12))
    assert s._pushes                                    # held mid-run
    outer = p.wrappers[0]

    def boom():
        raise KeyboardInterrupt

    try:
        outer(lambda: (_edit(s, 2, x, out, torch.ones(1, 1, 12)), boom()))
    except KeyboardInterrupt:
        pass
    assert not any(s._pushes.values())                  # nothing latent-sized outlives the run
    s._made = 5
    outer(lambda: None)
    assert s._made == 0                                 # a new run counts from zero


def test_a_new_pass_inside_one_run_drops_what_was_held():
    s, x, out, d = input_steer.Steer("t"), _x(), _x(), torch.ones(1, 1, 12)
    _edit(s, 2, x, out, d)
    s.begin(x, _t(1), _to(1))                           # index went backwards
    assert not s._pushes[s._key(_to(1))]


def test_a_push_of_the_wrong_size_is_not_added_and_says_so():
    log._reset()
    s, x, out = input_steer.Steer("t"), _x(), _x()
    _edit(s, 2, x, out, torch.ones(1, 1, 12))
    bigger = torch.randn(1, 1, 20)
    assert torch.equal(s.begin(bigger, _t(3), _to(3)).x, bigger)
    assert _said("changed size")


def test_a_push_is_delivered_once():
    s, x, out = input_steer.Steer("t"), _x(), _x()
    _edit(s, 2, x, out, torch.ones(1, 1, 12))
    s.begin(x, _t(3), _to(3))
    assert s._delivered == 1
    s.begin(x, _t(3), _to(3))                           # same index again: multi-call, nothing re-delivered
    assert s._delivered == 1


def test_a_sampler_that_calls_twice_per_step_is_declared_and_left_alone():
    log._reset()
    s, x, out = input_steer.Steer("t"), _x(), _x()
    sched = torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0])
    to = lambda i: {"sample_sigmas": sched, "sigmas": sched[i:i + 1]}
    # heun: step 1 calls at sigma_1 then sigma_2; step 2 calls at sigma_2 again, then sigma_3
    s.begin(x, sched[1:2], to(1))
    s.begin(x, sched[2:3], to(2))
    second = s.begin(x, sched[2:3], to(2))
    assert second.gate == 0.0 and second.keep(out, out + 1) is out
    assert _said("more than once per step") and not _said("Active |")


def test_a_schedule_that_repeats_a_sigma_is_blamed_on_the_schedule_not_the_sampler():
    log._reset()
    s, x = input_steer.Steer("t"), _x()
    sched = torch.tensor([1.0, 0.8, 0.8, 0.5, 0.0])
    step = s.begin(x, sched[1:2], {"sample_sigmas": sched, "sigmas": sched[1:2]})
    assert step.gate == 0.0 and _said("repeats a sigma") and not _said("second-order")


def test_after_a_multi_call_verdict_nothing_is_held_or_steered_for_the_run():
    s, x, out = input_steer.Steer("t"), _x(), _x()
    sched = torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0])
    to = lambda i: {"sample_sigmas": sched, "sigmas": sched[i:i + 1]}
    s.begin(x, sched[1:2], to(1))
    s.begin(x, sched[1:2], to(1))                       # same index twice: the verdict
    later = s.begin(x, sched[2:3], to(2))
    assert later.gate == 0.0 and later.keep(out, out + 1) is out and not any(s._pushes.values())


def test_each_run_says_its_own_result_not_just_the_first_of_the_prompt():
    s, x = input_steer.Steer("t"), _x()
    log._reset()
    s.begin(x, _t(3), _to(3))
    assert _said("Inactive | no step made")
    seen = len([r for r in log.history() if "no step made" in r["message"]])
    s.reset()
    s.begin(x, _t(3), _to(3))                           # run 2, same prompt: log.once not cleared
    assert len([r for r in log.history() if "no step made" in r["message"]]) == seen + 1
