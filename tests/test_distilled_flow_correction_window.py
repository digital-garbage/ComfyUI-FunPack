"""`final_correction_steps` counts steps that can actually be corrected.

The Heun corrector evaluates the model at `sigma_next`. On the terminal step sigma_next is 0,
which is degenerate — and that step returns early, before the corrector is reached. So a window
measured back from the END of the schedule began on a step that never runs the corrector, and
`final_correction_steps=1` (the old default) performed ZERO corrections on any schedule ending
at 0 — which is every stock scheduler.

These tests pin the arithmetic rather than the sampler loop: the loop needs a real model, but
the off-by-one lives entirely in how the window is derived from the sigma array.
"""
import inspect
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import samplers


def window(sigmas, final_correction_steps):
    """The sampler's own derivation, read out of its source so it cannot drift from it."""
    src = inspect.getsource(samplers.sample_funpack_distilled_flow)
    assert "correctable = sum(1 for i in range(sched_steps) if float(sigmas[i + 1]) > 0)" in src
    sched_steps = len(sigmas) - 1
    correctable = sum(1 for i in range(sched_steps) if float(sigmas[i + 1]) > 0)
    n = max(0, min(correctable, int(final_correction_steps)))
    return correctable - n, sched_steps


def corrected_steps(sigmas, n):
    """Which step indices actually reach the corrector."""
    start, sched_steps = window(sigmas, n)
    return [i for i in range(sched_steps)
            if i >= start and float(sigmas[i + 1]) > 0]


STOCK = [1.0, 0.988, 0.973, 0.952, 0.923, 0.878, 0.800, 0.632, 0.0]   # simple, 8 steps
USERS = [1.00, 0.94, 0.83, 0.72, 0.55, 0.30, 0.10, 0.00]              # hand-tuned, 7 steps


def test_one_means_one_correction():
    """The bug: this used to be zero."""
    assert len(corrected_steps(STOCK, 1)) == 1


def test_it_corrects_the_last_real_step_not_the_terminal_one():
    assert corrected_steps(STOCK, 1) == [6]        # step 7 of 8; step 8 lands on sigma 0


def test_two_means_two():
    assert corrected_steps(STOCK, 2) == [5, 6]


def test_zero_corrects_nothing():
    assert corrected_steps(STOCK, 0) == []


def test_the_default_is_zero_because_that_is_what_runs_today():
    """Fixing the arithmetic without moving the default would have added a model call to
    every existing run. 0 reproduces the behaviour every run has actually had."""
    sig = inspect.signature(samplers.sample_funpack_distilled_flow)
    assert sig.parameters["final_correction_steps"].default == 0
    node = samplers.FunPackDistilledFlowSampler
    assert inspect.signature(node.get_sampler).parameters["final_correction_steps"].default == 0
    widget = node.INPUT_TYPES()["required"]["final_correction_steps"][1]
    assert widget["default"] == 0
    assert "EXTRA MODEL CALL" in widget["tooltip"]


def test_it_works_on_a_hand_written_schedule():
    assert corrected_steps(USERS, 1) == [5]        # 0.30 -> 0.10; the 0.10 -> 0 step cannot be


def test_a_schedule_that_does_not_reach_zero_can_correct_its_last_step():
    """Nothing is special-cased about the final index — only about sigma 0."""
    tail = [1.0, 0.7, 0.4, 0.1]
    assert corrected_steps(tail, 1) == [2]


def test_asking_for_more_than_exist_is_clamped_not_wrapped():
    assert corrected_steps([1.0, 0.5, 0.0], 3) == [0]
    assert corrected_steps(STOCK, 99) == list(range(7))


def test_the_terminal_step_is_never_corrected_at_any_setting():
    for n in range(0, 4):
        assert 7 not in corrected_steps(STOCK, n)


def test_sharpness_shares_the_window_deliberately():
    """quality_sharpness' own tooltip defines its window as the Heun-correction steps, so it
    reads the same index. Pinned because decoupling them silently would change its behaviour."""
    src = inspect.getsource(samplers.sample_funpack_distilled_flow)
    assert len(re.findall(r"if i >= correction_start_idx:", src)) == 2
