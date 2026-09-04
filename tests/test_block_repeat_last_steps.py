"""H3 block repeat / span loop: the step-window that confines the repeat to part of the
schedule instead of firing on every step. Positive N = final N steps (tail); negative N =
first |N| steps (head) — see engine_settings.js hint for why both directions exist."""
import sys

import torch

sys.path.insert(0, ".")
from samplers import FunPackLTXAVSceneChainSampler as S  # noqa: E402


def _args(step, total=6):
    # sigmas holds N+1 values for N steps; sample_sigmas is the full descending schedule.
    sched = torch.linspace(1.0, 0.0, total + 1)
    return {"transformer_options": {"sample_sigmas": sched, "sigmas": sched[step:step + 1]}}


def test_zero_means_every_step():
    for step in range(6):
        assert S._in_last_steps(_args(step), 0) is True


def test_last_two_of_six():
    hits = [S._in_last_steps(_args(step), 2) for step in range(6)]
    assert hits == [False, False, False, False, True, True]


def test_window_wider_than_schedule_is_every_step():
    assert all(S._in_last_steps(_args(step), 50) for step in range(6))


def test_first_two_of_six():
    hits = [S._in_last_steps(_args(step), -2) for step in range(6)]
    assert hits == [True, True, False, False, False, False]


def test_head_and_tail_are_disjoint_at_the_same_magnitude():
    tail = [S._in_last_steps(_args(step), 2) for step in range(6)]
    head = [S._in_last_steps(_args(step), -2) for step in range(6)]
    assert not any(t and h for t, h in zip(tail, head))


def test_missing_schedule_fails_open():
    assert S._in_last_steps({"transformer_options": {}}, 2) is True
    assert S._in_last_steps({"transformer_options": {}}, -2) is True


if __name__ == "__main__":
    test_zero_means_every_step()
    test_last_two_of_six()
    test_window_wider_than_schedule_is_every_step()
    test_first_two_of_six()
    test_head_and_tail_are_disjoint_at_the_same_magnitude()
    test_missing_schedule_fails_open()
    print("ok")
