"""H3 block repeat / span loop: the last-N-steps window that confines the repeat to the
tail of the schedule instead of firing on every step (see engine_settings.js hint for why)."""
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


def test_missing_schedule_fails_open():
    assert S._in_last_steps({"transformer_options": {}}, 2) is True


if __name__ == "__main__":
    test_zero_means_every_step()
    test_last_two_of_six()
    test_window_wider_than_schedule_is_every_step()
    test_missing_schedule_fails_open()
    print("ok")
