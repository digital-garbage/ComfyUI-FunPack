import random

import pytest

from core import rated_dial
from core.rated_dial import Dial


def test_a_liked_value_pulls_the_centre_and_a_disliked_one_pushes_it_away():
    d = Dial(start=0.5, lo=0.0, hi=1.5, explore=0.1)
    assert d.centre([]) == 0.5
    assert d.centre([(1.0, 1.0, None)]) > 0.5
    assert d.centre([(1.0, -1.0, None)]) < 0.5
    assert d.centre([(9.0, 1.0, None)] * 20) == 1.5          # bounded


def test_a_run_that_barely_felt_the_feature_counts_for_less():
    d = Dial(start=0.5, lo=0.0, hi=1.5, explore=0.1)
    past = [(0.5, 0.0, 1.0)] * 3                              # three runs with a usual effect of 1
    weak = d.centre(past + [(1.0, 1.0, 0.1)])
    strong = d.centre(past + [(1.0, 1.0, 2.0)])
    assert weak < strong


def test_unusualness_needs_history_and_is_clamped():
    assert rated_dial.unusualness(5.0, [1.0]) == 1.0
    assert rated_dial.unusualness(100.0, [1.0, 1.0, 1.0]) == 2.0
    assert rated_dial.unusualness(0.0001, [1.0, 1.0, 1.0]) == 0.25


def test_rows_become_history_and_pick_stays_in_range():
    import torch
    rows = [{"reward": 1.0, "rows": {"v": torch.tensor(1.0), "e": torch.tensor(0.2)}},
            {"reward": -1.0, "rows": {"other": torch.tensor(1.0)}}]
    assert rated_dial.history(rows) == [(1.0, 1.0, pytest.approx(0.2))]
    v, centre, n = Dial(0.5, 0.0, 1.5, 5.0).pick([], random.Random(1))
    assert 0.0 <= v <= 1.5 and centre == 0.5 and n == 0
