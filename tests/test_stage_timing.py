"""Which stage of a run cost what, for the runs the encoder tally exonerates.

2.4s at three rated samples and 24.9s at five is not smooth growth, and the shape of that
curve says nothing about which stage draws it. This names them.
"""
import sys

import pytest

sys.path.insert(0, ".")


@pytest.fixture
def refiner():
    from conditioning import FunPackVideoRefinerV2
    r = FunPackVideoRefinerV2()
    r._v2_stage_times = {}
    return r


def test_nothing_timed_says_nothing(refiner):
    assert refiner._v2_stage_status() == ""


def test_a_stage_is_timed_and_named(refiner):
    with refiner._v2_stage("value function"):
        pass
    refiner._v2_stage_times["value function"]["seconds"] = 8.0
    assert "value function 8.0s" in refiner._v2_stage_status()


def test_repeat_calls_accumulate_and_are_counted(refiner):
    for _ in range(3):
        with refiner._v2_stage("value function"):
            pass
    slot = refiner._v2_stage_times["value function"]
    assert slot["calls"] == 3
    slot["seconds"] = 9.0
    assert "value function 9.0s x3" in refiner._v2_stage_status()


def test_cheap_stages_do_not_crowd_the_line(refiner):
    with refiner._v2_stage("trivial"):
        pass
    assert refiner._v2_stage_status() == ""          # under the floor


def test_the_costliest_leads(refiner):
    for name, secs in (("phrase memory", 1.0), ("value function", 9.0)):
        with refiner._v2_stage(name):
            pass
        refiner._v2_stage_times[name]["seconds"] = secs
    line = refiner._v2_stage_status()
    assert line.index("value function") < line.index("phrase memory")


def test_the_share_of_the_run_is_reported(refiner):
    with refiner._v2_stage("value function"):
        pass
    refiner._v2_stage_times["value function"]["seconds"] = 5.0
    assert "50% of 10.0s" in refiner._v2_stage_status(total_seconds=10.0)


def test_a_raising_stage_is_still_timed(refiner):
    """A stage that throws must not swallow the exception or lose its measurement."""
    with pytest.raises(ValueError):
        with refiner._v2_stage("value function"):
            raise ValueError("boom")
    assert refiner._v2_stage_times["value function"]["calls"] == 1
