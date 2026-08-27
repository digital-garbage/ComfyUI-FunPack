"""What the refinement key costs per run, and which collection is making it big.

The key is read and rewritten in full every run and holds base64 conditioning tensors, so
its cost grows with everything ever learned. "2.0s, then 2.4s, slowly but surely" is what
that looks like from outside; this line is what turns it into something to act on.
"""
import sys

import pytest

sys.path.insert(0, ".")


@pytest.fixture
def refiner():
    from conditioning import FunPackVideoRefinerV2
    r = FunPackVideoRefinerV2()
    r._v2_state_io = {}
    return r


def test_no_io_says_nothing(refiner):
    assert refiner._v2_state_io_status() == ""


def test_size_and_both_phases_are_reported(refiner):
    refiner._v2_note_state_io("load", 0.4, 5 * 1_048_576)
    refiner._v2_note_state_io("save", 0.9, 6 * 1_048_576)
    line = refiner._v2_state_io_status()
    assert "6.0 MB" in line          # the size after writing, not before
    assert "load 0.40s" in line and "save 0.90s" in line


def test_the_biggest_collections_are_named(refiner):
    data = {"global": {"phrase_memory": {str(i): {} for i in range(120)},
                       "conditioning_deltas": {"a": {}, "b": {}},
                       "path_outcomes": {}},
            "prompt_histories": {str(i): {} for i in range(7)}}
    refiner._v2_note_state_io("save", 1.0, 1_048_576, data)
    line = refiner._v2_state_io_status()
    assert "phrase_memory 120" in line
    assert "prompt_histories 7" in line
    assert "path_outcomes" not in line      # empty ones are not worth the width


def test_the_largest_leads(refiner):
    data = {"global": {"phrase_memory": {"a": {}},
                       "conditioning_deltas": {str(i): {} for i in range(50)}}}
    refiner._v2_note_state_io("save", 1.0, 1_048_576, data)
    line = refiner._v2_state_io_status()
    assert line.index("conditioning_deltas") < line.index("phrase_memory")


def test_a_run_starts_from_zero(refiner):
    refiner._v2_note_state_io("load", 1.0, 1_048_576)
    refiner._v2_state_io = {}
    assert refiner._v2_state_io_status() == ""
