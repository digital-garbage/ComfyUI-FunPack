"""Where the time before sampling goes, counted by purpose.

On H3 the text encoder is Qwen3-VL-32B and most calls to it are bookkeeping — classifying a
phrase, matching a memory to the scene — not the prompt. Counting passes by purpose is what
turns "26 seconds before generation starts" into a list of things to remove.
"""
import sys

import pytest

sys.path.insert(0, ".")


@pytest.fixture
def refiner():
    from conditioning import FunPackVideoRefinerV2
    r = FunPackVideoRefinerV2()
    r._v2_reset_encode_tally()
    return r


def test_nothing_encoded_says_nothing(refiner):
    assert refiner._v2_encode_tally_status() == ""


def test_passes_are_counted_by_purpose(refiner):
    refiner._v2_tally_encode("prompt", 4.0)
    refiner._v2_tally_encode("phrase classification", 1.0)
    refiner._v2_tally_encode("phrase classification", 1.5)
    line = refiner._v2_encode_tally_status()
    assert "3 pass(es) in 6.5s" in line
    assert "phrase classification 2x 2.5s" in line
    assert "prompt 1x 4.0s" in line


def test_the_costliest_purpose_is_named_first(refiner):
    """The line is read to decide what to remove, so the biggest cost leads."""
    refiner._v2_tally_encode("prompt", 1.0)
    refiner._v2_tally_encode("category vectors", 9.0)
    line = refiner._v2_encode_tally_status()
    assert line.index("category vectors") < line.index("prompt 1x")


def test_cache_hits_are_shown_but_cost_nothing(refiner):
    refiner._v2_tally_encode("category vectors", 2.0)
    refiner._v2_tally_encode("category vectors", 0.0, cached=True)
    line = refiner._v2_encode_tally_status()
    assert "1 pass(es) in 2.0s" in line
    assert "1 served from cache" in line
    assert "(+1 cached)" in line


def test_the_share_of_the_run_is_reported(refiner):
    refiner._v2_tally_encode("prompt", 5.0)
    assert "50% of Studio's 10.0s" in refiner._v2_encode_tally_status(total_seconds=10.0)


def test_a_run_starts_from_zero(refiner):
    """The tally is per run. A second run must not inherit the first one's numbers."""
    refiner._v2_tally_encode("prompt", 3.0)
    refiner._v2_reset_encode_tally()
    assert refiner._v2_encode_tally_status() == ""
