"""Every cut opener is a hard cut that reads as a cut, never as a camera move."""

import random
import re

from .. import engine


def test_every_opener_is_a_hard_cut_and_no_opener_reads_as_a_camera_move():
    for opener in engine.CUT_OPENERS:
        assert not engine.CAMERA.search(opener), opener
        assert not any(w in opener for w in ("dissolve", "fade", "wipe")), opener
        assert engine.CUT.search(opener), opener
        assert re.search(r"\b(?:cuts|transitions|changes|switches) to\b", opener), opener



def test_the_transition_slider_at_zero_is_always_a_plain_cut_and_at_one_always_a_mixed_transition():
    for seed in range(200):
        opener, kind = engine._opener(random.Random(seed), 0.0, None)
        assert kind is None and opener in engine.CUT_OPENERS
    last, kinds = None, []
    for seed in range(300):
        opener, last = engine._opener(random.Random(seed), 1.0, last)
        assert last is not None, "1 is never a plain cut"
        kinds.append(last)
    assert len(set(kinds)) == 3, "a mixture of dissolve, fade and wipe"
    assert all(a != b for a, b in zip(kinds, kinds[1:])), "never the same kind twice running"
