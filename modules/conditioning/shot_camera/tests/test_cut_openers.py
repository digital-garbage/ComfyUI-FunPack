"""Every cut opener is a hard cut that reads as a cut, never as a camera move."""

import re

from .. import engine


def test_every_opener_is_a_hard_cut_and_no_opener_reads_as_a_camera_move():
    for opener in engine.CUT_OPENERS:
        assert not engine.CAMERA.search(opener), opener
        assert not any(w in opener for w in ("dissolve", "fade", "wipe")), opener
        assert engine.CUT.search(opener), opener
        assert re.search(r"\b(?:cuts|transitions|changes|switches) to\b", opener), opener

