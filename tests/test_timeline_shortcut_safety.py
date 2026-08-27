"""A stray letter must not edit the timeline.

Reported from a rental: typing in the Composer, focus was lost, typing continued, and a clip
was silently cut in half. `s` splits, `Delete` removes — single letters, no modifier, and the
only thing separating "typing" and "editing" was where focus happened to be. Focus loss lands
on <body>, which the handler read as "nothing is focused, so this must be for the timeline".

Two guards, for two different ways the user cannot see what they are about to change:

1. A writing surface is open (the Composer, any floating window). Letters belong to it
   whether or not it currently holds focus.
2. "Show timeline on hover" is on and the timeline is collapsed. Editing a selection that is
   not on screen gives no feedback that anything happened at all.

Transport keys are deliberately NOT gated by (2): they drive the player, which is on screen
either way. They ARE gated by (1) — space belongs to whatever you are typing into.

The handler lives in a DOM-heavy module with no node harness, so these read the source. The
behaviour of `isVisible` itself is covered in timeline_peek.test.js.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FRONTEND = Path(__file__).resolve().parents[1] / "movie_editor" / "frontend"


@pytest.fixture(scope="module")
def timeline():
    return (FRONTEND / "timeline.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def handler(timeline):
    """The keydown handler body, so a guard elsewhere in the file cannot pass for one here."""
    start = timeline.index('window.addEventListener("keydown"')
    return timeline[start:start + 6000]


def test_an_open_writing_surface_stops_the_handler(handler):
    assert "if (textSurfaceOpen()) return;" in handler


def test_the_composer_counts_as_one(timeline):
    assert "window.Composer?.isOpen?.()" in timeline


def test_any_floating_window_counts(timeline):
    """The Composer is the one that bit, but every floating window is somewhere you type."""
    assert ".fw:not([hidden])" in timeline


def test_the_check_survives_a_missing_composer(timeline):
    """Simple mode has no Composer. A throw here would take the whole handler down with it,
    which is worse than the bug."""
    body = timeline[timeline.index("function textSurfaceOpen()"):]
    body = body[:body.index("\n  }")]
    assert "try {" in body and "catch" in body


def test_clip_edits_need_a_visible_timeline(handler):
    assert "const timelineShowing = window.TimelinePeek?.isVisible?.() !== false;" in handler
    assert handler.count("if (!timelineShowing) return;") >= 3


def test_delete_is_gated_on_visibility(handler):
    """The most destructive key, and the one with the least feedback when it lands on a
    selection nobody can see."""
    block = handler[handler.index('if (e.key === "Delete"'):]
    assert block.index("if (!timelineShowing) return;") < block.index("selectedOverlayId")


def test_transport_keys_stay_live_when_the_timeline_is_collapsed(handler):
    """They drive the player, which is on screen either way. Gating them would make peek mode
    feel broken for the thing it exists to give room to."""
    for key in ('"j"', '"k"', '"l"', '"ArrowLeft"', '"ArrowRight"'):
        line = next(ln for ln in handler.splitlines() if f"e.key === {key}" in ln)
        assert "timelineShowing" not in line


def test_the_inline_field_blur_does_not_reach_past_its_own_surface(handler):
    """A number field on a clip blurs itself so i/o/s reach the clip. That is right for a
    clip, and wrong while the Composer is open — it would hand the keystroke to the timeline
    from a window the user is typing in."""
    blur = handler[:handler.index("a.blur();")]
    assert "!textSurfaceOpen()" in blur


def test_split_still_works_normally(handler):
    """The guards must not have neutered the shortcut. With nothing open and the timeline on
    screen, `s` still splits."""
    assert re.search(r'e\.key === "s" \|\| e\.key === "S"', handler)
    assert "splitSelectedAtPlayhead()" in handler
