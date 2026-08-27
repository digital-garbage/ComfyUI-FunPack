"""The settings card prints what produced the video, not every widget that exists.

Reported from a rental: too much on the card to read, including a region-sharpening prompt
printed under a sharpener that was switched off. A value left at its default produced
nothing, and a value belonging to a disabled feature produced nothing either — but printed
side by side with the handful that DID decide the render, they read as if they had.

Two rules: skip a value equal to the widget's own default, and skip a value whose feature
is off.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movie_editor.backend import settings_card as sc

FRONTEND = Path(__file__).resolve().parents[1] / "movie_editor" / "frontend"


@pytest.fixture
def defaults(monkeypatch):
    """Stand in for the node's declared defaults, so these tests describe the RULE."""
    table = {
        "segmented_detailing": False, "detail_targets": "hands, face",
        "context_windows": False, "context_window_overlap": 40,
        "joyai_memory": False, "joyai_audio_memory": False, "v2a_grad_scale": 1.0,
        "h3_gain_mode": "learned", "h3_video_detail": 1.0,
        "cfg": 1.0, "frame_overlap": 8,
    }
    monkeypatch.setattr(sc, "_node_defaults", lambda _cls: table)
    return table


def rows(values, defaults_unused=None):
    return dict(sc._live_rows(values, "FunPackLTXAVSceneChainSampler"))


# ── rule 1: a default decided nothing ───────────────────────────────────────

def test_a_value_left_at_its_default_is_not_printed(defaults):
    assert rows({"cfg": 1.0}) == {}


def test_a_changed_value_is_printed(defaults):
    assert "cfg" in rows({"cfg": 3.5})


def test_a_value_the_node_does_not_declare_is_kept(defaults):
    """Not every key on the card comes from a widget. Unknown means unjudgeable, and
    dropping it would hide something the card was asked to show."""
    assert "mystery" in rows({"mystery": 7})


# ── rule 2: an off feature decided nothing ──────────────────────────────────

def test_a_setting_under_a_disabled_toggle_is_not_printed(defaults):
    """The one that was reported: a region-sharpening prompt under a disabled sharpener."""
    assert rows({"detail_targets": "the hands, the face"}) == {}


def test_the_same_setting_is_printed_once_its_feature_is_on(defaults):
    got = rows({"segmented_detailing": True, "detail_targets": "the hands, the face"})
    assert got["detail_targets"] == "the hands, the face"
    assert "segmented_detailing" in got


def test_a_default_value_under_an_enabled_toggle_is_still_dropped(defaults):
    """The two rules are independent: on does not make a default worth printing."""
    got = rows({"context_windows": True, "context_window_overlap": 40})
    assert "context_window_overlap" not in got


def test_a_chain_of_owners_is_walked_to_the_end(defaults):
    """v2a_grad_scale belongs to joyai_audio_memory, which belongs to joyai_memory. An
    owner that is itself off takes everything under it with it."""
    assert rows({"joyai_audio_memory": True, "v2a_grad_scale": 0.5}) == {}
    got = rows({"joyai_memory": True, "joyai_audio_memory": True, "v2a_grad_scale": 0.5})
    assert "v2a_grad_scale" in got


def test_an_owner_matched_by_value_not_by_truthiness(defaults):
    """h3_gain_mode is a combo: 'manual' switches its dials on, 'learned' does not — and
    'learned' is a perfectly truthy string."""
    assert rows({"h3_video_detail": 1.4}) == {}
    assert "h3_video_detail" in rows({"h3_gain_mode": "manual", "h3_video_detail": 1.4})


def test_a_cycle_in_the_table_does_not_hang(monkeypatch, defaults):
    monkeypatch.setitem(sc._OWNED_BY, "a", ("b", None))
    monkeypatch.setitem(sc._OWNED_BY, "b", ("a", None))
    assert sc._live_rows({"a": 1, "b": 1}, "X") is not None


# ── the table matches the Editor it was copied from ─────────────────────────

def test_the_ownership_table_is_the_editors_own():
    """The Editor hides a knob whose feature is off; the card must hide the same ones. Two
    copies of one rule in two languages, so the test regenerates this one from that one."""
    src = (FRONTEND / "engine_settings.js").read_text(encoding="utf-8")
    found = {}
    for name, body in re.findall(r'\{\s*name:\s*"([a-z0-9_]+)"(.*?)\n\s*(?=\{ name:|\];)',
                                 src, re.S):
        m = re.search(r'dependsOn:\s*"([a-z0-9_]+)"(?:,\s*dependsValue:\s*"([^"]*)")?', body)
        if m:
            found[name] = (m.group(1), m.group(2))
    assert found, "no dependsOn rows parsed — the parser, not the table, is what broke"
    assert found == sc._OWNED_BY


# ── the sections it feeds ───────────────────────────────────────────────────

def test_an_empty_section_is_dropped_rather_than_printed_bare(defaults):
    out = sc._sampling_sections({}, {"cfg": 1.0})
    assert not [s for s in out if s["title"] == "Chain Sampler"]


def test_a_section_with_something_to_say_survives(defaults):
    out = sc._sampling_sections({}, {"cfg": 3.5})
    chain = next(s for s in out if s["title"] == "Chain Sampler")
    assert chain["rows"] == [("cfg", "3.5")]


def test_defaults_are_read_off_the_node_not_copied_here():
    """A default copied into this file is one more thing to keep in step with the node."""
    import inspect
    src = inspect.getsource(sc._node_defaults)
    assert "INPUT_TYPES()" in src
