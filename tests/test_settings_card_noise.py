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
    """Stand in for the node's declared inputs, so these tests describe the RULE."""
    table = {
        "segmented_detailing": False, "detail_targets": "hands, face",
        "context_windows": False, "context_window_overlap": 40,
        "joyai_memory": False, "joyai_audio_memory": False, "v2a_grad_scale": 1.0,
        "h3_video_detail": 1.0,
        "cfg": 1.0, "frame_overlap": 8,
    }
    declared = frozenset(table) | {"mystery", "adjustments"}
    monkeypatch.setattr(sc, "_node_inputs", lambda _cls: (declared, table))
    return table


def rows(values, defaults_unused=None):
    return dict(sc._live_rows(values, "FunPackLTXAVSceneChainSampler"))


# ── rule 1: a default decided nothing ───────────────────────────────────────

def test_a_value_left_at_its_default_is_not_printed(defaults):
    assert rows({"cfg": 1.0}) == {}


def test_a_changed_value_is_printed(defaults):
    assert "cfg" in rows({"cfg": 3.5})


def test_a_declared_key_with_no_default_is_kept(defaults):
    assert "mystery" in rows({"mystery": 7})


def test_a_key_the_node_does_not_declare_is_dropped(defaults):
    """Reported from a rental: the card showed `block_steer` and a `steps` value. Neither is
    an input on the Chain Sampler — block_steer was removed from the pack entirely — so
    ComfyUI never passes them to anything. Printed beside live settings they read as live."""
    assert rows({"block_steer": True, "steps": 40}) == {}


def test_an_empty_value_decided_nothing_either(defaults):
    assert rows({"adjustments": [], "mystery": ""}) == {}


def test_the_builders_private_keys_are_not_settings(defaults):
    assert rows({"_movie_editor_scene_ratings": {"a": 1}}) == {}


# ── the studio_settings blob, which has no node to ask ──────────────────────

def test_a_refiner_value_at_its_editor_default_is_dropped():
    assert sc._live_rows({"temporal_style": "natural", "steer_mode": "relative"},
                         defaults=sc._EDITOR_DEFAULTS) == []


def test_a_changed_refiner_value_survives():
    got = dict(sc._live_rows({"temporal_style": "slow_motion"},
                             defaults=sc._EDITOR_DEFAULTS))
    assert got["temporal_style"] == "slow_motion"


def test_an_unrecognised_refiner_key_is_kept():
    """No node declares these, so unknown means unjudgeable — not inert. Dropping it would
    hide something rather than tidy it."""
    got = dict(sc._live_rows({"some_future_setting": True}, defaults=sc._EDITOR_DEFAULTS))
    assert "some_future_setting" in got


def test_a_key_for_a_feature_that_no_longer_exists_is_dropped():
    """The exception to the rule above. prompt_repair and prefer_wired_conditioning are read
    by nothing at all — they only survive in old studio_settings blobs. There is no node to
    check the refiner block against, so these are named rather than derived."""
    assert sc._live_rows({"prompt_repair": True, "prefer_wired_conditioning": True},
                         defaults=sc._EDITOR_DEFAULTS) == []


def test_an_owner_live_for_more_than_one_value_is_matched_on_any_of_them():
    """absolute_strength belongs to steer_mode and is live for BOTH "absolute" and "both".
    Reading only the singular dependsValue missed the plural dependsVals the Studio rows
    use, so it printed under "relative", where it does nothing."""
    assert sc._OWNED_BY["absolute_strength"] == ("steer_mode", ("absolute", "both"))
    rows = dict(sc._live_rows({"steer_mode": "relative", "absolute_strength": 0.05},
                              defaults=sc._EDITOR_DEFAULTS))
    assert "absolute_strength" not in rows
    for mode in ("absolute", "both"):
        got = dict(sc._live_rows({"steer_mode": mode, "absolute_strength": 0.05},
                                 defaults=sc._EDITOR_DEFAULTS))
        assert "absolute_strength" in got, mode


def test_a_boolean_owner_survives_the_journey_from_javascript():
    """The Editor writes JS literals, so a boolean owner arrives as "true" while Python's
    own str(True) is "True"."""
    assert sc._OWNED_BY["negative_erase_mode"] == ("negative_erase", ("true",))
    got = dict(sc._live_rows({"negative_erase": True, "negative_erase_mode": "subtract"},
                             defaults=sc._EDITOR_DEFAULTS))
    assert got["negative_erase_mode"] == "subtract"


def test_a_knob_the_editor_does_not_offer_still_gets_an_owner():
    """Two reasons a knob has no dependsOn to find: it is rendered by hand (identity_projector),
    or the Editor no longer offers it at all while a raw ComfyUI graph still can
    (segmented detailing). The card has to hide both when their feature is off."""
    assert sc._OWNED_BY["identity_projector"] == ("identity_transfer_enabled", None)
    assert sc._OWNED_BY["detail_targets"] == ("segmented_detailing", None)


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


def test_an_owner_can_be_matched_by_value_not_just_truthiness(monkeypatch, defaults):
    """A combo owner switches its dials on for ONE of its values, and the others are
    perfectly truthy strings. No shipped knob uses this today; the rule still has to."""
    monkeypatch.setitem(sc._OWNED_BY, "h3_video_detail", ("mode", ("manual",)))
    monkeypatch.setitem(defaults, "mode", "learned")
    assert rows({"h3_video_detail": 1.4}) == {}
    assert "h3_video_detail" in rows({"mode": "manual", "h3_video_detail": 1.4})


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
    for name, body in re.findall(r'\{\s*name:\s*"([a-z0-9_.]+)"(.*?)\n\s*(?=\{ name:|\];)',
                                 src, re.S):
        m = re.search(r'dependsOn:\s*"([a-z0-9_]+)"', body)
        if not m:
            continue
        one = re.search(r'dependsValue:\s*"([^"]*)"', body)
        many = re.search(r'dependsVals:\s*\[([^\]]*)\]', body)
        if one:
            vals = (one.group(1),)
        elif many:
            vals = tuple(v.strip().strip('"') for v in many.group(1).split(",") if v.strip())
        else:
            vals = None
        found[name] = (m.group(1), vals)
    assert found, "no dependsOn rows parsed — the parser, not the table, is what broke"
    assert {**found, **sc._OWNED_EXTRA} == sc._OWNED_BY


def test_the_defaults_table_is_the_editors_own():
    src = (FRONTEND / "engine_settings.js").read_text(encoding="utf-8")
    found = {}
    for name, body in re.findall(r'\{\s*name:\s*"([a-z0-9_.]+)"(.*?)\n\s*(?=\{ name:|\];)',
                                 src, re.S):
        m = re.search(r'default:\s*(true|false|-?[\d.]+|"[^"]*")', body)
        if m:
            raw = m.group(1)
            found[name] = {"true": True, "false": False}.get(raw) if raw in ("true", "false") \
                else (raw.strip('"') if raw.startswith('"') else float(raw))
    assert found, "no default rows parsed"
    for name, value in found.items():
        assert name in sc._EDITOR_DEFAULTS, name
        assert float(sc._EDITOR_DEFAULTS[name]) == value if isinstance(value, float) \
            else sc._EDITOR_DEFAULTS[name] == value, name


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
    src = inspect.getsource(sc._node_inputs)
    assert "INPUT_TYPES()" in src
