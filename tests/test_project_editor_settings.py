"""Editor preferences riding along in the project file.

Autocomplete, shortcut ideas, the anchor toggle, the custom-i2v bypass and the shortcut
revolver are all per-MACHINE: the first four live in the browser's localStorage and the
revolver in a server-side sidecar next to the shortcut DB. Renting a fresh GPU box gives
you neither, so every one of them silently reverts to its default and has to be set again.
`Project.editor_settings` is what carries them across, so the invariants worth pinning are:

1. It survives a save/load round trip of the project file at all (the Project dataclass is
   a whitelist — an unknown key is dropped, not stored).
2. It is optional. A project written before this existed loads with an empty dict, never a
   missing attribute and never a set of defaults that would overwrite a configured browser.
3. Nothing in the generation path reads it — it is editor preference, not a render setting.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movie_editor.backend.timeline import Project  # noqa: E402


SETTINGS = {
    "autocomplete": False,
    "suggestions": True,
    "anchorEnabled": False,
    "anchorGuideHasI2v": True,
    "anchorGuideBypass": {"node": "slot_1", "input": "strength", "value": 0.0},
    "revolver": {"enabled": True, "random": True},
}


def test_editor_settings_survive_a_project_round_trip():
    p = Project.from_dict({"name": "trip", "editor_settings": SETTINGS})
    assert p.editor_settings == SETTINGS
    # to_dict -> from_dict is exactly what save/load does on disk.
    assert Project.from_dict(p.to_dict()).editor_settings == SETTINGS


def test_a_project_saved_before_this_existed_carries_nothing():
    """An ABSENT key must mean 'leave this browser alone'. If old projects loaded a full
    set of defaults instead, opening one would quietly switch off a revolver you had on."""
    p = Project.from_dict({"name": "old"})
    assert p.editor_settings == {}
    assert Project.from_dict(p.to_dict()).editor_settings == {}


@pytest.mark.parametrize("bad", [None, "", 0, []])
def test_a_malformed_value_degrades_to_empty_rather_than_raising(bad):
    assert Project.from_dict({"name": "x", "editor_settings": bad}).editor_settings == {}


def test_two_projects_keep_their_own_preferences():
    a = Project.from_dict({"name": "a", "editor_settings": {"anchorEnabled": True}})
    b = Project.from_dict({"name": "b", "editor_settings": {"anchorEnabled": False}})
    assert a.editor_settings["anchorEnabled"] is True
    assert b.editor_settings["anchorEnabled"] is False


def test_generation_never_reads_editor_settings():
    """It is a preference, not a render setting: the build path must not branch on it, or
    two machines with different editor prefs would generate different video."""
    src = (Path(__file__).resolve().parents[1] / "movie_editor" / "backend" / "builder.py").read_text()
    assert "editor_settings" not in src
