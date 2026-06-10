"""Movie Editor server helpers."""

from pathlib import Path

from movie_editor.backend.timeline import Project, Scene
from movie_editor.server import _run_studio_inputs

CONTINUE = "__funpack_continue__"


def _project(scenes, rating=""):
    return Project(
        name="t",
        scenes=[Scene(id="s1", text="a", rating=rating)],
        conditioning_slot="funpack",
    )


def test_studio_inputs_user_rating():
    p = _project([], rating="Perfect")
    si = _run_studio_inputs(p, p.scenes)
    assert si["rating"] == "Perfect"


def test_studio_inputs_continue_when_unrated():
    p = _project([])
    si = _run_studio_inputs(p, p.scenes)
    assert si["rating"] == CONTINUE


def test_continue_rating_is_valid_studio_combo_value():
    """ComfyUI rejects /prompt overrides not in the node's rating combo list."""
    root = Path(__file__).resolve().parents[2]
    src = (root / "conditioning.py").read_text(encoding="utf-8")
    assert 'MOVIE_EDITOR_CONTINUE_RATING = "__funpack_continue__"' in src
    assert "MOVIE_EDITOR_CONTINUE_RATING]" in src


def test_studio_inputs_skips_custom_conditioning():
    p = _project([])
    p.conditioning_slot = "custom"
    assert _run_studio_inputs(p, p.scenes) == {}