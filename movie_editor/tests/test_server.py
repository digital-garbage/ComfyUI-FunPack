"""Movie Editor server helpers."""

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


def test_studio_inputs_skips_custom_conditioning():
    p = _project([])
    p.conditioning_slot = "custom"
    assert _run_studio_inputs(p, p.scenes) == {}