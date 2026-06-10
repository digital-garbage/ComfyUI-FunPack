"""Movie Editor server helpers."""

from pathlib import Path

from movie_editor.backend.timeline import Project, Scene
from movie_editor.server import _run_sampler_inputs, _run_studio_inputs

CONTINUE = "__funpack_continue__"


def _project(scenes, rating="", **kw):
    p = Project(
        name="t",
        scenes=[Scene.from_dict(s) for s in scenes],
        conditioning_slot="funpack",
        sampler_slot="funpack",
        **{k: v for k, v in kw.items() if k != "scenes"},
    )
    if not scenes:
        p.scenes = [Scene(id="s1", text="a", rating=rating)]
    elif rating:
        p.scenes[0].rating = rating
    return p


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


def test_mixed_solo_sampler_inputs():
    full = _project(scenes=[
        {"id": "s1", "text": "a", "source": {"type": "image", "media_ref": "img1"}},
        {"id": "s2", "text": "b", "source": {"type": "mixed", "media_ref": "img2"}},
    ])
    solo = Project.from_dict(full.to_dict())
    solo.scenes = [full.scenes[1]]
    samp = _run_sampler_inputs(solo, 1, full=full)
    assert samp["frame_overlap"] == 0
    assert samp["carry_i2v_guides"] is False
    import json
    guides = json.loads(samp["funpack_scene_guides"])
    assert guides["scenes"][0][0]["media_ref"] == "img1"


def test_studio_inputs_skips_custom_conditioning():
    p = _project([])
    p.conditioning_slot = "custom"
    assert _run_studio_inputs(p, p.scenes) == {}