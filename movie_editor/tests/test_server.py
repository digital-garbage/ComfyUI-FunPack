"""Movie Editor server helpers."""

from pathlib import Path

from movie_editor.backend import bridge
from movie_editor.backend.timeline import Project, Scene
from movie_editor.server import _parse_has_scenes, _parse_prompt_variants, _resolve_run_seed, _run_sampler_inputs, _run_studio_inputs, _shortcut_seed

CONTINUE = "__funpack_continue__"
FRESH = "__funpack_fresh_prompt__"


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


def test_studio_inputs_multi_scene_ratings():
    p = _project(scenes=[
        {"id": "s1", "text": "a", "rating": "Perfect"},
        {"id": "s2", "text": "b", "rating": "Missing action"},
        {"id": "s3", "text": "c"},
    ])
    active = p.scenes
    si = _run_studio_inputs(p, active)
    assert si["rating"] == CONTINUE
    assert si["_movie_editor_scene_ratings"] == [
        {"index": 0, "rating": "Perfect"},
        {"index": 1, "rating": "Missing action"},
    ]


def test_studio_inputs_single_scene_in_chain_keeps_global_rating():
    p = _project(scenes=[{"id": "s1", "text": "a", "rating": "Nailed it"}])
    si = _run_studio_inputs(p, p.scenes)
    assert si["rating"] == "Nailed it"
    assert "_movie_editor_scene_ratings" not in si


def test_studio_inputs_continue_when_unrated():
    p = _project([])
    si = _run_studio_inputs(p, p.scenes)
    assert si["rating"] == CONTINUE


def test_studio_inputs_fresh_prompt_when_changed():
    p = _project([])
    si = _run_studio_inputs(p, p.scenes, prompt_changed=True)
    assert si["rating"] == FRESH


def test_continue_rating_is_valid_studio_combo_value():
    """ComfyUI rejects /prompt overrides not in the node's rating combo list."""
    root = Path(__file__).resolve().parents[2]
    src = (root / "conditioning.py").read_text(encoding="utf-8")
    assert 'MOVIE_EDITOR_CONTINUE_RATING = "__funpack_continue__"' in src
    assert "MOVIE_EDITOR_CONTINUE_RATING" in src
    assert 'MOVIE_EDITOR_FRESH_PROMPT_RATING = "__funpack_fresh_prompt__"' in src
    assert "MOVIE_EDITOR_FRESH_PROMPT_RATING" in src
    assert "V2_RATING_LABELS" in src


def test_image_solo_sampler_inputs_no_prior_guides():
    full = _project(scenes=[
        {"id": "s1", "text": "a", "source": {"type": "carry"}},
        {"id": "s2", "text": "b", "source": {"type": "image", "media_ref": "img2"}},
    ])
    solo = Project.from_dict(full.to_dict())
    solo.scenes = [full.scenes[1]]
    samp = _run_sampler_inputs(solo, 1, full=full)
    assert samp["frame_overlap"] == 0
    assert samp["carry_i2v_guides"] is False
    assert "funpack_scene_guides" not in samp


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


def test_carry_chain_auto_mid_scene_guide():
    import json
    full = _project(scenes=[
        {"id": "s1", "text": "a"},
        {"id": "s2", "text": "b", "source": {"type": "carry"}},
    ])
    samp = _run_sampler_inputs(full, 2, full=full)
    assert samp["mid_scene_guide"] is True
    assert samp["carry_i2v_guides"] is False
    guides = json.loads(samp["funpack_scene_guides"])
    assert guides["scenes"][1][0]["source"] == "template"


def test_studio_inputs_skips_custom_conditioning():
    p = _project([])
    p.conditioning_slot = "custom"
    assert _run_studio_inputs(p, p.scenes) == {}


def test_shortcut_seed_uses_sampler_inputs():
    p = _project([], sampler_inputs={"seed": 4242})
    assert _shortcut_seed(p) == 4242


def test_shortcut_seed_random_when_unset():
    p = _project([])
    assert _shortcut_seed(p) == 0


def test_resolve_run_seed_fixed():
    p = _project([], sampler_inputs={"seed": 99})
    assert _resolve_run_seed(p) == 99


def test_resolve_run_seed_random_when_unset():
    p = _project([])
    seeds = {_resolve_run_seed(p) for _ in range(8)}
    assert len(seeds) > 1
    assert all(1 <= s <= 0xFFFFFFFFFFFFFFFF for s in seeds)


def test_parse_has_scenes():
    assert _parse_has_scenes({"scenes": [{"text": "a"}]})
    assert not _parse_has_scenes({"scenes": []})
    assert not _parse_has_scenes(None)


def test_parse_prompt_variants_partial_success(monkeypatch):
    def ok(_prompt, seed=0):
        return {"anchor": "", "scenes": [{"index": 0, "text": "a"}], "transitions": []}

    def boom(_prompt):
        raise RuntimeError("verbatim blew up")

    monkeypatch.setattr(bridge, "parse_timeline", ok)
    monkeypatch.setattr(bridge, "parse_timeline_raw", ok)
    monkeypatch.setattr(bridge, "parse_timeline_verbatim", boom)

    payload, errors = _parse_prompt_variants("scene one", seed=0)
    assert _parse_has_scenes(payload["parsed_raw"])
    assert errors == {"parsed_verbatim": "RuntimeError: verbatim blew up"}


def test_parse_prompt_variants_all_fail(monkeypatch):
    def boom(*_args, **_kwargs):
        raise KeyError()

    monkeypatch.setattr(bridge, "parse_timeline", boom)
    monkeypatch.setattr(bridge, "parse_timeline_raw", boom)
    monkeypatch.setattr(bridge, "parse_timeline_verbatim", boom)

    payload, errors = _parse_prompt_variants("scene one", seed=0)
    assert payload["parsed"] is None
    assert len(errors) == 3
    assert errors["parsed"] == "KeyError (no message)"