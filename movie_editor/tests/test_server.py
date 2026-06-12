"""Movie Editor server helpers."""

from pathlib import Path

from movie_editor.backend import bridge
from movie_editor.backend.timeline import Project, Scene
from movie_editor.server import (
    _build_render_filter,
    _clip_bytes_for_media,
    _clip_needs_trim,
    _has_graphics_export_content,
    _parse_has_scenes,
    _parse_prompt_variants,
    _playback_render_from_query,
    _project_models,
    _resolve_run_seed,
    _run_sampler_inputs,
    _run_studio_inputs,
    _scene_playback_clip_spec,
    _shortcut_seed,
    _timeline_duration_sec,
)

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


def test_identity_pin_media_refs_without_mixed_anchors():
    import json
    from movie_editor.server import _attach_scene_anchors

    full = _project(
        continuity_settings={"identity_pin_ref": "pin1"},
        scenes=[{"id": "s1", "text": "a"}, {"id": "s2", "text": "b", "source": {"type": "carry"}}],
    )
    samp = _run_sampler_inputs(full, 2, full=full)
    assert "funpack_scene_guides" in samp
    media_pack = {
        "primary": None,
        "by_scene": {"_ref_pin1": {"filename": "funpack_movie_pin1.png", "media_ref": "pin1"}},
    }
    _attach_scene_anchors(samp, media_pack, full)
    refs = json.loads(samp["funpack_scene_media_refs"])
    assert refs["pin1"] == "funpack_movie_pin1.png"
    assert "funpack_scene_anchors" not in samp


def test_prepare_media_extra_refs_do_not_become_primary(monkeypatch, tmp_path):
    from movie_editor import server
    from movie_editor.backend import config, media

    indir = tmp_path / "input"
    indir.mkdir()
    mediadir = tmp_path / "media"
    mediadir.mkdir()
    (mediadir / "pin1.png").write_bytes(b"fake")

    monkeypatch.setattr(config, "MEDIA_DIR", mediadir)
    monkeypatch.setattr(media, "path_for", lambda mid: mediadir / "pin1.png" if mid == "pin1" else None)

    class FP:
        @staticmethod
        def get_input_directory():
            return str(indir)

    monkeypatch.setitem(__import__("sys").modules, "folder_paths", FP)

    p = _project(scenes=[{"id": "s1", "text": "a", "source": {"type": "carry"}}])
    pack = server._prepare_media(p, ["pin1"])
    assert pack is not None
    assert pack["primary"] is None
    assert pack["by_scene"]["_ref_pin1"]["filename"].startswith("funpack_movie_pin1")


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


def test_resolve_comfy_media_path_temp(monkeypatch, tmp_path):
    from movie_editor import server

    temp = tmp_path / "temp"
    temp.mkdir()
    clip = temp / "clip.mp4"
    clip.write_bytes(b"x")

    class FP:
        @staticmethod
        def get_output_directory():
            return str(tmp_path / "out")

        @staticmethod
        def get_temp_directory():
            return str(temp)

    monkeypatch.setitem(__import__("sys").modules, "folder_paths", FP)
    assert server._resolve_comfy_media_path("clip.mp4", "", "temp") == str(clip)
    assert server._resolve_comfy_media_path("nope.mp4", "", "temp") is not None


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


def test_generate_filters_video_clips():
    from movie_editor.backend.timeline import Project, Scene, SceneSource, is_video_clip
    from movie_editor import server

    assert server.is_video_clip is is_video_clip
    p = Project(scenes=[
        Scene(text="gen", source=SceneSource(type="carry")),
        Scene(text="", source=SceneSource(type="video", media_ref="v1")),
    ])
    active = [s for s in p.scenes if not s.excluded and not is_video_clip(s)]
    assert len(active) == 1
    assert active[0].text == "gen"


def test_clip_bytes_copy_from_media_bin(tmp_path, monkeypatch):
    from movie_editor.backend import config, media

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "MEDIA_DIR", tmp_path / "media")
    entry = media.save_upload("take.mp4", b"MP4DATA")
    assert _clip_needs_trim({"bin_media_ref": entry["id"]}) is False
    assert _clip_bytes_for_media({"bin_media_ref": entry["id"]}) == b"MP4DATA"
    assert _clip_needs_trim({"bin_media_ref": entry["id"], "in": 1.0}) is True


def test_scene_playback_clip_spec_chain_offsets():
    p = _project(scenes=[
        {"id": "s1", "text": "a"},
        {"id": "s2", "text": "b", "source_in": 0.5},
        {"id": "s3", "text": "c", "source_dur": 2.5},
    ], frame_rate=25, num_frames_per_scene=97)
    p.scene_renders = {
        "s1": {"inSec": 0, "media": {"filename": "chain.mp4", "subfolder": "", "type": "output"}},
        "s2": {"inSec": 3.88, "media": {"filename": "chain.mp4", "subfolder": "", "type": "output"}},
        "s3": {"inSec": 7.4, "media": {"filename": "chain.mp4", "subfolder": "", "type": "output"}},
    }
    c1 = _scene_playback_clip_spec(p, "s1")
    c2 = _scene_playback_clip_spec(p, "s2")
    c3 = _scene_playback_clip_spec(p, "s3")
    assert c1["in"] == 0
    assert c1["dur"] == 97 / 25
    assert c2["in"] == 3.88 + 0.5
    assert c3["in"] == 7.4
    assert c3["dur"] == 2.5
    assert c1["filename"] == c2["filename"] == c3["filename"] == "chain.mp4"


def test_scene_playback_clip_spec_live_render_query():
    p = _project(scenes=[{"id": "s1", "text": "a"}, {"id": "s2", "text": "b", "source_in": 0.5}])
    p.scene_renders = {}
    override = _playback_render_from_query({
        "filename": "fresh.mp4",
        "subfolder": "out",
        "type": "output",
        "render_in": "3.88",
    })
    c = _scene_playback_clip_spec(p, "s2", render_override=override)
    assert c["filename"] == "fresh.mp4"
    assert c["subfolder"] == "out"
    assert c["in"] == 3.88 + 0.5
    assert _playback_render_from_query({}) is None


def test_project_models_keeps_empty_slots(monkeypatch):
    from movie_editor.backend import nodes

    monkeypatch.setattr(nodes, "load_models", lambda: {"slots": [{"id": "global", "role": "unet"}]})
    p = _project(scenes=[{"id": "s1", "text": "a"}])
    p.models = {"slots": [], "full_control": False}
    assert _project_models(p)["slots"] == []
    assert _project_models(p)["slots"] != nodes.load_models()["slots"]


def test_timeline_duration_sec_from_overlays_and_audio():
    p = _project(scenes=[])
    p.scenes = []
    p.overlay_tracks = [{"start_sec": 2, "duration_sec": 5}]
    p.audio_tracks = [{"start_sec": 10, "source_dur": 3}]
    assert _timeline_duration_sec(p) == 13
    assert _has_graphics_export_content(p) is True


def test_build_render_filter_blank_canvas():
    filt, has_audio = _build_render_filter(
        [],
        tracks=[{"start_sec": 0, "volume": 1.0}],
        keep_original=False,
        base_input=1,
        blank_canvas={"w": 768, "h": 512, "fps": 25, "dur": 8.0},
    )
    assert "[0:v]format=yuv420p,setsar=1[vbase]" in filt
    assert "[aout]" in filt
    assert has_audio is True


def test_timeline_snapshot_mismatch_detects_stale_anchor():
    from movie_editor.backend.timeline import Scene, SceneSource
    from movie_editor.server import _timeline_snapshot_mismatch

    p = Project(
        name="t",
        anchor="new anchor",
        scenes=[
            Scene(id="s1", text="scene one", source=SceneSource(type="image", media_ref="img_b")),
            Scene(id="s2", text="scene two", source=SceneSource(type="carry")),
        ],
    )
    snap = {
        "anchor": "new anchor",
        "scenes": [
            {"id": "s1", "text": "scene one", "source": {"type": "image", "media_ref": "img_a"}},
            {"id": "s2", "text": "scene two", "source": {"type": "carry", "media_ref": None}},
        ],
    }
    assert _timeline_snapshot_mismatch(snap, p) == "scene s1 anchor image"


def test_timeline_snapshot_mismatch_none_when_aligned():
    from movie_editor.backend.timeline import Scene, SceneSource
    from movie_editor.server import _timeline_snapshot_mismatch

    p = Project(
        name="t",
        anchor="anchor",
        scenes=[
            Scene(id="s1", text="one", source=SceneSource(type="image", media_ref="img_a")),
            Scene(id="s2", text="two", source=SceneSource(type="carry")),
        ],
    )
    snap = {
        "anchor": "anchor",
        "scenes": [
            {"id": "s1", "text": "one", "source": {"type": "image", "media_ref": "img_a"}},
            {"id": "s2", "text": "two", "source": {"type": "carry", "media_ref": None}},
        ],
    }
    assert _timeline_snapshot_mismatch(snap, p) is None