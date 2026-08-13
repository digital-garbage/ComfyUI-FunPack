"""Movie Editor server helpers."""

from pathlib import Path

import pytest
from aiohttp import web

import movie_editor.server as srv
from movie_editor.backend import bridge
from movie_editor.backend.timeline import (
    Project, Scene, build_combined_prompt, build_generation_scene_segments,
)
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


def test_temp_kind_classifies_media_and_skips_other():
    assert srv._temp_kind(".mp4") == "video"
    assert srv._temp_kind(".PNG") == "image"
    assert srv._temp_kind(".wav") == "audio"
    assert srv._temp_kind(".txt") is None
    assert srv._temp_kind(".json") is None


def test_list_temp_media_lists_media_newest_first(tmp_path, monkeypatch):
    (tmp_path / "old.png").write_bytes(b"x")
    (tmp_path / "new.mp4").write_bytes(b"yy")
    (tmp_path / "notes.txt").write_text("skip me")
    sub = tmp_path / "previews"
    sub.mkdir()
    (sub / "clip.webm").write_bytes(b"zzz")
    import os
    os.utime(tmp_path / "old.png", (1000, 1000))
    os.utime(tmp_path / "new.mp4", (2000, 2000))
    os.utime(sub / "clip.webm", (1500, 1500))

    fake = type("FP", (), {"get_temp_directory": staticmethod(lambda: str(tmp_path))})
    monkeypatch.setitem(__import__("sys").modules, "folder_paths", fake)

    files = srv._list_temp_media()
    names = [f["filename"] for f in files]
    assert "notes.txt" not in names
    assert names == ["new.mp4", "clip.webm", "old.png"]  # newest mtime first
    clip = next(f for f in files if f["filename"] == "clip.webm")
    assert clip["subfolder"] == "previews"
    assert clip["kind"] == "video"
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


def test_corrupt_project_surfaces_clean_error_not_500(monkeypatch):
    # A corrupt / schema-incompatible project file used to 500 with no detail on open
    # (while list_projects still lists it). _project_or_404 must turn that into a clear,
    # recoverable 4xx with a reason — never a bare unhandled crash.
    # server.py nulls its `web` when ComfyUI's server module is absent (test env); use real aiohttp.
    monkeypatch.setattr(srv, "web", web)
    def _boom(_pid):
        raise ValueError("bad schema\nsecond line of noise")
    monkeypatch.setattr(srv.projects, "get", _boom)
    with pytest.raises(web.HTTPUnprocessableEntity) as ei:
        srv._project_or_404("p1")
    reason = ei.value.reason
    assert "could not be loaded" in reason
    assert "\n" not in reason  # reason must stay a single header-safe line


def test_missing_project_still_404(monkeypatch):
    monkeypatch.setattr(srv, "web", web)
    monkeypatch.setattr(srv.projects, "get", lambda _pid: None)
    with pytest.raises(web.HTTPNotFound):
        srv._project_or_404("nope")


def test_generation_prompt_has_no_injected_scene_markers():
    # The generation prompt must be the verbatim text the user wrote — never an injected
    # 'scene N' delimiter. Boundaries travel structurally via build_generation_scene_segments.
    p = _project(scenes=[
        {"id": "s1", "text": "a red car drives down a street", "source": {"type": "image", "media_ref": "img1"}},
        {"id": "s2", "text": "the car stops at a cafe", "source": {"type": "carry"}},
        {"id": "s3", "text": "a woman exits the cafe", "source": {"type": "carry"}},
    ])
    gen = build_combined_prompt(p, for_generation=True)
    assert "scene 1" not in gen.lower()
    assert "scene 2" not in gen.lower()
    assert gen == "a red car drives down a street the car stops at a cafe a woman exits the cafe"
    # boundaries are carried structurally instead — one entry per generative scene
    seg = build_generation_scene_segments(p)
    assert seg["scenes"] == [
        "a red car drives down a street", "the car stops at a cafe", "a woman exits the cafe"]


def test_score_slider_sampler_inputs_pass_through():
    # FreeSliders score_slider is a plain sampler knob (like embed_guidance): the editor
    # sets it in sampler_inputs and _run_sampler_inputs must carry it through untouched
    # so builder.py can stamp it onto the Chain Sampler node.
    full = _project(
        scenes=[{"id": "s1", "text": "a"}, {"id": "s2", "text": "b", "source": {"type": "carry"}}],
        sampler_inputs={"score_slider": True, "score_slider_strength": 2.0,
                        "embed_guidance_source": "relative"},
    )
    samp = _run_sampler_inputs(full, 2, full=full)
    assert samp["score_slider"] is True
    assert samp["score_slider_strength"] == 2.0
    assert samp["embed_guidance_source"] == "relative"


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


def test_ghost_playback_clip_spec_from_query():
    """Ghost segments: the scene is gone from the project, so the trim window (in/dur)
    must come entirely from the query — and malformed/incomplete queries return None
    (the handler then 404s instead of encoding garbage)."""
    from movie_editor.server import _ghost_playback_clip_spec
    c = _ghost_playback_clip_spec({
        "filename": "chain.mp4", "subfolder": "out", "type": "output",
        "render_in": "7.4", "dur": "2.5",
    })
    assert c == {"filename": "chain.mp4", "subfolder": "out", "type": "output",
                 "in": 7.4, "dur": 2.5}
    assert _ghost_playback_clip_spec({"filename": "chain.mp4"}) is None          # no dur
    assert _ghost_playback_clip_spec({"dur": "2.5"}) is None                      # no filename
    assert _ghost_playback_clip_spec({"filename": "c.mp4", "dur": "oops"}) is None  # bad float
    # render_in defaults to 0 (ghost of the chain's first scene).
    c0 = _ghost_playback_clip_spec({"filename": "c.mp4", "dur": "4.0"})
    assert c0["in"] == 0 and c0["type"] == "output"


def test_project_models_keeps_empty_slots(monkeypatch):
    from movie_editor.backend import nodes

    monkeypatch.setattr(nodes, "load_models", lambda: {"slots": [{"id": "global", "role": "unet"}]})
    p = _project(scenes=[{"id": "s1", "text": "a"}])
    p.models = {"slots": [], "full_control": False}
    assert _project_models(p)["slots"] == []
    assert _project_models(p)["slots"] != nodes.load_models()["slots"]


# ── the project remembers which model family it is for ──────────────────────
# The whole class of bug: an H3 project whose model_family never got persisted reads back as
# LTXAV, and LTXAV's wiring routes the audio VAE to LTXVAudioVAEDecode — a node H3's core
# does not contain. Everything below guards one link in that chain.

def test_a_project_with_no_pipeline_of_its_own_inherits_the_default_family(monkeypatch):
    """Every project on disk before this fix has model_family=None. They must come back as
    the family the user actually works in, not silently as LTXAV."""
    from movie_editor.backend import nodes

    monkeypatch.setattr(nodes, "load_models", lambda: {"slots": [], "model_family": "minimax_h3"})
    p = _project(scenes=[{"id": "s1", "text": "a"}])
    p.models = {"slots": []}
    assert _project_models(p)["model_family"] == "minimax_h3"


def test_an_explicit_family_is_never_overridden_by_the_default(monkeypatch):
    from movie_editor.backend import nodes

    monkeypatch.setattr(nodes, "load_models", lambda: {"slots": [], "model_family": "minimax_h3"})
    p = _project(scenes=[{"id": "s1", "text": "a"}])
    p.models = {"slots": [], "model_family": "ltxav"}
    assert _project_models(p)["model_family"] == "ltxav"


def test_a_project_with_its_own_loaders_keeps_its_silence(monkeypatch):
    """A project that already has slots was wired under the old LTXAV assumption. Flipping it
    to H3 behind the user's back would break that wiring — the pipeline dialog now shows the
    real answer, so this stays a decision they make, not one inferred for them."""
    from movie_editor.backend import nodes

    monkeypatch.setattr(nodes, "load_models", lambda: {"slots": [], "model_family": "minimax_h3"})
    p = _project(scenes=[{"id": "s1", "text": "a"}])
    p.models = {"slots": [{"id": "a", "role": "unet"}]}
    assert _project_models(p).get("model_family") is None


def test_a_project_wired_to_ltx_only_core_nodes_is_never_flipped(monkeypatch):
    """The case that decided the hold-back rule, as a fixture rather than a live project.

    `core_overrides` on concat/audiodec means LTXVConcatAVLatent and LTXVAudioVAEDecode —
    both dropped from H3's core. Inheriting a global H3 family here would leave those
    overrides pointing at nodes the graph never builds, which is the very failure this
    whole change exists to stop.
    """
    from movie_editor.backend import nodes

    monkeypatch.setattr(nodes, "load_models", lambda: {"slots": [], "model_family": "minimax_h3"})
    p = _project(scenes=[{"id": "s1", "text": "a"}])
    p.models = {
        "slots": [],
        "links": {"audiodec": {"audio_vae": "out:x:0"}},
        "core_overrides": {"concat": {}, "audiodec": {}},
    }
    assert _project_models(p).get("model_family") is None


def test_inheriting_the_family_does_not_mutate_the_stored_project(monkeypatch):
    from movie_editor.backend import nodes

    monkeypatch.setattr(nodes, "load_models", lambda: {"slots": [], "model_family": "minimax_h3"})
    p = _project(scenes=[{"id": "s1", "text": "a"}])
    p.models = {"slots": []}
    _project_models(p)
    assert "model_family" not in p.models     # read-time fill only, not a silent write


def test_no_default_family_means_nothing_is_invented(monkeypatch):
    from movie_editor.backend import nodes

    monkeypatch.setattr(nodes, "load_models", lambda: {"slots": []})
    p = _project(scenes=[{"id": "s1", "text": "a"}])
    p.models = {"slots": []}
    assert _project_models(p).get("model_family") is None


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

# ── MiniMax H3 reference media ────────────────────────────────────────────────

def test_h3_references_round_trip_on_the_project():
    """The ORDER is the contract — Studio numbers the prompt tags from this list and the
    Chain Sampler encodes the same list, so persistence must not reorder or coerce it."""
    from movie_editor.backend.timeline import Project
    refs = [
        {"kind": "image", "filename": "face.png"},
        {"kind": "video", "filename": "walk.mp4", "audio": "walk.wav"},
        {"kind": "audio", "filename": "voice.wav"},
    ]
    p = Project.from_dict({"h3_references": refs})
    assert p.h3_references == refs
    assert Project.from_dict(p.to_dict()).h3_references == refs
    # absent on every project that predates the feature, never None
    assert Project.from_dict({}).h3_references == []


def test_the_ports_endpoint_answers_for_the_project_not_the_global_default(monkeypatch):
    """/api/pipeline-ports resolves its family the same way, from ?pid=.

    Without that it described whatever project was saved last, so opening one on the other
    family read its saved wires back as "(not allowed)" and let the panel's reconcile drop
    them against overrides for a core that isn't there.
    """
    from movie_editor.backend import nodes, pipeline_wiring

    monkeypatch.setattr(nodes, "load_models", lambda: {"slots": [], "model_family": "ltxav"})
    p = _project(scenes=[{"id": "s1", "text": "a"}])
    p.models = {"slots": [{"id": "a", "role": "unet"}], "model_family": "minimax_h3"}

    assert pipeline_wiring.family_of(_project_models(p)) == "minimax_h3"
    assert pipeline_wiring.family_of(nodes.load_models()) == "ltxav"
    # No pid — an app with no project open — still answers with the global default.
    assert pipeline_wiring.family_of(_project_models(None)) == "ltxav"
