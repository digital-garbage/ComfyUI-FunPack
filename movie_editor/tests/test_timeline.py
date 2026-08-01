"""Unit tests for prompt assembly and the project store.

None of these require a running ComfyUI. Run from the repo root:
    pytest movie_editor/tests -q
"""
import json
import sys
from pathlib import Path

# Make `movie_editor` importable when run from the repo root.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from movie_editor.backend.timeline import (  # noqa: E402
    Project,
    Scene,
    STUDIO_DEFAULT_GUIDE,
    build_combined_prompt,
    build_generation_scene_segments,
    build_auto_continuity_guides,
    build_mixed_solo_guides_payload,
    build_scene_anchors_payload,
    build_scene_guides_payload,
    continuity_media_refs,
    continuity_settings_for_run,
    effective_anchor,
    effective_negative_prompt,
    effective_postfix,
    generation_prompt_fingerprint,
    normalize_continuity_settings,
    resolve_scene_identity_pin,
    collapse_generative_units,
    group_generative_units,
    is_mixed_source,
    normalize_guide_settings,
)


def _project(**kw):
    p = Project(**{k: v for k, v in kw.items() if k != "scenes"})
    p.scenes = [Scene.from_dict(s) for s in kw.get("scenes", [])]
    return p


def test_generation_prompt_is_verbatim_no_injected_separators():
    p = _project(
        anchor="red-haired heroine",
        intro_transition="cut to",
        scenes=[
            {"text": "flying over the city", "transition_to_next": "blur"},
            {"text": "landing on a rooftop"},
        ],
    )
    prompt = build_combined_prompt(p, for_generation=True)
    # anchor + scene texts, VERBATIM — nothing injected.
    assert prompt == "red-haired heroine flying over the city landing on a rooftop"
    assert "scene 1" not in prompt and "scene 2" not in prompt
    # boundaries + the real transitions travel STRUCTURALLY instead of in the prompt
    seg = build_generation_scene_segments(p)
    assert seg["anchor"] == "red-haired heroine"
    assert seg["scenes"] == ["cut to flying over the city", "blur landing on a rooftop"]


def test_generation_prompt_fingerprint_has_no_split_markers():
    p = _project(
        anchor="red-haired heroine",
        intro_transition="cut to",
        scenes=[
            {"text": "flying over the city", "transition_to_next": "blur"},
            {"text": "landing on a rooftop"},
        ],
    )
    fp = generation_prompt_fingerprint(p, p)
    # generation prompt is now verbatim (boundaries are structural) — no injected separators
    assert "scene 1" not in fp["generation_prompt"]
    assert fp["generation_prompt"] == "red-haired heroine flying over the city landing on a rooftop"


def test_generation_prompt_hash_changes_when_text_changes():
    p = _project(scenes=[{"text": "walks forward"}])
    h1 = generation_prompt_fingerprint(p, p)["prompt_hash"]
    p.scenes[0].text = "runs backward"
    h2 = generation_prompt_fingerprint(p, p)["prompt_hash"]
    assert h1 != h2


def test_generation_run_hash_changes_when_anchor_changes():
    p = _project(scenes=[{"id": "s1", "text": "a", "source": {"type": "image", "media_ref": "img1"}}])
    r1 = generation_prompt_fingerprint(p, p)
    p.scenes[0].source.media_ref = "img2"
    r2 = generation_prompt_fingerprint(p, p)
    assert r1["prompt_hash"] == r2["prompt_hash"]
    assert r1["run_hash"] != r2["run_hash"]


def test_project_mode_scene_inherits_length_ignoring_stale_frames():
    p = _project(num_frames_per_scene=97, frame_rate=25,
                 scenes=[{"text": "a", "frames": 49, "frames_mode": "project",
                          "fps": 12, "fps_mode": "project"}])
    sc = p.scenes[0]
    # project mode → the leftover per-scene frames/fps are ignored; the scene tracks the project.
    assert sc.eff_frames(p) == 97
    assert sc.eff_fps(p) == 25
    # changing the project length is now followed (no stale 49)
    p.num_frames_per_scene = 121
    assert sc.eff_frames(p) == 121


def test_timeline_and_custom_modes_keep_per_scene_length():
    p = _project(num_frames_per_scene=97,
                 scenes=[{"text": "a", "frames": 49, "frames_mode": "timeline"},
                         {"text": "b", "frames": 60, "frames_mode": "custom"}])
    assert p.scenes[0].eff_frames(p) == 49     # trimmed on the timeline
    assert p.scenes[1].eff_frames(p) == 60     # locked in the inspector


def test_postfix_travels_in_segments_but_not_global_prompt():
    p = _project(
        anchor="a knight",
        postfix="cinematic, 4k",
        scenes=[{"text": "in a forest"}, {"text": "at a castle"}],
    )
    # Postfix is a separate setting — never part of the verbatim global/combined prompt.
    prompt = build_combined_prompt(p, for_generation=True)
    assert "cinematic" not in prompt
    assert prompt == "a knight in a forest at a castle"
    # …but it rides along structurally so Studio can append it to every scene.
    seg = build_generation_scene_segments(p)
    assert seg["postfix"] == "cinematic, 4k"


def test_postfix_toggle_disables_it_without_losing_text():
    p = _project(postfix="cinematic, 4k", postfix_enabled=False,
                 scenes=[{"text": "in a forest"}])
    assert effective_postfix(p) == ""                       # toggled off → inert
    assert build_generation_scene_segments(p)["postfix"] == ""
    assert p.postfix == "cinematic, 4k"                     # text is preserved
    p.postfix_enabled = True
    assert effective_postfix(p) == "cinematic, 4k"


def test_run_hash_changes_when_postfix_changes():
    p = _project(scenes=[{"id": "s1", "text": "a", "source": {"type": "image", "media_ref": "img1"}}])
    r1 = generation_prompt_fingerprint(p, p)
    p.postfix = "moody lighting"
    r2 = generation_prompt_fingerprint(p, p)
    assert r1["run_hash"] != r2["run_hash"]


def test_carry_scenes_have_no_injected_labels():
    p = _project(
        anchor="a knight",
        scenes=[{"text": "in a forest"}, {"text": "at a castle"}],
    )
    prompt = build_combined_prompt(p, for_generation=True)
    # carry scenes (no transition) used to get an injected 'scene N' label — no longer.
    assert "scene 1" not in prompt and "scene 2" not in prompt
    assert prompt == "a knight in a forest at a castle"
    # the two boundaries are still there, structurally
    seg = build_generation_scene_segments(p)
    assert seg["scenes"] == ["in a forest", "at a castle"]


def test_video_clips_excluded_from_combined_prompt():
    p = _project(scenes=[
        {"text": "gen scene", "source": {"type": "carry"}},
        {"text": "", "source": {"type": "video", "media_ref": "vid1"}},
    ])
    prompt = build_combined_prompt(p)
    assert "gen scene" in prompt
    assert build_combined_prompt(p, include_excluded=True).count("gen scene") >= 1


def test_is_video_clip():
    from movie_editor.backend.timeline import Scene, SceneSource, is_generative_scene, is_video_clip

    sc = Scene(source=SceneSource(type="video", media_ref="v1"))
    assert is_video_clip(sc)
    assert not is_generative_scene(sc)


def test_excluded_scenes_dropped_unless_requested():
    p = _project(
        anchor="a cat",
        scenes=[
            {"text": "on a sofa", "transition_to_next": "cut"},
            {"text": "SKIP ME", "excluded": True, "transition_to_next": "cut"},
            {"text": "in the garden"},
        ],
    )
    assert "SKIP ME" not in build_combined_prompt(p)
    assert "SKIP ME" in build_combined_prompt(p, include_excluded=True)


def test_no_anchor_first_scene_has_no_leading_separator():
    p = _project(scenes=[{"text": "opening shot", "transition_to_next": "cut"}, {"text": "second"}])
    prompt = build_combined_prompt(p)
    assert prompt == "opening shot second"


def test_project_roundtrip_preserves_forward_compat_fields():
    p = _project(
        anchor="x",
        scenes=[{"text": "a", "source": {"type": "image", "media_ref": "asset1"}, "excluded": True}],
    )
    d = p.to_dict()
    p2 = Project.from_dict(json.loads(json.dumps(d)))
    assert p2.scenes[0].source.type == "image"
    assert p2.scenes[0].source.media_ref == "asset1"
    assert p2.scenes[0].excluded is True


def test_split_subclips_collapse_to_one_generative_unit():
    # Split subclips carry concrete per-scene lengths in timeline mode (how the split UI makes them).
    p = _project(
        scenes=[
            {"id": "u1", "text": "one long shot", "frames": 49, "frames_mode": "timeline",
             "gen_unit_id": "u1", "cut_offset_frames": 0},
            {"id": "u1b", "text": "", "frames": 49, "frames_mode": "timeline",
             "gen_unit_id": "u1", "cut_offset_frames": 49},
            {"id": "s2", "text": "second scene", "frames": 97, "frames_mode": "timeline"},
        ],
    )
    units = group_generative_units(p.scenes)
    assert len(units) == 2
    assert len(units[0][1]) == 2
    prompt = build_combined_prompt(p, for_generation=True)
    assert prompt.count("one long shot") == 1
    assert "second scene" in prompt
    collapsed = collapse_generative_units(p)
    assert len(collapsed.scenes) == 2
    assert collapsed.scenes[0].frames == 98
    assert collapsed.scenes[0].text == "one long shot"


def test_guide_settings_default_off():
    gs = normalize_guide_settings({})
    assert gs["stack_enabled"] is False
    assert gs["accumulate_prior"] is False


def test_guide_entry_is_importable_and_round_trips():
    """GuideEntry's class header was once overwritten by a neighbouring function,
    leaving its body stranded as dead code after a `return`. Nothing caught it because
    the guide stack defaults off, so no other test reaches the code that uses it."""
    from movie_editor.backend.timeline import GuideEntry

    g = GuideEntry.from_dict({"enabled": True, "source": "image", "media_ref": "img1"})
    assert g.enabled is True and g.source == "image" and g.media_ref == "img1"
    assert g.to_dict()["strength"] == 0.35


def test_guide_stack_enabled_paths_do_not_raise():
    """Both consumers of GuideEntry only run when the stack is switched on — the exact
    configuration that used to raise NameError."""
    p = _project(
        guide_settings={"stack_enabled": True},
        scenes=[
            {"id": "s1", "text": "a", "source": {"type": "image", "media_ref": "img1"}},
            {"id": "s2", "text": "b", "source": {"type": "mixed", "media_ref": "img2"},
             "guides": [{"enabled": True, "source": "image", "media_ref": "img3"}]},
        ],
    )
    payload = build_scene_guides_payload(p)
    assert payload["stack_enabled"] is True
    assert payload["scenes"][1][0]["media_ref"] == "img3"
    assert "img3" in continuity_media_refs(p, p)


def test_continuity_defaults_auto_on():
    cs = normalize_continuity_settings({})
    assert cs["auto_enabled"] is True
    assert cs["prior_scene_guides"] is True
    assert cs["mid_scene_guide"] is True


def test_auto_continuity_solo_image_no_prior_guides():
    p = _project(
        scenes=[
            {"id": "s1", "text": "a", "source": {"type": "carry"}},
            {"id": "s2", "text": "b", "source": {"type": "image", "media_ref": "img2"}},
        ],
    )
    segment = Project.from_dict(p.to_dict())
    segment.scenes = [p.scenes[1]]
    assert build_auto_continuity_guides(p, segment) is None


def test_auto_continuity_solo_mixed_prior_anchor():
    p = _project(
        scenes=[
            {"id": "s1", "text": "a", "source": {"type": "image", "media_ref": "img1"}},
            {"id": "s2", "text": "b", "source": {"type": "mixed", "media_ref": "img2"}},
        ],
    )
    segment = Project.from_dict(p.to_dict())
    segment.scenes = [p.scenes[1]]
    payload = build_auto_continuity_guides(p, segment)
    assert payload["scenes"][0][0]["media_ref"] == "img1"


def test_auto_continuity_identity_pin_and_carry_chain():
    p = _project(
        continuity_settings={"identity_pin_ref": "pin1"},
        scenes=[
            {"id": "s1", "text": "a"},
            {"id": "s2", "text": "b", "source": {"type": "carry"}},
        ],
    )
    payload = build_auto_continuity_guides(p, p)
    assert payload["scenes"][0][0]["media_ref"] == "pin1"
    assert payload["scenes"][1][0]["media_ref"] == "pin1"
    assert any(g["source"] == "template" for g in payload["scenes"][1])
    refs = continuity_media_refs(p, p)
    assert "pin1" in refs
    p = _project(scenes=[{"text": "a"}, {"text": "b"}])
    assert build_scene_guides_payload(p) is None


def test_guide_stack_studio_default_per_continuation_scene():
    p = _project(
        guide_settings={"stack_enabled": True},
        scenes=[{"id": "s1", "text": "a"}, {"id": "s2", "text": "b"}, {"id": "s3", "text": "c"}],
    )
    payload = build_scene_guides_payload(p)
    assert payload is not None
    assert payload["scenes"][0] is None
    assert payload["scenes"][1] == [STUDIO_DEFAULT_GUIDE]
    assert payload["scenes"][2] == [STUDIO_DEFAULT_GUIDE]


def test_mixed_solo_guides_borrow_prior_anchor():
    p = _project(
        scenes=[
            {"id": "s1", "text": "a", "source": {"type": "image", "media_ref": "img1"}},
            {"id": "s2", "text": "b", "source": {"type": "mixed", "media_ref": "img2"}},
        ],
    )
    payload = build_mixed_solo_guides_payload(p, p.scenes[1])
    assert payload is not None
    guide = payload["scenes"][0][0]
    assert guide["source"] == "image"
    assert guide["media_ref"] == "img1"


def test_mixed_solo_guides_template_when_prior_is_carry():
    p = _project(
        scenes=[
            {"id": "s1", "text": "a", "source": {"type": "carry"}},
            {"id": "s2", "text": "b", "source": {"type": "mixed", "media_ref": "img2"}},
        ],
    )
    payload = build_mixed_solo_guides_payload(p, p.scenes[1])
    assert payload["scenes"][0][0]["source"] == "template"


def test_mixed_source_anchors_separate_from_guides():
    p = _project(
        scenes=[
            {"id": "s1", "text": "a", "source": {"type": "image", "media_ref": "img1"}},
            {"id": "s2", "text": "b", "source": {"type": "mixed", "media_ref": "img2"}},
        ],
    )
    assert is_mixed_source(p.scenes[1])
    assert build_scene_guides_payload(p) is None
    anchors = build_scene_anchors_payload(p)
    assert anchors["1"]["scene_id"] == "s2"
    assert anchors["1"]["media_ref"] == "img2"
    assert anchors["1"]["strength"] == 1.0


def test_guide_stack_accumulate_prior():
    p = _project(
        guide_settings={"stack_enabled": True, "accumulate_prior": True},
        scenes=[{"id": "s1", "text": "a"}, {"id": "s2", "text": "b"}, {"id": "s3", "text": "c"}],
    )
    payload = build_scene_guides_payload(p)
    scene3 = payload["scenes"][2]
    assert len(scene3) == 2
    assert scene3[0]["source"] == "scene" and scene3[0]["scene_index"] == 0
    assert scene3[1]["source"] == "scene" and scene3[1]["scene_index"] == 1


def test_effective_anchor_is_manual_text():
    p = _project(anchor="dimly lit apartment", scenes=[{"text": "walks in"}])
    assert effective_anchor(p) == "dimly lit apartment"
    prompt = build_combined_prompt(p)
    assert prompt.startswith("dimly lit apartment")


def test_effective_negative_is_manual_text():
    p = _project(negative_prompt="blurry, low quality", scenes=[{"text": "a"}])
    assert effective_negative_prompt(p) == "blurry, low quality"


def test_project_identity_pin_from_continuity_settings():
    p = _project(
        continuity_settings={"identity_pin_ref": "pin1"},
        scenes=[
            {"id": "s1", "text": "a", "source": {"type": "carry"}},
            {"id": "s2", "text": "b", "source": {"type": "carry"}},
        ],
    )
    cs = continuity_settings_for_run(p)
    assert resolve_scene_identity_pin(cs) == "pin1"
    guides = build_auto_continuity_guides(p, p)
    assert guides["scenes"][0][0]["media_ref"] == "pin1"


def test_stale_identity_pin_ignored_when_auto_off():
    p = _project(
        continuity_settings={"auto_enabled": False, "identity_pin_ref": "stale_pin"},
        scenes=[{"text": "a"}, {"text": "b", "source": {"type": "carry"}}],
    )
    assert continuity_media_refs(p, p) == []


def test_scene_anchor_media_refs():
    p = _project(
        scenes=[
            {"text": "a", "source": {"type": "generated_frame", "media_ref": "frame1"}},
            {"text": "b", "source": {"type": "carry"}},
        ],
    )
    from movie_editor.backend.timeline import scene_anchor_media_refs

    assert scene_anchor_media_refs(p) == ["frame1"]


def test_projects_store_crud(tmp_path, monkeypatch):
    from movie_editor.backend import config, projects
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    created = projects.create("demo")
    assert projects.get(created.id).name == "demo"
    created.anchor = "hero"
    projects.save(created)
    assert projects.get(created.id).anchor == "hero"
    assert any(x["id"] == created.id for x in projects.list_projects())
    assert projects.delete(created.id) is True
    assert projects.get(created.id) is None
