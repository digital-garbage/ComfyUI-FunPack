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
    build_character_bible_anchor,
    build_combined_prompt,
    build_auto_continuity_guides,
    build_mixed_solo_guides_payload,
    build_scene_anchors_payload,
    build_scene_guides_payload,
    continuity_media_refs,
    continuity_settings_for_run,
    effective_anchor,
    effective_negative_prompt,
    generation_prompt_fingerprint,
    normalize_character_bible,
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


def test_anchor_plus_two_scenes_emits_separators():
    p = _project(
        anchor="red-haired heroine",
        intro_transition="cut to",
        scenes=[
            {"text": "flying over the city", "transition_to_next": "blur"},
            {"text": "landing on a rooftop"},
        ],
    )
    prompt = build_combined_prompt(p, for_generation=True)
    # anchor first, intro marker before scene 0, scene marker before scene 1
    assert prompt.startswith("red-haired heroine")
    assert "cut to" in prompt
    assert "blur" in prompt
    assert prompt.index("cut to") < prompt.index("flying")
    assert prompt.index("blur") < prompt.index("landing")


def test_generation_prompt_fingerprint_includes_split_markers():
    p = _project(
        anchor="red-haired heroine",
        intro_transition="cut to",
        scenes=[
            {"text": "flying over the city", "transition_to_next": "blur"},
            {"text": "landing on a rooftop"},
        ],
    )
    fp = generation_prompt_fingerprint(p, p)
    assert "cut to" in fp["generation_prompt"]
    assert "blur" in fp["generation_prompt"]
    assert fp["display_prompt"] != fp["generation_prompt"]


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


def test_missing_markers_fall_back_to_scene_labels():
    p = _project(
        anchor="a knight",
        scenes=[{"text": "in a forest"}, {"text": "at a castle"}],
    )
    prompt = build_combined_prompt(p, for_generation=True)
    # No explicit markers -> generic "scene N" boundaries so the split still separates.
    assert "scene 1" in prompt
    assert "scene 2" in prompt


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
    p = _project(
        scenes=[
            {"id": "u1", "text": "one long shot", "frames": 49, "gen_unit_id": "u1", "cut_offset_frames": 0},
            {"id": "u1b", "text": "", "frames": 49, "gen_unit_id": "u1", "cut_offset_frames": 49},
            {"id": "s2", "text": "second scene", "frames": 97},
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


def test_character_bible_merges_into_anchor():
    p = _project(
        anchor="dimly lit apartment",
        character_bible={
            "name": "Nicole",
            "appearance": "long red hair, green eyes",
            "always_include": "photorealistic",
        },
        scenes=[{"text": "walks in"}],
    )
    anchor = effective_anchor(p)
    assert anchor.startswith("Character: Nicole.")
    assert "Appearance: long red hair, green eyes." in anchor
    assert anchor.endswith("dimly lit apartment")
    prompt = build_combined_prompt(p)
    assert "Character: Nicole." in prompt


def test_character_bible_never_include_merges_negative():
    p = _project(
        negative_prompt="blurry, low quality",
        character_bible={"never_include": "extra limbs, deformed hands"},
        scenes=[{"text": "a"}],
    )
    neg = effective_negative_prompt(p)
    assert "blurry" in neg
    assert "extra limbs" in neg


def test_scene_characters_merge_into_prompt(monkeypatch):
    char_map = {
        "c1": {"id": "c1", "name": "Nicole", "appearance": "red hair"},
        "c2": {"id": "c2", "name": "Alex", "appearance": "tall"},
    }
    monkeypatch.setattr(
        "movie_editor.backend.timeline._load_character_map",
        lambda: char_map,
    )
    p = _project(scenes=[
        {"text": "walks in", "character_ids": ["c1"]},
        {"text": "sits down", "character_ids": ["c1", "c2"]},
    ])
    prompt = build_combined_prompt(p, for_generation=True)
    assert "Character: Nicole." in prompt
    assert "Character: Alex." in prompt
    assert prompt.index("Nicole") < prompt.index("walks")
    assert prompt.index("Alex") < prompt.index("sits")


def test_scene_character_face_ref_drives_identity_pin(monkeypatch):
    char_map = {"c1": {"id": "c1", "name": "Nicole", "face_ref": "face1"}}
    monkeypatch.setattr(
        "movie_editor.backend.timeline._load_character_map",
        lambda: char_map,
    )
    p = _project(
        continuity_settings={"identity_pin_ref": "other"},
        scenes=[
            {"id": "s1", "text": "a", "source": {"type": "carry"}},
            {"id": "s2", "text": "b", "source": {"type": "carry"}, "character_ids": ["c1"]},
        ],
    )
    cs = continuity_settings_for_run(p)
    sc2 = p.scenes[1]
    assert resolve_scene_identity_pin(sc2, char_map, cs) == "face1"
    guides = build_auto_continuity_guides(p, p)
    assert guides["scenes"][1][0]["media_ref"] == "face1"


def test_character_bible_refs_in_continuity_media():
    p = _project(
        character_bible={"face_ref": "f1", "body_ref": "b1", "detail_ref": "d1"},
        scenes=[{"text": "a"}],
    )
    refs = continuity_media_refs(p, p)
    assert refs == ["f1", "b1", "d1"]
    p.continuity_settings = {"auto_enabled": False}
    assert continuity_media_refs(p, p) == []


def test_stale_identity_pin_ignored_when_auto_off():
    p = _project(
        continuity_settings={"auto_enabled": False, "identity_pin_ref": "stale_pin"},
        scenes=[{"text": "a"}, {"text": "b", "source": {"type": "carry"}}],
    )
    assert continuity_media_refs(p, p) == []


def test_scene_character_media_refs(monkeypatch):
    char_map = {"c1": {"id": "c1", "face_ref": "f1", "body_ref": "b1"}}
    monkeypatch.setattr(
        "movie_editor.backend.timeline._load_character_map",
        lambda: char_map,
    )
    p = _project(scenes=[{"text": "a", "character_ids": ["c1"]}])
    from movie_editor.backend.timeline import character_media_refs_for_project
    assert character_media_refs_for_project(p, char_map) == ["f1", "b1"]


def test_character_bible_changes_generation_hash():
    p = _project(scenes=[{"text": "walks"}])
    h1 = generation_prompt_fingerprint(p, p)["prompt_hash"]
    p.character_bible = {"name": "Alex"}
    h2 = generation_prompt_fingerprint(p, p)["prompt_hash"]
    assert h1 != h2


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
