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
    build_scene_guides_payload,
    collapse_generative_units,
    group_generative_units,
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
    prompt = build_combined_prompt(p)
    # anchor first, intro marker before scene 0, scene marker before scene 1
    assert prompt.splitlines()[0] == "red-haired heroine"
    assert "cut to" in prompt
    assert "blur" in prompt
    assert prompt.index("cut to") < prompt.index("flying")
    assert prompt.index("blur") < prompt.index("landing")


def test_missing_markers_fall_back_to_scene_labels():
    p = _project(
        anchor="a knight",
        scenes=[{"text": "in a forest"}, {"text": "at a castle"}],
    )
    prompt = build_combined_prompt(p)
    # No explicit markers -> generic "scene N" boundaries so the split still separates.
    assert "scene 2" in prompt
    assert "scene 3" in prompt


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
    assert prompt.splitlines()[0] == "opening shot"


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
