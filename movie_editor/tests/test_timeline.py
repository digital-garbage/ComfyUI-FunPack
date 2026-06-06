"""Unit tests for prompt assembly, project store, and workflow injection.

None of these require a running ComfyUI. Run from the repo root:
    pytest movie_editor/tests -q
"""
import json
import sys
from pathlib import Path

import pytest

# Make `movie_editor` importable when run from the repo root.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from movie_editor.backend.timeline import Project, Scene, build_combined_prompt  # noqa: E402
from movie_editor.backend import workflow  # noqa: E402


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


def test_workflow_injection_into_fixture_graph():
    graph = {
        "1": {"class_type": "FunPackPromptCombiner", "inputs": {"text": ""}},
        "2": {"class_type": "FunPackLTXAVSceneChainSampler",
              "inputs": {"seed": 0, "num_frames_per_scene": 97, "max_scenes": 8,
                         "model": ["9", 0]}},
        "3": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"frame_rate": 25}},
    }
    out, applied = workflow.inject(graph, {
        "prompt": "hello world",
        "seed": 42,
        "num_frames_per_scene": 121,
        "max_scenes": 5,
        "frame_rate": 30,
    })
    assert out["1"]["inputs"]["text"] == "hello world"
    assert out["2"]["inputs"]["seed"] == 42
    assert out["2"]["inputs"]["num_frames_per_scene"] == 121
    assert out["2"]["inputs"]["max_scenes"] == 5
    assert out["3"]["inputs"]["frame_rate"] == 30
    # wired link on model must be preserved
    assert out["2"]["inputs"]["model"] == ["9", 0]
    assert any("prompt ->" in a for a in applied)


def test_workflow_injection_requires_prompt_sink():
    graph = {"2": {"class_type": "FunPackLTXAVSceneChainSampler", "inputs": {"seed": 0}}}
    with pytest.raises(workflow.WorkflowError):
        workflow.inject(graph, {"prompt": "no text node here"})
