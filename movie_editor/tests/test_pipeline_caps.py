"""Tests for pipeline capability flags (Studio / Chain Sampler availability)."""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from movie_editor.backend import pipeline_caps  # noqa: E402
from movie_editor.backend.timeline import Project, Scene, SceneSource  # noqa: E402


def _scene(stype: str, ref: str | None = None) -> Scene:
    return Scene(
        id="s1",
        text="x",
        source=SceneSource(type=stype, media_ref=ref),
    )


def test_disable_core_turns_off_studio_and_chain():
    p = Project(name="t")
    m = {"disable_core": True, "workflow_import": {"name": "W"}}
    c = pipeline_caps.capabilities(p, m)
    assert c["studio"] is False
    assert c["chain_sampler"] is False


def test_custom_slots_without_disable_core():
    p = Project(name="t", conditioning_slot="custom1", sampler_slot="custom2")
    c = pipeline_caps.capabilities(p, {"slots": []})
    assert c["studio"] is False
    assert c["chain_sampler"] is False


def test_effective_source_fallback_without_chain():
    sc = _scene("carry")
    assert pipeline_caps.effective_source_type(sc, False) == "empty"
    sc2 = _scene("mixed", "img1")
    assert pipeline_caps.effective_source_type(sc2, False) == "empty"
    sc3 = _scene("image", "img1")
    assert pipeline_caps.effective_source_type(sc3, False) == "image"


def test_effective_source_unchanged_with_chain():
    sc = _scene("carry")
    assert pipeline_caps.effective_source_type(sc, True) == "carry"
