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


def test_source_needs_anchor_media_by_mode():
    assert pipeline_caps.source_needs_anchor_media(_scene("image"), True) is True
    assert pipeline_caps.source_needs_anchor_media(_scene("mixed"), True) is True
    assert pipeline_caps.source_needs_anchor_media(_scene("anchor_guide"), True) is True
    assert pipeline_caps.source_needs_anchor_media(_scene("carry"), True) is False
    assert pipeline_caps.source_needs_anchor_media(_scene("empty"), True) is False


def test_scenes_missing_anchor_media_flags_only_the_unset_ones():
    """A ref that IS set but has left the media bin is server-side
    _missing_scene_anchor_media's job; this catches the ref never being set."""
    p = Project(name="t")
    ok = _scene("image", "img1"); ok.id = "s_ok"
    bad = _scene("image"); bad.id = "s_bad"
    carry = _scene("carry"); carry.id = "s_carry"
    skipped = _scene("mixed"); skipped.id = "s_excluded"; skipped.excluded = True
    p.scenes = [ok, bad, carry, skipped]
    assert pipeline_caps.scenes_missing_anchor_media(p, True) == ["s_bad"]


def test_scenes_missing_anchor_media_silent_without_chain_sampler():
    """Without Chain Sampler an anchorless image scene degrades to t2v by design,
    so warning about a missing anchor there would be noise."""
    p = Project(name="t")
    bad = _scene("image"); bad.id = "s_bad"
    p.scenes = [bad]
    assert pipeline_caps.scenes_missing_anchor_media(p, False) == []
