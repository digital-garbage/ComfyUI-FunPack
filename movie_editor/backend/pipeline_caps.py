"""Which FunPack pipeline nodes are active for a project (Studio / Chain Sampler).

UI and generation use these flags so Studio-only and Chain-only features stay off when
the built-in pipeline is disabled or custom conditioning/sampler slots are selected.
"""
from __future__ import annotations

from typing import Any, Optional

from .timeline import Scene, source_type


def _models_dict(models: Optional[dict]) -> dict:
    return models if isinstance(models, dict) else {}


def uses_funpack_studio(project: Any, models: Optional[dict] = None) -> bool:
    if _models_dict(models).get("disable_core"):
        return False
    slot = getattr(project, "conditioning_slot", None) or "funpack"
    return slot == "funpack"


def uses_chain_sampler(project: Any, models: Optional[dict] = None) -> bool:
    if _models_dict(models).get("disable_core"):
        return False
    slot = getattr(project, "sampler_slot", None) or "funpack"
    return slot == "funpack"


def capabilities(project: Any, models: Optional[dict] = None) -> dict[str, bool]:
    m = _models_dict(models)
    return {
        "studio": uses_funpack_studio(project, m),
        "chain_sampler": uses_chain_sampler(project, m),
        "disable_core": bool(m.get("disable_core")),
        "imported_workflow": bool(m.get("workflow_import")),
    }


def effective_source_type(scene: Scene, chain_available: bool) -> str:
    """Runtime source mode for generation/media prep.

    Without Chain Sampler: fall back to t2v (empty) unless the scene has an explicit
    image anchor. Stored carry/mixed/etc. values are preserved in the project file.
    """
    st = source_type(scene)
    if chain_available:
        return st
    if st == "image" and scene.source and scene.source.media_ref:
        return "image"
    if st == "v2v" and scene.source and scene.source.media_ref:
        return "v2v"
    return "empty"


# Source modes whose whole point is a media-bin asset. A scene set to one of these
# with nothing picked still generates — the anchor is simply absent — so this is a
# warning, not a block.
_ANCHOR_MEDIA_SOURCES = ("image", "mixed", "generated_frame", "v2v", "anchor_guide")


def source_needs_anchor_media(scene: Scene, chain_available: bool) -> bool:
    """True when generation expects a media-bin asset for this scene's source."""
    return effective_source_type(scene, chain_available) in _ANCHOR_MEDIA_SOURCES


def is_t2v(project) -> bool:
    """True when the project starts shots from the prompt rather than an image."""
    return str(getattr(project, "generation_mode", "i2v") or "i2v").lower() == "t2v"


def scenes_missing_anchor_media(project, chain_available: bool) -> list[str]:
    """Scene ids whose source mode wants an anchor but has none selected.

    Distinct from server-side _missing_scene_anchor_media, which catches a ref that
    IS set but has fallen out of the media bin. This catches the ref never being set
    at all — that scene is skipped silently when anchors are assembled, so without
    this the user just gets a shot that quietly ignored its own source setting.

    A t2v project expects no anchors, so nothing here is missing.
    """
    if is_t2v(project):
        return []
    missing: list[str] = []
    for sc in getattr(project, "scenes", None) or []:
        if getattr(sc, "excluded", False):
            continue
        src = getattr(sc, "source", None)
        if not src or getattr(src, "media_ref", None):
            continue
        if source_needs_anchor_media(sc, chain_available):
            missing.append(sc.id)
    return missing






# ── Simple mode ───────────────────────────────────────────────────────────────
# Simple mode is not a skin. It generates what you asked for and nothing else.
#
# A setting belongs here only if it CANNOT WORK in this mode — it is a no-op without a
# trained key (there is no rating UI to feed one), or it describes a relationship between
# shots (there is one shot). "Costs real time for a quality bet" is NOT a reason: that is
# the user's call to make, and making it for them is how the second pass ended up stripped
# from a mode whose predecessor allowed it, while its switch stayed on screen.
#
# Applied per RUN, to a copy. The project keeps whatever the user set in the Editor, so
# switching back restores it; only the graph that gets built is stripped.
SIMPLE_MODE_SAMPLER_OFF: dict[str, Any] = {
    # rating-driven — no-ops without a trained key, and this mode has no rating UI
    "embed_guidance": False,
    "score_slider": False,
    "taste_nearest_prompt": False,
    "output_guidance": False,
    "dynashift": False,
    # cross-shot memory and guides: real per-scene cost
    "mid_scene_guide": False,
    "joyai_memory": False,
    "joyai_audio_memory": False,
    "carry_i2v_guides": False,
    # experimental / expensive sampling
    "alg_anchor": False,
    "alg_blur_guides": False,
    "bounded_attention_enabled": False,
    "identity_transfer_enabled": False,
    "segmented_detailing": False,
    "plateau_cache": False,
    "context_windows": False,
    # second_pass / second_pass_op are deliberately ABSENT. They need no rated history and
    # no second shot, so they work here exactly as they do in the Editor — and upscaling a
    # single shot is one of the main things this mode is for.
}

# Studio refiner keys, same reasoning.
SIMPLE_MODE_REFINER_OFF: dict[str, Any] = {
    "reference_injection": False,
    "value_guidance": False,
    "steer_mode": "relative",
}


def apply_simple_mode(sampler_inputs: Optional[dict], studio_settings: Optional[dict]) -> tuple[dict, dict]:
    """Return (sampler_inputs, studio_settings) copies with the enhancements forced off."""
    si = dict(sampler_inputs or {})
    si.update(SIMPLE_MODE_SAMPLER_OFF)
    ss = dict(studio_settings or {})
    refiner = dict(ss.get("refiner") or {})
    refiner.update(SIMPLE_MODE_REFINER_OFF)
    ss["refiner"] = refiner
    return si, ss
