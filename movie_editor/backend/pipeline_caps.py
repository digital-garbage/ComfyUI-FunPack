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
# REFINEMENT, and only refinement. Simple mode has no rating UI, so anything driven by
# ratings or a trained key cannot do its job here — that is the one and only reason a
# setting belongs in these lists. Everything else runs exactly as it does in the Editor:
# cross-shot memory, guides, experimental sampling, the second pass, all of it.
#
# Cost is NOT a reason. Deciding a feature was too expensive for someone is how the second
# pass ended up stripped from a mode whose predecessor allowed it, while its switch stayed
# on screen doing nothing.
#
# These mirror the frontend's RATING_GATED_KNOBS / RATING_GATED_STUDIO (engine_settings.js)
# and the velocity gate in sampler_panel.js. Keep them in step: a control that is hidden
# here but still live at runtime is the same lie in the other direction.
#
# Applied per RUN, to a copy. The project keeps whatever the user set in the Editor, so
# switching back restores it; only the graph that gets built is stripped.
SIMPLE_MODE_SAMPLER_OFF: dict[str, Any] = {
    "embed_guidance": False,
    "score_slider": False,
    "taste_nearest_prompt": False,
    "output_guidance": False,
    "trajectory_guidance": False,
    "dynashift": False,
}

# Studio refiner keys, same rule.
SIMPLE_MODE_REFINER_OFF: dict[str, Any] = {
    "reference_injection": False,
    "value_guidance": False,
    "steer_mode": "relative",
}

# Per-sampler refinement, inside studio_settings["samplers"]: velocity bias replays a rated
# velocity bank and rescue reacts to one. Both are hidden in Simple mode already.
SIMPLE_MODE_SAMPLER_ENTRY_OFF: dict[str, Any] = {
    "velocity_bias_mode": "off",
    "rescue_mode": False,
}


def _strip_sampler_entries(samplers: Any) -> Any:
    """Force the per-sampler refinement keys off in a `samplers` config, whatever shape it
    has — the panel writes a dict keyed by pass, but a list is cheap to tolerate and a
    wrong guess here would silently leave the bank replaying."""
    if isinstance(samplers, dict):
        # A dict of per-pass entries, or one entry that IS the dict. Telling them apart by
        # the keys we would set is enough and needs no knowledge of the pass names.
        if any(k in samplers for k in SIMPLE_MODE_SAMPLER_ENTRY_OFF):
            return {**samplers, **SIMPLE_MODE_SAMPLER_ENTRY_OFF}
        return {k: _strip_sampler_entries(v) for k, v in samplers.items()}
    if isinstance(samplers, list):
        return [_strip_sampler_entries(v) for v in samplers]
    return samplers


def apply_simple_mode(sampler_inputs: Optional[dict], studio_settings: Optional[dict]) -> tuple[dict, dict]:
    """Return (sampler_inputs, studio_settings) copies with refinement forced off."""
    si = dict(sampler_inputs or {})
    si.update(SIMPLE_MODE_SAMPLER_OFF)
    ss = dict(studio_settings or {})
    refiner = dict(ss.get("refiner") or {})
    refiner.update(SIMPLE_MODE_REFINER_OFF)
    ss["refiner"] = refiner
    if ss.get("samplers") is not None:
        ss["samplers"] = _strip_sampler_entries(ss["samplers"])
    return si, ss
