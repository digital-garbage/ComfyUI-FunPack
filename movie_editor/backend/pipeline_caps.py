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


# Source modes that only make sense with FunPack Chain Sampler in the graph.


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




