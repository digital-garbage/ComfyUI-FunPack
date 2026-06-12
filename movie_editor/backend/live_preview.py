"""In-process live preview state for Cutting Room generation runs.

The Chain Sampler writes per-scene preview frames here; the Movie Editor server
serves them while a job is running.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

_lock = threading.Lock()
_state: dict[str, Any] = {
    "active": False,
    "dir": "",
    "scene_index": -1,
    "scene_count": 0,
    "updated_at": 0.0,
    "frame_path": "",
}


def reset(preview_dir: str, scene_count: int = 0) -> None:
    with _lock:
        _state["active"] = True
        _state["dir"] = str(preview_dir or "")
        _state["scene_index"] = -1
        _state["scene_count"] = int(scene_count or 0)
        _state["updated_at"] = time.time()
        _state["frame_path"] = ""
    if preview_dir:
        Path(preview_dir).mkdir(parents=True, exist_ok=True)


def clear() -> None:
    with _lock:
        _state["active"] = False
        _state["dir"] = ""
        _state["scene_index"] = -1
        _state["scene_count"] = 0
        _state["updated_at"] = time.time()
        _state["frame_path"] = ""


def publish(scene_index: int, frame_path: str, scene_count: Optional[int] = None) -> None:
    with _lock:
        if not _state.get("active"):
            return
        _state["scene_index"] = int(scene_index)
        if scene_count is not None:
            _state["scene_count"] = int(scene_count)
        _state["frame_path"] = str(frame_path or "")
        _state["updated_at"] = time.time()


def snapshot() -> dict[str, Any]:
    with _lock:
        return dict(_state)


def parse_config(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        out = json.loads(str(raw))
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


PREVIEW_VAE_PORT = "FunPackLTXAVSceneChainSampler.preview_vae"
PREVIEW_VAE_WIRE = f"port:{PREVIEW_VAE_PORT}"


def _wire_targets(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t) for t in raw if t]
    return [str(raw)]


def preview_vae_wire(models: Any) -> tuple[Optional[str], Optional[str]]:
    """Return (slot_id, vae_output) wired to Chain Sampler live preview, if any."""
    if not isinstance(models, dict):
        return None, None
    for slot in models.get("slots") or []:
        sid = slot.get("id")
        if not sid:
            continue
        for out_name, raw in (slot.get("wires") or {}).items():
            if PREVIEW_VAE_WIRE in _wire_targets(raw):
                return str(sid), str(out_name or "VAE")
    co = (models.get("core_overrides") or {}).get("sampler") or {}
    src = str(co.get("preview_vae") or "")
    if src.startswith("out:"):
        parts = src.split(":", 2)
        if len(parts) >= 3 and parts[1]:
            return str(parts[1]), str(parts[2] or "VAE")
    lp = models.get("live_preview") or {}
    sid = lp.get("slot_id")
    if sid:
        return str(sid), str(lp.get("vae_output") or "VAE")
    return None, None


def models_enabled(models: Any) -> bool:
    sid, _ = preview_vae_wire(models)
    return bool(sid)


def api_payload(ui_prefix: str = "/funpack/movie") -> dict[str, Any]:
    """Status dict for /api/live-preview and /api/progress."""
    import os

    snap = snapshot()
    if not snap.get("active"):
        return {"active": False}
    frame_path = snap.get("frame_path") or ""
    ready = bool(frame_path and os.path.isfile(frame_path))
    payload: dict[str, Any] = {
        "active": True,
        "ready": ready,
        "scene_index": snap.get("scene_index", -1),
        "scene_count": snap.get("scene_count", 0),
        "updated_at": snap.get("updated_at", 0),
    }
    if ready:
        ts = int(float(snap.get("updated_at") or 0) * 1000)
        payload["url"] = f"{ui_prefix}/api/live-preview/frame?t={ts}"
    return payload
