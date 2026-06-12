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


def models_enabled(models: Any) -> bool:
    if not isinstance(models, dict):
        return False
    lp = models.get("live_preview") or {}
    return bool(lp.get("enabled") and lp.get("slot_id"))
