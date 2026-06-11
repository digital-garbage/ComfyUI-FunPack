"""Post-render NLE presets: clip effects and video transitions for the timeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import config

NLE_PATH = config.DATA_DIR / "nle_library.json"

DEFAULT_EFFECTS: list[dict[str, Any]] = [
    {"id": "zoom_in", "name": "Zoom in (push)", "description": "Timed push-in (set ratio, length, start in inspector)"},
    {"id": "zoom_out", "name": "Zoom out (pull back)", "description": "Timed pull-back (set ratio, length, start in inspector)"},
    {
        "id": "blur",
        "name": "Gaussian blur",
        "param": {"label": "Strength (0–1)", "default": 0.3, "min": 0, "max": 1, "step": 0.05},
    },
    {
        "id": "fade_in",
        "name": "Fade in",
        "param": {"label": "Seconds", "default": 0.5, "min": 0, "max": 10, "step": 0.1},
    },
    {
        "id": "fade_out",
        "name": "Fade out",
        "param": {"label": "Seconds", "default": 0.5, "min": 0, "max": 10, "step": 0.1},
    },
]

DEFAULT_VIDEO_TRANSITIONS: list[dict[str, Any]] = [
    {
        "id": "crossfade",
        "name": "Crossfade",
        "type": "crossfade",
        "param": {"label": "Frames", "default": 16, "min": 1, "max": 120, "step": 1},
    },
    {
        "id": "fadeblack",
        "name": "Fade to black",
        "type": "fadeblack",
        "param": {"label": "Frames", "default": 16, "min": 1, "max": 120, "step": 1},
    },
    {
        "id": "wipeleft",
        "name": "Wipe left",
        "type": "wipeleft",
        "param": {"label": "Frames", "default": 16, "min": 1, "max": 120, "step": 1},
    },
    {
        "id": "wiperight",
        "name": "Wipe right",
        "type": "wiperight",
        "param": {"label": "Frames", "default": 16, "min": 1, "max": 120, "step": 1},
    },
]


def _default_db() -> dict[str, Any]:
    return {
        "version": 1,
        "effects": DEFAULT_EFFECTS,
        "video_transitions": DEFAULT_VIDEO_TRANSITIONS,
    }


def _normalize_param(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    return {
        "label": str(raw.get("label") or "Value"),
        "default": float(raw.get("default", 0)),
        "min": float(raw.get("min", 0)),
        "max": float(raw.get("max", 1)),
        "step": float(raw.get("step", 0.1)),
    }


def normalize(data: dict[str, Any] | None) -> dict[str, Any]:
    base = _default_db()
    if not isinstance(data, dict):
        return base
    effects = data.get("effects")
    if not isinstance(effects, list) or not effects:
        effects = base["effects"]
    vts = data.get("video_transitions")
    if not isinstance(vts, list) or not vts:
        vts = base["video_transitions"]
    out_effects: list[dict[str, Any]] = []
    for item in effects:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        entry: dict[str, Any] = {
            "id": str(item["id"]),
            "name": str(item.get("name") or item["id"]),
        }
        if item.get("description"):
            entry["description"] = str(item["description"])
        param = _normalize_param(item.get("param"))
        if param:
            entry["param"] = param
        out_effects.append(entry)
    out_vts: list[dict[str, Any]] = []
    for item in vts:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        entry = {
            "id": str(item["id"]),
            "name": str(item.get("name") or item["id"]),
            "type": str(item.get("type") or item["id"]),
        }
        param = _normalize_param(item.get("param"))
        if param:
            entry["param"] = param
        out_vts.append(entry)
    return {
        "version": 1,
        "effects": out_effects or base["effects"],
        "video_transitions": out_vts or base["video_transitions"],
    }


def load() -> dict[str, Any]:
    config.ensure_dirs()
    if not NLE_PATH.exists():
        data = _default_db()
        NLE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data
    try:
        raw = json.loads(NLE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        raw = None
    data = normalize(raw)
    if raw != data:
        NLE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
