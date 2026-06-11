"""In-flight chain sampler progress for the Movie Editor.

While a multi-scene FunPackLTXAVSceneChainSampler run is executing, partial decodes
are published here so the editor can show finished scenes before the full run ends.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

_lock = threading.Lock()
_by_key: dict[str, dict[str, Any]] = {}
_prompt_to_key: dict[str, str] = {}


def begin(prompt_id: str, progress_key: str, scene_count: int) -> None:
    if not prompt_id or not progress_key or scene_count <= 1:
        return
    with _lock:
        _prompt_to_key[prompt_id] = progress_key
        _by_key[progress_key] = {
            "scene_count": int(scene_count),
            "completed_scenes": 0,
            "media": None,
            "ts": time.time(),
        }


def update(progress_key: str, completed_scenes: int, media: dict) -> None:
    if not progress_key or not media:
        return
    with _lock:
        slot = _by_key.get(progress_key)
        if not slot:
            return
        slot["completed_scenes"] = max(int(completed_scenes), int(slot.get("completed_scenes") or 0))
        slot["media"] = dict(media)
        slot["ts"] = time.time()


def read_for_prompt(prompt_id: str) -> Optional[dict]:
    with _lock:
        key = _prompt_to_key.get(prompt_id)
        if not key:
            return None
        slot = _by_key.get(key)
        if not slot or not slot.get("media"):
            return None
        completed = int(slot.get("completed_scenes") or 0)
        total = int(slot.get("scene_count") or 0)
        if completed <= 0 or total <= 1 or completed >= total:
            return None
        return {
            "partial": True,
            "completed_scenes": completed,
            "scene_count": total,
            "media": [dict(slot["media"])],
        }


def finish(prompt_id: str) -> None:
    with _lock:
        key = _prompt_to_key.pop(prompt_id, None)
        if key:
            _by_key.pop(key, None)
