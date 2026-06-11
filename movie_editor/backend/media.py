"""Media bin: uploaded assets (images / clips) the editor can assign to scenes.

Files live under config.MEDIA_DIR as <id><ext>; a sibling index.json records the
display name, kind, size and timestamp. Pure file/JSON I/O — no ComfyUI imports.
"""
from __future__ import annotations

import json
import mimetypes
import time
import uuid
from pathlib import Path
from typing import Optional

from . import config

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
_VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif"}
_AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus", ".weba"}


def _index_path() -> Path:
    return config.MEDIA_DIR / "index.json"


def _load_index() -> list[dict]:
    p = _index_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_index(items: list[dict]) -> None:
    config.ensure_dirs()
    _index_path().write_text(json.dumps(items, indent=2))


def _kind(ext: str) -> str:
    ext = ext.lower()
    if ext in _IMAGE_EXT:
        return "image"
    if ext in _VIDEO_EXT:
        return "video"
    if ext in _AUDIO_EXT:
        return "audio"
    return "other"


def list_media() -> list[dict]:
    # newest first; drop entries whose file vanished
    out = []
    for it in _load_index():
        if (config.MEDIA_DIR / it.get("filename", "")).is_file():
            out.append(it)
    out.sort(key=lambda i: i.get("added", 0), reverse=True)
    return out


def save_upload(orig_name: str, data: bytes) -> dict:
    config.ensure_dirs()
    ext = Path(orig_name or "").suffix or ".bin"
    mid = uuid.uuid4().hex[:12]
    filename = mid + ext
    (config.MEDIA_DIR / filename).write_bytes(data)
    entry = {
        "id": mid,
        "name": Path(orig_name).name or filename,
        "filename": filename,
        "kind": _kind(ext),
        "size": len(data),
        "added": time.time(),
    }
    items = _load_index()
    items.append(entry)
    _save_index(items)
    return entry


def get(mid: str) -> Optional[dict]:
    return next((i for i in _load_index() if i.get("id") == mid), None)


def rename(mid: str, name: str) -> Optional[dict]:
    """Update display name only — file on disk and id stay the same."""
    label = str(name or "").strip()
    if not label:
        return None
    items = _load_index()
    for it in items:
        if it.get("id") == mid:
            it["name"] = label
            _save_index(items)
            return dict(it)
    return None


def path_for(mid: str) -> Optional[Path]:
    it = get(mid)
    if not it:
        return None
    p = config.MEDIA_DIR / it["filename"]
    return p if p.is_file() else None


def content_type(mid: str) -> str:
    it = get(mid)
    if not it:
        return "application/octet-stream"
    return mimetypes.guess_type(it["filename"])[0] or "application/octet-stream"


def delete(mid: str) -> bool:
    items = _load_index()
    keep, removed = [], None
    for it in items:
        if it.get("id") == mid:
            removed = it
        else:
            keep.append(it)
    if removed is None:
        return False
    try:
        (config.MEDIA_DIR / removed["filename"]).unlink(missing_ok=True)
    except OSError:
        pass
    _save_index(keep)
    return True
