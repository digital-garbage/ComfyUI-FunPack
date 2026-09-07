"""Media: files the user brought in, rather than something a run produced.

A reference image, an imported clip -- anything meant to feed INTO a
generation instead of coming out of one. Files live under config.MEDIA_DIR as
<id><ext>; a sibling index.json records the display name, kind, size and
timestamp. Pure file/JSON I/O, same shape as projects.py, deliberately not
importing it: a media id and a project id look alike but mean different
things, and a store that borrowed the other's helpers would make that
harder to see, not easier.
"""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import time
import uuid
from pathlib import Path

from . import config

#: Guards the index's read-modify-write. Nothing in save_upload/delete awaits,
#: so on today's single request per event-loop-turn this is never contended --
#: but that safety is an accident of the call path, not something the index
#: itself enforces, and the first `asyncio.to_thread` wrapped around a big
#: upload (core/routes.py already does this for other slow I/O) turns two
#: concurrent writers into a lost update or a crash on the shared tmp file.
_LOCK = threading.Lock()

_ID = re.compile(r"\A[0-9a-f]{12}\Z")

#: What a name is allowed to become on disk. Not a sanitiser: a name that
#: merely survives cleaning can still be "..", so path safety comes from the
#: generated id, never from this.
MAX_NAME = 120

_IMAGE_EXT = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"})
_VIDEO_EXT = frozenset({".mp4", ".webm", ".mov", ".mkv", ".avi"})
_AUDIO_EXT = frozenset({".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus", ".weba"})

#: What core will accept onto disk at all. A media route is a place a browser
#: can send arbitrary bytes; this is the refusal, not a UI hint -- anything
#: not on this list is rejected before it reaches a filename.
ALLOWED_EXT = _IMAGE_EXT | _VIDEO_EXT | _AUDIO_EXT

#: Matches ComfyUI's own --max-upload-size default (100MB): FunPack's routes
#: ride on ComfyUI's Application in production, which already refuses a bigger
#: body before this ever runs. A larger number here would be a limit this
#: store claims to enforce but never gets the chance to.
MAX_BYTES = 100 * 1024 * 1024


def is_id(value) -> bool:
    return isinstance(value, str) and bool(_ID.match(value))


def _kind(ext: str) -> str:
    if ext in _IMAGE_EXT:
        return "image"
    if ext in _VIDEO_EXT:
        return "video"
    if ext in _AUDIO_EXT:
        return "audio"
    return "other"


def _dir():
    config.MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    return config.MEDIA_DIR


def _index_path() -> Path:
    return _dir() / "index.json"


def _load_index() -> list[dict]:
    p = _index_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return []  # an unreadable index is not a reason to have no list
    return data if isinstance(data, list) else []


def _save_index(items: list[dict]) -> None:
    path = _index_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, indent=2), encoding="utf-8")
    tmp.replace(path)  # a concurrent read never sees a half-written index


def listing() -> list[dict]:
    """Every upload, newest first. An entry whose file vanished is dropped
    rather than shown -- there is nothing behind it to open."""
    out = [it for it in _load_index() if (config.MEDIA_DIR / it.get("filename", "")).is_file()]
    out.sort(key=lambda i: i.get("added", 0), reverse=True)
    return out


def get(mid: str) -> dict | None:
    if not is_id(mid):
        return None
    return next((it for it in _load_index() if it.get("id") == mid), None)


def save_upload(orig_name: str, data: bytes) -> dict:
    """Write one uploaded file and record it. Raises ValueError for anything
    this store refuses -- an empty upload, an extension not on the allowlist,
    or a file over MAX_BYTES -- so the route can turn that into a 400 naming
    why, rather than a file silently written and then never explained."""
    if not data:
        raise ValueError("that file is empty")
    if len(data) > MAX_BYTES:
        raise ValueError(f"that file is over the {MAX_BYTES // (1024 * 1024)}MB limit")
    ext = Path(orig_name or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise ValueError(f"{ext or 'that file type'} is not something FunPack imports")

    mid = uuid.uuid4().hex[:12]
    filename = mid + ext
    _dir()
    (config.MEDIA_DIR / filename).write_bytes(data)
    entry = {
        "id": mid,
        "name": (Path(orig_name or "").name or filename)[:MAX_NAME],
        "filename": filename,
        "kind": _kind(ext),
        "size": len(data),
        "added": time.time(),
    }
    # The write above needs no lock -- filename is this call's own uuid, so two
    # uploads never touch the same file. The index below is shared, and a
    # read here that raced another upload's write would silently drop it.
    with _LOCK:
        items = _load_index()
        items.append(entry)
        _save_index(items)
    return entry


def path_for(mid: str) -> Path | None:
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
    with _LOCK:
        items = _load_index()
        keep, removed = [], None
        for it in items:
            if it.get("id") == mid:
                removed = it
            else:
                keep.append(it)
        if removed is None:
            return False
        _save_index(keep)
    try:
        (config.MEDIA_DIR / removed["filename"]).unlink(missing_ok=True)
    except OSError:
        pass
    return True
