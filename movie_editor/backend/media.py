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


THUMB_MAX_DIM = 320  # long edge, px -- matches the frontend's own thumbnail cap (mediabrowser.js)


def _thumbs_dir() -> Path:
    d = config.MEDIA_DIR / "thumbnails"
    d.mkdir(parents=True, exist_ok=True)
    return d


def thumb_path_for(mid: str) -> Optional[Path]:
    """A small cached JPEG for this media item's grid thumbnail, generating it on first
    request. None for audio/other (nothing to draw) or a missing/unreadable source.

    The full media file was always what the bin's thumbnail grid requested — the frontend
    downscaled it AFTER downloading it in full, so a bin of real photos/video re-downloaded
    every original file just to show a 96px square. This is the actual size reduction: a
    resized derivative cached once on disk, so later requests (a page reload, scrolling a
    thumbnail back into view) transfer kilobytes instead of megabytes.
    """
    it = get(mid)
    if not it or it.get("kind") not in ("image", "video"):
        return None
    src = path_for(mid)
    if src is None:
        return None
    out = _thumbs_dir() / f"{mid}.jpg"
    try:
        if out.is_file() and out.stat().st_mtime >= src.stat().st_mtime:
            return out
    except OSError:
        pass
    try:
        if it["kind"] == "image":
            _make_image_thumb(src, out)
        else:
            _make_video_thumb(src, out)
        return out if out.is_file() else None
    except Exception:
        return None


def _tmp_thumb_path(out: Path) -> Path:
    # Unique per attempt, not a shared "<id>.tmp" -- two requests for the same not-yet-cached
    # id (a quick double-render before the first request's own cache write lands) would
    # otherwise both write the same temp file and could interleave into a corrupt image that
    # then gets renamed into place and served as if valid.
    return out.with_name(f"{out.stem}.{uuid.uuid4().hex[:8]}.tmp")


def _make_image_thumb(src: Path, out: Path) -> None:
    from PIL import Image
    img = Image.open(src)
    img.thumbnail((THUMB_MAX_DIM, THUMB_MAX_DIM))  # in place, preserves aspect, never upscales
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    tmp = _tmp_thumb_path(out)
    try:
        img.save(tmp, "JPEG", quality=82)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)


def _make_video_thumb(src: Path, out: Path) -> None:
    import shutil
    import subprocess
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH")
    tmp = _tmp_thumb_path(out)
    # force_original_aspect_ratio=decrease: shrink to fit inside THUMB_MAX_DIM square,
    # preserving aspect, never upscaling a source already smaller than the cap.
    scale = f"scale='min({THUMB_MAX_DIM},iw)':'min({THUMB_MAX_DIM},ih)':force_original_aspect_ratio=decrease"
    try:
        # -f mjpeg forces the output format explicitly -- the tmp filename's own extension
        # (.tmp, not .jpg) gives ffmpeg nothing to infer a container from otherwise, and it
        # refuses to guess.
        proc = subprocess.run(
            [ff, "-y", "-i", str(src), "-vframes", "1", "-vf", scale, "-f", "mjpeg", str(tmp)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not tmp.is_file():
            raise RuntimeError((proc.stderr or "ffmpeg failed")[-500:])
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)


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
    try:
        (_thumbs_dir() / f"{mid}.jpg").unlink(missing_ok=True)
    except OSError:
        pass
    _save_index(keep)
    return True
