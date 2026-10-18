"""Training sets: a named list of pictures and clips with a caption each, kept as one JSON file.

An item points at a file, it never copies it: `{"kind": "bin", "id"}` is a file in the media bin,
`{"kind": "output", "type", "subfolder", "filename"}` is a result ComfyUI wrote. A set is data the
person edits on screen; the screen is the truth, so `save` takes the whole document.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

from ..._core import config
from .captions import MAX_CAPTION

IMAGE_EXT = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"})
VIDEO_EXT = frozenset({".mp4", ".webm", ".mov", ".mkv", ".avi"})
OUTPUT_TYPES = ("output", "temp", "input")
#: Where an item came from. "own" is a file the person brought; the rest are rated results.
LABELS = ("own", "liked", "disliked", "bad_composition", "bad_image")
MAX_ITEMS = 5000
MAX_SETS = 200

_LOCK = threading.RLock()
_NAME = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9 _.\-]{0,59}\Z")
_ID = re.compile(r"\A[0-9a-f]{12}\Z")


def _dir() -> Path:
    path = config.TRAINING_DIR / "sets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def valid_name(name) -> bool:
    return isinstance(name, str) and bool(_NAME.match(name)) and ".." not in name


def _path(name: str) -> Path:
    if not valid_name(name):
        raise ValueError("A set name is letters, digits, spaces, dots, dashes and underscores (up to 60), starting with a letter or digit.")
    return _dir() / f"{name}.json"


def kind_of(filename: str) -> str | None:
    ext = Path(str(filename)).suffix.lower()
    return "image" if ext in IMAGE_EXT else "video" if ext in VIDEO_EXT else None


def clean_src(raw) -> dict | None:
    """A source the server can resolve later, or None. Nothing about it is trusted to be a path."""
    if not isinstance(raw, dict):
        return None
    if raw.get("kind") == "bin":
        mid = raw.get("id")
        return {"kind": "bin", "id": mid} if isinstance(mid, str) and _ID.match(mid) else None
    if raw.get("kind") == "output":
        name, sub, typ = raw.get("filename"), raw.get("subfolder") or "", raw.get("type") or "output"
        if not (isinstance(name, str) and isinstance(sub, str) and typ in OUTPUT_TYPES and name and kind_of(name)):
            return None
        if "\\" in name or "/" in name or ".." in Path(sub).parts or Path(sub).is_absolute() or "\\" in sub or len(name) > 255 or len(sub) > 255:
            return None
        return {"kind": "output", "type": typ, "subfolder": sub, "filename": name}
    return None


def src_key(src: dict) -> str:
    return f"bin:{src['id']}" if src["kind"] == "bin" else f"{src['type']}:{src['subfolder']}/{src['filename']}"


def clean_item(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    src = clean_src(raw.get("src"))
    if src is None:
        return None
    iid = raw.get("id")
    label = raw.get("label")
    kind = raw.get("media_kind")
    return {
        "id": iid if isinstance(iid, str) and _ID.match(iid) else uuid.uuid4().hex[:12],
        "src": src,
        "media_kind": kind if kind in ("image", "video") else (kind_of(src.get("filename", "")) or "image"),
        "label": label if label in LABELS else "own",
        "caption": str(raw.get("caption") or "")[:MAX_CAPTION],
        "name": str(raw.get("name") or src.get("filename") or src.get("id") or "")[:120],
    }


def clean(doc, name: str) -> dict:
    """A whole document, repaired: unknown things dropped, duplicates (same file) merged."""
    doc = doc if isinstance(doc, dict) else {}
    seen, items = set(), []
    for raw in (doc.get("items") or [])[:MAX_ITEMS]:
        item = clean_item(raw)
        if item is None or src_key(item["src"]) in seen:
            continue
        seen.add(src_key(item["src"]))
        items.append(item)
    settings = doc.get("settings")
    return {"name": name, "trigger": " ".join(str(doc.get("trigger") or "").split())[:MAX_CAPTION],
            "postfix": " ".join(str(doc.get("postfix") or "").split())[:MAX_CAPTION], "items": items,
            "settings": settings if isinstance(settings, dict) else {}, "updated": time.time()}


def names() -> list[str]:
    return sorted(p.stem for p in _dir().glob("*.json") if valid_name(p.stem))


def summaries() -> list[dict]:
    out = []
    for name in names():
        doc = get(name)
        if doc is None:
            continue
        counts = {label: sum(1 for i in doc["items"] if i["label"] == label) for label in LABELS}
        out.append({"name": name, "count": len(doc["items"]), "labels": counts,
                    "captioned": sum(1 for i in doc["items"] if i["caption"]), "updated": doc.get("updated", 0)})
    return out


def get(name: str) -> dict | None:
    try:
        path = _path(name)
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(doc, dict):
        return None
    return {**clean(doc, name), "updated": doc.get("updated", 0)}


def save(name: str, doc) -> dict:
    out = clean(doc, name)
    with _LOCK:
        path = _path(name)
        if not path.exists() and len(names()) >= MAX_SETS:
            raise ValueError(f"At most {MAX_SETS} sets: delete one first.")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
        tmp.replace(path)
    return out


def create(name: str) -> dict:
    if get(name) is not None:
        raise ValueError(f"A set called {name!r} already exists.")
    return save(name, {})


def delete(name: str) -> bool:
    with _LOCK:
        path = _path(name)
        if not path.exists():
            return False
        path.unlink()
        return True


def add(name: str, entries) -> tuple[dict, int]:
    """Append entries (items without ids are fine); returns (document, how many were new)."""
    with _LOCK:
        doc = get(name) or clean({}, name)
        before = len(doc["items"])
        doc["items"] = list(doc["items"]) + [e for e in (entries or []) if isinstance(e, dict)]
        merged = save(name, doc)
    return merged, len(merged["items"]) - before


def resolve(src: dict, roots: dict, bin_path) -> Path | None:
    """The file an item points at, or None when it is gone or outside its root.

    `roots` maps an output type to its directory; `bin_path(id)` is the media bin's lookup.
    """
    if src["kind"] == "bin":
        path = bin_path(src["id"])
        return Path(path) if path else None
    root = roots.get(src["type"])
    if not root:
        return None
    root = Path(root).resolve()
    path = (root / src["subfolder"] / src["filename"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return None
    return path
