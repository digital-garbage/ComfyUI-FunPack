"""Global character library for the Movie Editor (stored under FUNPACK_MOVIE_DATA)."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from . import config

CHARACTERS_PATH = config.DATA_DIR / "characters.json"


def normalize_character_record(raw: Optional[dict]) -> dict[str, Any]:
    raw = raw or {}
    return {
        "name": str(raw.get("name", "")).strip(),
        "appearance": str(raw.get("appearance", "")).strip(),
        "body": str(raw.get("body", "")).strip(),
        "wardrobe": str(raw.get("wardrobe", "")).strip(),
        "always_include": str(raw.get("always_include", "")).strip(),
        "never_include": str(raw.get("never_include", "")).strip(),
        "face_ref": raw.get("face_ref") or None,
        "body_ref": raw.get("body_ref") or None,
        "detail_ref": raw.get("detail_ref") or None,
    }


def _ensure_db() -> dict:
    config.ensure_dirs()
    if not CHARACTERS_PATH.exists():
        CHARACTERS_PATH.write_text(json.dumps({"characters": {}}, indent=2), encoding="utf-8")
    data = json.loads(CHARACTERS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        data = {"characters": {}}
    data.setdefault("characters", {})
    return data


def _write_db(data: dict) -> None:
    config.ensure_dirs()
    CHARACTERS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def character_items(data: Optional[dict] = None) -> list[dict]:
    db = data or _ensure_db()
    items = []
    for cid, raw in (db.get("characters") or {}).items():
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item["id"] = str(item.get("id") or cid)
        items.append(item)
    items.sort(key=lambda c: (c.get("name") or "").lower())
    return items


def load_character_map() -> dict[str, dict]:
    return {c["id"]: c for c in character_items()}


def list_characters() -> dict:
    return {"characters": character_items()}


def save_character(payload: dict) -> dict:
    db = _ensure_db()
    chars = db.setdefault("characters", {})
    original_id = str(payload.get("original_id") or "").strip()
    cid = str(payload.get("id") or original_id or _new_id())
    if original_id and original_id != cid and original_id in chars:
        del chars[original_id]
    now = time.time()
    prev = chars.get(cid) or {}
    item = normalize_character_record(payload)
    item["id"] = cid
    item["created_at"] = float(prev.get("created_at", now))
    item["updated_at"] = now
    chars[cid] = item
    _write_db(db)
    return {"characters": character_items(db)}


def delete_character(char_id: str) -> dict:
    db = _ensure_db()
    chars = db.setdefault("characters", {})
    chars.pop(str(char_id), None)
    _write_db(db)
    return {"characters": character_items(db)}