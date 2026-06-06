"""Bridge to ComfyUI from inside its own process.

Parse/library calls go DIRECT to FunPack functions (no HTTP). Queue/history/view
go over loopback HTTP to ComfyUI's own server so we reuse its prompt validation and
output handling exactly.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from . import config


# ── In-process FunPack calls (no HTTP) ───────────────────────────────────────

def _funpack_imports():
    try:
        from conditioning import parse_timeline_segments
        from templates import apply_prompt_shortcuts, load_transition_db, transition_items
    except ImportError:  # imported as a package submodule
        from ...conditioning import parse_timeline_segments  # type: ignore
        from ...templates import (  # type: ignore
            apply_prompt_shortcuts, load_transition_db, transition_items,
        )
    return parse_timeline_segments, apply_prompt_shortcuts, load_transition_db, transition_items


def parse_timeline(prompt: str, seed: int = 0) -> dict:
    """{anchor, scenes, transitions} — the canonical split Studio will see."""
    parse_timeline_segments, apply_prompt_shortcuts, *_ = _funpack_imports()
    expanded, _applied = apply_prompt_shortcuts(str(prompt or ""), seed=int(seed or 0))
    return parse_timeline_segments(expanded)


def transitions() -> dict:
    _, _, load_transition_db, transition_items = _funpack_imports()
    data = load_transition_db()
    return {"data": data, "transitions": transition_items(data)}


def _library_fns():
    """FunPack shortcut/transition CRUD, in-process (no HTTP)."""
    try:
        from templates import (  # type: ignore
            shortcut_items, save_shortcut_item, delete_shortcut_item,
            transition_items, save_transition_item, delete_transition_item,
        )
    except ImportError:
        from ...templates import (  # type: ignore
            shortcut_items, save_shortcut_item, delete_shortcut_item,
            transition_items, save_transition_item, delete_transition_item,
        )
    return (shortcut_items, save_shortcut_item, delete_shortcut_item,
            transition_items, save_transition_item, delete_transition_item)


def shortcuts() -> dict:
    si, *_ = _library_fns()
    return {"shortcuts": si()}


def save_shortcut(payload: dict) -> dict:
    si, save, *_ = _library_fns()
    save(payload)
    return {"shortcuts": si()}


def delete_shortcut(name: str) -> dict:
    si, _save, delete, *_ = _library_fns()
    delete(name)
    return {"shortcuts": si()}


def save_transition(payload: dict) -> dict:
    _si, _ss, _ds, ti, save, *_ = _library_fns()
    save(payload)
    return {"transitions": ti()}


def delete_transition(name: str) -> dict:
    _si, _ss, _ds, ti, _sv, delete, *_ = _library_fns()
    delete(name)
    return {"transitions": ti()}


def export_shortcuts() -> dict:
    try:
        from templates import load_shortcut_db  # type: ignore
    except ImportError:
        from ...templates import load_shortcut_db  # type: ignore
    return load_shortcut_db()


def import_shortcuts(incoming: dict) -> dict:
    try:
        from templates import (  # type: ignore
            shortcut_items, load_shortcut_db, save_shortcut_db, normalize_shortcut_db,
        )
    except ImportError:
        from ...templates import (  # type: ignore
            shortcut_items, load_shortcut_db, save_shortcut_db, normalize_shortcut_db,
        )
    imported = normalize_shortcut_db(incoming)
    data = load_shortcut_db()
    shortcuts = data.setdefault("shortcuts", {})
    count = 0
    for key, item in imported.get("shortcuts", {}).items():
        if isinstance(item, dict):
            shortcuts[str(key)] = item
            count += 1
    save_shortcut_db(data)
    data = load_shortcut_db()
    return {"imported": count, "shortcuts": shortcut_items(data)}


def export_transitions() -> dict:
    try:
        from templates import load_transition_db  # type: ignore
    except ImportError:
        from ...templates import load_transition_db  # type: ignore
    return load_transition_db()


def import_transitions(incoming: dict) -> dict:
    try:
        from templates import (  # type: ignore
            transition_items, load_transition_db, save_transition_db, normalize_transition_db,
        )
    except ImportError:
        from ...templates import (  # type: ignore
            transition_items, load_transition_db, save_transition_db, normalize_transition_db,
        )
    imported = normalize_transition_db(incoming)
    data = load_transition_db()
    transitions = data.setdefault("transitions", {})
    count = 0
    for key, item in imported.get("transitions", {}).items():
        if isinstance(item, dict):
            transitions[str(key)] = item
            count += 1
    save_transition_db(data)
    data = load_transition_db()
    return {"imported": count, "transitions": transition_items(data)}


# ── Loopback HTTP to ComfyUI (queue + results) ───────────────────────────────

def _url(path: str) -> str:
    return f"{config.comfy_base_url()}{path}"


async def _session():
    import aiohttp
    return aiohttp.ClientSession()


class ComfyError(RuntimeError):
    pass


async def queue_prompt(graph: dict, client_id: Optional[str] = None) -> dict:
    payload = {"prompt": graph, "client_id": client_id or uuid.uuid4().hex}
    async with await _session() as s:
        async with s.post(_url("/prompt"), json=payload) as r:
            data = await r.json()
            if r.status >= 400 or data.get("node_errors"):
                raise ComfyError(data.get("error") or data.get("node_errors") or f"HTTP {r.status}")
            return data


async def history(prompt_id: str) -> dict:
    async with await _session() as s:
        async with s.get(_url(f"/history/{prompt_id}")) as r:
            r.raise_for_status()
            return await r.json()


async def is_running(prompt_id: str) -> bool:
    async with await _session() as s:
        async with s.get(_url("/queue")) as r:
            r.raise_for_status()
            state = await r.json()
    for bucket in ("queue_running", "queue_pending"):
        for item in state.get(bucket, []):
            if len(item) > 1 and item[1] == prompt_id:
                return True
    return False


_object_info_cache = {"data": None}


def _build_object_info_inprocess() -> Optional[dict]:
    """Build the node registry directly from ComfyUI's NODE_CLASS_MAPPINGS — no HTTP,
    no port guessing. Mirrors server.py's /object_info (the subset the editor reads).
    Returns None if we're not running inside ComfyUI."""
    try:
        import nodes as comfy_nodes  # ComfyUI's global node registry
    except Exception:
        return None
    import json
    mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", None)
    if not mappings:
        return None
    disp = getattr(comfy_nodes, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}
    out: dict[str, Any] = {}
    for name, cls in mappings.items():
        try:
            if hasattr(cls, "GET_NODE_INFO_V1"):       # V3 internal nodes
                info = cls.GET_NODE_INFO_V1()
            else:
                rt = list(getattr(cls, "RETURN_TYPES", ()) or ())
                info = {
                    "input": cls.INPUT_TYPES(),
                    "output": rt,
                    "output_name": list(getattr(cls, "RETURN_NAMES", rt) or rt),
                    "display_name": disp.get(name, name),
                    "category": getattr(cls, "CATEGORY", ""),
                }
            # normalize tuples -> lists (the HTTP path gets this for free via JSON);
            # default=str drops any exotic, non-serializable widget defaults.
            out[name] = json.loads(json.dumps(info, default=str))
        except Exception:
            continue
    return out or None


async def object_info(refresh: bool = False) -> dict:
    """Full ComfyUI node registry (class -> input/output spec). Cached; combo lists
    (installed files) refresh when `refresh=True` — same effect as pressing R in ComfyUI.

    Built in-process from NODE_CLASS_MAPPINGS (reliable, no port resolution). Falls
    back to loopback HTTP only if the in-process registry isn't importable."""
    if _object_info_cache["data"] is not None and not refresh:
        return _object_info_cache["data"]
    if refresh:
        try:  # force a filesystem rescan of model combos, like ComfyUI's R key
            import folder_paths
            folder_paths.filename_list_cache.clear()
        except Exception:
            pass
    data = _build_object_info_inprocess()
    if data is None:
        async with await _session() as s:
            async with s.get(_url("/object_info")) as r:
                r.raise_for_status()
                data = await r.json()
    _object_info_cache["data"] = data
    return data


async def fetch_view(filename: str, subfolder: str = "", type_: str = "output") -> tuple[bytes, str]:
    from urllib.parse import urlencode
    q = urlencode({"filename": filename, "subfolder": subfolder, "type": type_})
    async with await _session() as s:
        async with s.get(_url(f"/view?{q}")) as r:
            r.raise_for_status()
            return await r.read(), r.headers.get("content-type", "application/octet-stream")
