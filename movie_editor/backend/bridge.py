"""Bridge to ComfyUI from inside its own process.

Parse/library calls go DIRECT to FunPack functions (no HTTP). Queue/history/view
go over loopback HTTP to ComfyUI's own server so we reuse its prompt validation and
output handling exactly.
"""
from __future__ import annotations

import importlib
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from . import config

_FUNPACK_ROOT = Path(__file__).resolve().parents[2]
_FUNPACK_PATH_ENSURED = False


def _ensure_funpack_path() -> None:
    """ComfyUI adds the custom-node folder to sys.path, but unit tests and some load
    orders do not. Never use relative `from ...conditioning` fallbacks - they mask the
    real ImportError with 'attempted relative import beyond top-level package'."""
    global _FUNPACK_PATH_ENSURED
    if _FUNPACK_PATH_ENSURED:
        return
    root = str(_FUNPACK_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    _FUNPACK_PATH_ENSURED = True


def format_funpack_error(exc: BaseException) -> str:
    msg = str(exc).strip()
    if msg:
        return f"{type(exc).__name__}: {msg}"
    return f"{type(exc).__name__} (no message)"


def _funpack_attr(module: str, name: str):
    _ensure_funpack_path()
    return getattr(importlib.import_module(module), name)


# ── In-process FunPack calls (no HTTP) ───────────────────────────────────────

def _funpack_imports():
    parse_timeline_segments = _funpack_attr("conditioning", "parse_timeline_segments")
    apply_prompt_shortcuts = _funpack_attr("templates", "apply_prompt_shortcuts")
    load_transition_db = _funpack_attr("templates", "load_transition_db")
    transition_items = _funpack_attr("templates", "transition_items")
    return parse_timeline_segments, apply_prompt_shortcuts, load_transition_db, transition_items


def parse_timeline(prompt: str, seed: int = 0) -> dict:
    """{anchor, scenes, transitions} — the canonical split Studio will see, via the ONE splitter
    (split_scenes) generation now uses, so the preview's scene count/boundaries match generation
    exactly. Scene text is the EXPANDED text (what Studio encodes); the anchor is prepended by
    Studio per scene. The lossless raw editing split lives in parse_timeline_verbatim."""
    split_fn = _funpack_attr("conditioning", "split_scenes")
    s = split_fn(str(prompt or ""))
    scenes = [{"index": i, "text": (sc.get("expanded") or "")}
              for i, sc in enumerate(s.get("scenes", []) or [])]
    transitions = [{"after_scene": i - 1, "visual_effect": sc.get("effect")}
                   for i, sc in enumerate(s.get("scenes", []) or [])
                   if i > 0 and sc.get("effect")]
    return {"anchor": s.get("anchor_expanded", ""), "scenes": scenes, "transitions": transitions}


def parse_timeline_raw(prompt: str) -> dict:
    """Split WITHOUT expanding shortcuts — scene texts stay verbatim (shortcuts and
    transition triggers preserved). Used by editing views so the user keeps operating
    on shortcuts unchanged; only the full preview / generation expand them."""
    parse_timeline_segments, *_ = _funpack_imports()
    return parse_timeline_segments(str(prompt or ""))


def parse_timeline_verbatim(prompt: str) -> dict:
    """LOSSLESS split into anchor + scenes: boundaries from shortcut-aware transitions,
    but no expansion and no dropped words (anchor + scenes reproduce the prompt). This is
    what the editor uses to map the global prompt onto the timeline."""
    try:
        split_fn = _funpack_attr("conditioning", "split_timeline_verbatim")
        return split_fn(str(prompt or ""))
    except Exception:
        # Fall back to the Studio-style raw split so Apply still works when verbatim
        # mapping fails (bad shortcut regex, import edge case, etc.).
        return parse_timeline_raw(prompt)


def scene_refinement_keys(prompt: str, default_key: str = "default") -> list:
    """Per-scene refinement keys that generation will actually use, for the editor preview.

    Returns one entry per scene aligned to the canonical split_scenes():
      {"keys": [<non-default keys, sorted>], "uses_default": bool, "default_key": str}
    A scene with no fired non-default keys steers with the project default key
    (uses_default=True). Mirrors generation exactly by reusing the same resolver."""
    resolve = _funpack_attr("conditioning", "resolve_scene_refinement_keys")
    default_key = str(default_key or "default").strip() or "default"
    sets = resolve(str(prompt or ""))
    out = []
    for s in sets:
        keys = sorted(s)
        if keys:
            out.append({"keys": keys, "uses_default": False, "default_key": default_key})
        else:
            out.append({"keys": [default_key], "uses_default": True, "default_key": default_key})
    return out


def refinement_key_pool(prompt: str) -> list:
    """Every non-default refinement key whose shortcut fires anywhere in the prompt — the exact
    set a Studio session reset will wipe. Scene-count independent, so it mirrors the backend
    `_v2_reset_prompt_keys`. Used so the reset confirmation lists exactly what will be cleared.
    Derived from the one canonical split, like everything else."""
    pool_for = _funpack_attr("conditioning", "refinement_key_pool_for")
    try:
        return sorted(k for k in pool_for(str(prompt or "")) if k)
    except Exception:
        return []


def validate_generation_prompt(full, target) -> dict:
    """Build the generation prompt and run fingerprint; track changes since last queue."""
    from .timeline import (
        gen_unit_root,
        generation_prompt_fingerprint,
        group_generative_units,
    )

    bundle = generation_prompt_fingerprint(full, target)
    active = [s for s in target.scenes if not s.excluded]
    expected_roots = [gen_unit_root(group) for _, group in group_generative_units(active)]

    meta = full.generation_meta or {}
    prev_run = meta.get("run_hash") or meta.get("prompt_hash")
    run_hash = bundle["run_hash"]
    text_hash = bundle["prompt_hash"]
    run_changed = bool(prev_run and prev_run != run_hash)
    prev_text = meta.get("prompt_hash")
    text_changed = bool(prev_text and prev_text != text_hash)

    return {
        **bundle,
        "expected_scenes": len(expected_roots),
        "prompt_changed_since_last_queue": run_changed,
        "text_changed_since_last_queue": text_changed,
        "anchors_changed_since_last_queue": run_changed and not text_changed,
    }


# ── ComfyUI log capture ───────────────────────────────────────────────────────
# Tee stdout/stderr into a ring buffer so the editor can show ComfyUI's real backend
# log (writes still pass through to the terminal). Captures print() and logging.
import collections as _collections
import threading as _threading

_LOG = _collections.deque(maxlen=5000)
_LOG_LOCK = _threading.Lock()
_log_installed = False


class _Tee:
    def __init__(self, orig):
        self._orig = orig
        self._buf = ""

    def write(self, s):
        try:
            self._orig.write(s)
        except Exception:
            pass
        try:
            with _LOG_LOCK:
                self._buf += s
                while "\n" in self._buf:
                    line, self._buf = self._buf.split("\n", 1)
                    _LOG.append(line)
        except Exception:
            pass
        return len(s) if isinstance(s, str) else 0

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._orig, name)


def install_log_capture():
    global _log_installed
    if _log_installed:
        return
    import sys
    try:
        sys.stdout = _Tee(sys.stdout)
        sys.stderr = _Tee(sys.stderr)
        _log_installed = True
    except Exception:
        pass


def recent_log(limit: int = 500) -> list:
    with _LOG_LOCK:
        return list(_LOG)[-int(limit):]


def rating_labels() -> dict:
    """FunPack Studio V2 rating labels (for the Scene rating dropdown)."""
    labels = _funpack_attr("conditioning", "V2_RATING_LABELS")
    try:
        internal = {
            _funpack_attr("conditioning", "MOVIE_EDITOR_CONTINUE_RATING"),
            _funpack_attr("conditioning", "MOVIE_EDITOR_FRESH_PROMPT_RATING"),
        }
    except Exception:
        internal = {"__funpack_continue__", "__funpack_fresh_prompt__"}
    out = [
        str(l) for l in (labels or [])
        if l and str(l) not in internal and not str(l).startswith("__funpack_")
    ]
    return {"labels": out}


def transitions() -> dict:
    _, _, load_transition_db, transition_items = _funpack_imports()
    data = load_transition_db()
    return {"data": data, "transitions": transition_items(data)}


def _library_fns():
    """FunPack shortcut/transition CRUD, in-process (no HTTP)."""
    return (
        _funpack_attr("templates", "shortcut_items"),
        _funpack_attr("templates", "save_shortcut_item"),
        _funpack_attr("templates", "delete_shortcut_item"),
        _funpack_attr("templates", "transition_items"),
        _funpack_attr("templates", "save_transition_item"),
        _funpack_attr("templates", "delete_transition_item"),
    )


def _shortcut_categories() -> list:
    fn = _funpack_attr("templates", "shortcut_categories")
    return fn()


def shortcuts() -> dict:
    si, *_ = _library_fns()
    return {"shortcuts": si(), "categories": _shortcut_categories()}


def save_shortcut(payload: dict) -> dict:
    si, save, *_ = _library_fns()
    save(payload)
    return {"shortcuts": si(), "categories": _shortcut_categories()}


def delete_shortcut(name: str) -> dict:
    si, _save, delete, *_ = _library_fns()
    delete(name)
    return {"shortcuts": si(), "categories": _shortcut_categories()}


def save_category(payload: dict) -> dict:
    """Add a category (and optionally a sub-category under it) to the managed list."""
    add = _funpack_attr("templates", "add_shortcut_category")
    si, *_ = _library_fns()
    add(payload.get("category", ""), payload.get("sub_category", ""))
    return {"shortcuts": si(), "categories": _shortcut_categories()}


def save_transition(payload: dict) -> dict:
    _si, _ss, _ds, ti, save, *_ = _library_fns()
    save(payload)
    return {"transitions": ti()}


def delete_transition(name: str) -> dict:
    _si, _ss, _ds, ti, _sv, delete, *_ = _library_fns()
    delete(name)
    return {"transitions": ti()}


def export_shortcuts() -> dict:
    return _funpack_attr("templates", "load_shortcut_db")()


def import_shortcuts(incoming: dict) -> dict:
    shortcut_items = _funpack_attr("templates", "shortcut_items")
    load_shortcut_db = _funpack_attr("templates", "load_shortcut_db")
    save_shortcut_db = _funpack_attr("templates", "save_shortcut_db")
    normalize_shortcut_db = _funpack_attr("templates", "normalize_shortcut_db")
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
    return {"imported": count, "shortcuts": shortcut_items(data), "categories": _shortcut_categories()}


def export_transitions() -> dict:
    return _funpack_attr("templates", "load_transition_db")()


def import_transitions(incoming: dict) -> dict:
    transition_items = _funpack_attr("templates", "transition_items")
    load_transition_db = _funpack_attr("templates", "load_transition_db")
    save_transition_db = _funpack_attr("templates", "save_transition_db")
    normalize_transition_db = _funpack_attr("templates", "normalize_transition_db")
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


async def interrupt() -> dict:
    """Ask ComfyUI to interrupt the running prompt."""
    async with await _session() as s:
        async with s.post(_url("/interrupt")) as r:
            r.raise_for_status()
            return {"interrupted": True}


# ── sampler step progress ─────────────────────────────────────────────────────
# ComfyUI's ProgressBar calls a single global hook with (value, total). We chain it so
# we record the latest step without clobbering ComfyUI's own (websocket) progress.
_progress = {"value": 0, "max": 0, "ts": 0.0}
_progress_installed = False


def _install_progress_hook():
    global _progress_installed
    if _progress_installed:
        return
    try:
        import time as _t
        import comfy.utils as _cu
    except Exception:
        return
    prev = getattr(_cu, "PROGRESS_BAR_HOOK", None)

    def _hook(value, total, preview=None, *args, **kwargs):
        try:
            _progress["value"] = int(value)
            _progress["max"] = int(total)
            _progress["ts"] = _t.time()
        except Exception:
            pass
        if callable(prev):
            try:
                prev(value, total, preview, *args, **kwargs)
            except TypeError:
                try:
                    prev(value, total, preview)
                except Exception:
                    pass
            except Exception:
                pass

    try:
        _cu.set_progress_bar_global_hook(_hook)
        _progress_installed = True
    except Exception:
        pass


def reset_progress() -> None:
    _progress["value"] = 0
    _progress["max"] = 0
    _progress["ts"] = 0.0


def current_progress() -> dict:
    _install_progress_hook()
    return {"value": _progress["value"], "max": _progress["max"]}


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
