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


def expand_prompt_fragment(text: str) -> str:
    """Expand shortcuts in a standalone fragment (anchor / postfix) the way generation does —
    mirrors the _flatten(split_scenes(...)) the chain sampler runs on each, so the preview shows
    exactly the expanded text Studio will append/prepend. Empty in → empty out."""
    if not str(text or "").strip():
        return ""
    split_fn = _funpack_attr("conditioning", "split_scenes")
    s = split_fn(str(text))
    parts = []
    ae = (s.get("anchor_expanded") or "").strip()
    if ae:
        parts.append(ae)
    for sc in (s.get("scenes", []) or []):
        e = (sc.get("expanded") or "").strip()
        if e:
            parts.append(e)
    return " ".join(parts).strip()


def expand_prompt_for_node(text: str, variables=None) -> str:
    """A project text made ready for a node that encodes it ITSELF (a MiniMax H3 fl2va /
    ref2va prompt field driven by a linked input, say).

    Studio does two things to a prompt before it reaches a text encoder: it expands shortcut
    triggers, then resolves `$name` variables — in that order, variables dead last. A node
    outside Studio does neither, so handing it the raw project text would encode the literal
    `/shortcut` triggers and `$name` tokens. This applies both steps, so what that node
    encodes is what Studio would have encoded.

    Peeks at the shortcut revolver rather than committing it: the run's own expansion inside
    Studio is what advances the no-repeat cycle, and doing it twice would skip a slot.
    """
    expanded = expand_prompt_fragment(text)
    if not expanded or not variables:
        return expanded
    resolve_variables = _funpack_attr("templates", "resolve_variables")
    return resolve_variables(expanded, variables)[0]


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


# ComfyUI's own log file, read only to backfill the panel after a restart. Cached briefly:
# recent_log is polled every 1.5s and this must not become a file read per poll.
_LOG_FILE_CACHE: dict = {"at": 0.0, "lines": []}
_LOG_FILE_TTL = 5.0
_LOG_FILE_MAX_BYTES = 512 * 1024


def _comfy_log_file():
    """Path to ComfyUI's log, or None. It is `<user directory>/comfyui.log` — the same path
    the startup banner prints."""
    try:
        import folder_paths
        from pathlib import Path
        path = Path(folder_paths.get_user_directory()) / "comfyui.log"
        return path if path.is_file() else None
    except Exception:  # noqa: BLE001
        return None


def _log_file_tail(limit: int) -> list:
    """The last `limit` lines of ComfyUI's log file, or []. Never raises."""
    import time
    now = time.monotonic()
    if now - _LOG_FILE_CACHE["at"] < _LOG_FILE_TTL:
        return _LOG_FILE_CACHE["lines"][-int(limit):]
    try:
        path = _comfy_log_file()
    except Exception:  # noqa: BLE001
        path = None
    if path is None:
        return []
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > _LOG_FILE_MAX_BYTES:
                fh.seek(size - _LOG_FILE_MAX_BYTES)
                fh.readline()           # drop the partial first line
            lines = fh.read().decode("utf-8", errors="replace").splitlines()
    except Exception:  # noqa: BLE001
        lines = []
    _LOG_FILE_CACHE["at"] = now
    _LOG_FILE_CACHE["lines"] = lines
    return lines[-int(limit):]


def recent_log(limit: int = 500) -> list:
    """Recent backend output, newest last.

    The in-memory buffer only holds what this process printed, so a CRASH-AND-RESTART leaves
    the panel empty — exactly when the lines before the crash are the ones worth reading.
    ComfyUI's own log file survives that, so it backfills whatever the buffer is missing.
    Only consulted while the buffer is short: once this process has produced `limit` lines of
    its own, the file is never touched again.
    """
    limit = int(limit)
    with _LOG_LOCK:
        live = list(_LOG)[-limit:]
    if len(live) >= limit:
        return live
    older = _log_file_tail(limit - len(live))
    if not older:
        return live
    if not live:
        return older
    # The file also contains what is in the buffer. Cut the overlap so the panel does not
    # show the same lines twice across the seam.
    first = live[0]
    for i in range(len(older) - 1, -1, -1):
        if older[i].endswith(first):
            older = older[:i]
            break
    return older + live


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


def revolver_settings() -> dict:
    """Shortcut-revolver settings (enabled + random order). State lives server-side next to
    the shortcut DB so previews and generation share one cycle."""
    return _funpack_attr("templates", "revolver_settings")()


def set_revolver_settings(payload: dict) -> dict:
    fn = _funpack_attr("templates", "set_revolver_settings")
    return fn(
        enabled=payload.get("enabled"),
        random_order=payload.get("random"),
    )


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


def suggestion_stats() -> dict:
    """Mine the user's OWN prompting habits from all saved projects: which shortcuts
    land in the same scene (pairs) and which shot follows which (follows, in timeline
    order). Scene texts keep triggers verbatim (editor prompt model), so this is a
    plain text scan. Derived on demand from project files — no learned state stored.
    Stats are keyed by shortcut `key` (the library's stable id)."""
    import re

    from . import projects as _projects

    si, *_ = _library_fns()
    pats: list[tuple[str, Any]] = []
    for item in si():
        if item.get("enabled") is False:
            continue
        key = str(item.get("key") or item.get("name") or "")
        if not key:
            continue
        for trig in item.get("triggers") or []:
            t = str(trig or "").strip().lower()
            if t:
                pats.append((key, re.compile(r"(?<![a-z0-9_])" + re.escape(t) + r"(?![a-z0-9_])")))
    counts: dict[str, int] = {}
    pairs: dict[tuple, int] = {}
    follows: dict[tuple, int] = {}
    scenes_scanned = 0
    for meta in _projects.list_projects():
        proj = _projects.get(meta.get("id", ""))
        if proj is None:
            continue
        by_id = {s.id: s for s in proj.scenes}
        order = [sid for sid in (proj.timeline_order or []) if sid in by_id]
        if not order:
            order = [s.id for s in proj.scenes]
        prev: set[str] | None = None
        for sid in order:
            sc = by_id[sid]
            # Editorial cuts share the root's text — only the root owns the prompt.
            if sc.excluded or (sc.gen_unit_id and sc.cut_offset_frames):
                continue
            text = (sc.text or "").strip().lower()
            if not text:
                continue
            present = {k for k, pat in pats if pat.search(text)}
            scenes_scanned += 1
            for k in present:
                counts[k] = counts.get(k, 0) + 1
            for a in present:
                for b in present:
                    if a < b:
                        pairs[(a, b)] = pairs.get((a, b), 0) + 1
            if prev:
                for a in prev:
                    for b in present:
                        follows[(a, b)] = follows.get((a, b), 0) + 1
            prev = present
    return {
        "scenes": scenes_scanned,
        "counts": counts,
        "pairs": [[a, b, n] for (a, b), n in pairs.items()],
        "follows": [[a, b, n] for (a, b), n in follows.items()],
    }


def export_shortcuts() -> dict:
    return _funpack_attr("templates", "load_shortcut_db")()


def import_shortcuts(incoming: dict, replace: bool = False) -> dict:
    shortcut_items = _funpack_attr("templates", "shortcut_items")
    load_shortcut_db = _funpack_attr("templates", "load_shortcut_db")
    save_shortcut_db = _funpack_attr("templates", "save_shortcut_db")
    normalize_shortcut_db = _funpack_attr("templates", "normalize_shortcut_db")
    imported = normalize_shortcut_db(incoming)
    data = load_shortcut_db()
    # Replace wipes the existing set (categories included, since they come bundled in
    # the imported db) so the imported file becomes the whole library; Merge overlays.
    if replace:
        data["shortcuts"] = {}
        data["categories"] = imported.get("categories", []) or []
    shortcuts = data.setdefault("shortcuts", {})
    count = 0
    for key, item in imported.get("shortcuts", {}).items():
        if isinstance(item, dict):
            shortcuts[str(key)] = item
            count += 1
    save_shortcut_db(data)
    data = load_shortcut_db()
    return {"imported": count, "shortcuts": shortcut_items(data), "categories": _shortcut_categories()}


def clear_shortcuts() -> dict:
    """Wipe every shortcut (and managed categories) — the Delete-all action."""
    shortcut_items = _funpack_attr("templates", "shortcut_items")
    load_shortcut_db = _funpack_attr("templates", "load_shortcut_db")
    save_shortcut_db = _funpack_attr("templates", "save_shortcut_db")
    data = load_shortcut_db()
    data["shortcuts"] = {}
    data["categories"] = []
    save_shortcut_db(data)
    data = load_shortcut_db()
    return {"shortcuts": shortcut_items(data), "categories": _shortcut_categories()}


def export_transitions() -> dict:
    return _funpack_attr("templates", "load_transition_db")()


def import_transitions(incoming: dict, replace: bool = False) -> dict:
    transition_items = _funpack_attr("templates", "transition_items")
    load_transition_db = _funpack_attr("templates", "load_transition_db")
    save_transition_db = _funpack_attr("templates", "save_transition_db")
    normalize_transition_db = _funpack_attr("templates", "normalize_transition_db")
    imported = normalize_transition_db(incoming)
    data = load_transition_db()
    if replace:
        data["transitions"] = {}
    transitions = data.setdefault("transitions", {})
    count = 0
    for key, item in imported.get("transitions", {}).items():
        if isinstance(item, dict):
            transitions[str(key)] = item
            count += 1
    save_transition_db(data)
    data = load_transition_db()
    return {"imported": count, "transitions": transition_items(data)}


def clear_transitions() -> dict:
    """Wipe every split marker — the Delete-all action."""
    transition_items = _funpack_attr("templates", "transition_items")
    load_transition_db = _funpack_attr("templates", "load_transition_db")
    save_transition_db = _funpack_attr("templates", "save_transition_db")
    data = load_transition_db()
    data["transitions"] = {}
    save_transition_db(data)
    data = load_transition_db()
    return {"transitions": transition_items(data)}


# ── FunPack file manager (Composer ▸ Files) ──────────────────────────────────
# A read/delete view over the on-disk FunPack state so the user can audit and
# purge files the editor created without leaving the UI. Two roots only:
#   • template store  → templates.json / shortcuts.json / promptsplit.json
#   • refinement keys → <key>.json + sidecars (.sampler_ctx.json) + value/latent .pt
# Anything outside these resolved directories is unreachable (no traversal).

def _funpack_file_roots() -> dict:
    template_store_dir = _funpack_attr("templates", "template_store_dir")
    refinement_store_dir = _funpack_attr("templates", "refinement_store_dir")
    return {
        "library": {
            "label": "Prompt library",
            "dir": Path(template_store_dir()),
            "names": ["templates.json", "shortcuts.json", "promptsplit.json", "transitions.json"],
        },
        "refinements": {
            "label": "Refinement keys",
            "dir": Path(refinement_store_dir()),
            "names": None,  # whole directory
        },
    }


def _classify_refinement_file(name: str) -> str:
    low = name.lower()
    if low.endswith(".sampler_ctx.json"):
        return "sampler context"
    if low.endswith(".pt"):
        return "value/latent tensor"
    if low.endswith(".json"):
        return "refinement key"
    return "other"


def list_funpack_files() -> dict:
    groups = []
    for gid, spec in _funpack_file_roots().items():
        d: Path = spec["dir"]
        files = []
        if d.is_dir():
            if spec["names"] is None:
                entries = sorted(p.name for p in d.iterdir() if p.is_file())
            else:
                entries = [n for n in spec["names"] if (d / n).is_file()]
            for n in entries:
                p = d / n
                try:
                    stat = p.stat()
                except OSError:
                    continue
                row = {"name": n, "size": stat.st_size, "mtime": stat.st_mtime}
                if gid == "refinements":
                    row["kind"] = _classify_refinement_file(n)
                files.append(row)
        groups.append({"id": gid, "label": spec["label"], "dir": str(d), "files": files})
    return {"groups": groups}


def _resolve_funpack_file(group: str, name: str) -> Path:
    spec = _funpack_file_roots().get(group)
    if not spec:
        raise ValueError(f"Unknown file group: {group}")
    # Reject any path component — only a bare basename within the root is valid.
    if not name or name != Path(name).name or name in (".", ".."):
        raise ValueError("Invalid file name.")
    d: Path = spec["dir"]
    if spec["names"] is not None and name not in spec["names"]:
        raise ValueError("File is not managed by this group.")
    path = (d / name)
    if path.resolve().parent != d.resolve():
        raise ValueError("File is outside the managed directory.")
    return path


def delete_funpack_file(group: str, name: str) -> dict:
    path = _resolve_funpack_file(group, name)
    if path.is_file():
        path.unlink()
    return list_funpack_files()


def clear_funpack_files(group: str) -> dict:
    spec = _funpack_file_roots().get(group)
    if not spec:
        raise ValueError(f"Unknown file group: {group}")
    d: Path = spec["dir"]
    if d.is_dir():
        names = spec["names"]
        for p in d.iterdir():
            if not p.is_file():
                continue
            if names is not None and p.name not in names:
                continue
            try:
                p.unlink()
            except OSError:
                pass
    return list_funpack_files()


# ── Loopback HTTP to ComfyUI (queue + results) ───────────────────────────────

def _url(path: str) -> str:
    return f"{config.comfy_base_url()}{path}"


async def _session():
    import aiohttp
    return aiohttp.ClientSession()


class ComfyError(RuntimeError):
    pass


# Stable client_id stamped on every editor-queued prompt so a reloaded UI can tell its
# own in-flight generation apart from unrelated ComfyUI jobs (see active_generation).
EDITOR_CLIENT_ID = "funpack-movie-editor"


def _format_prompt_rejection(data: dict, status: int) -> str:
    """Why ComfyUI refused a prompt, in words.

    Its top-level `error` for a validation failure is a fixed blob ("Prompt outputs failed
    validation", details ""), and the part that names the node, the widget and the bad value
    lives in `node_errors` — which the old `error or node_errors` never reached, because the
    generic blob is always truthy. So every rejected prompt reported the same nothing.
    """
    parts = []
    err = data.get("error")
    if isinstance(err, dict):
        head = str(err.get("message") or err.get("type") or "").strip()
        detail = str(err.get("details") or "").strip()
        if head:
            parts.append(f"{head} ({detail})" if detail else head)
    elif err:
        parts.append(str(err))
    for node_id, entry in (data.get("node_errors") or {}).items():
        if not isinstance(entry, dict):
            parts.append(f"node {node_id}: {entry}")
            continue
        who = str(entry.get("class_type") or f"node {node_id}")
        for item in entry.get("errors") or []:
            if not isinstance(item, dict):
                parts.append(f"{who}: {item}")
                continue
            msg = str(item.get("message") or item.get("type") or "").strip()
            detail = str(item.get("details") or "").strip()
            parts.append(f"{who}: {msg}{f' — {detail}' if detail else ''}".strip())
    return "; ".join(p for p in parts if p) or f"HTTP {status}"


async def queue_prompt(graph: dict, client_id: Optional[str] = None,
                       extra_data: Optional[dict] = None) -> dict:
    payload = {"prompt": graph, "client_id": client_id or uuid.uuid4().hex}
    if extra_data:
        payload["extra_data"] = dict(extra_data)
    async with await _session() as s:
        async with s.post(_url("/prompt"), json=payload) as r:
            data = await r.json()
            if r.status >= 400 or data.get("node_errors"):
                raise ComfyError(_format_prompt_rejection(data, r.status))
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
    """Sampler step counter, plus what the Chain Sampler says it is doing right now.

    ComfyUI's progress channel is numbers only, so a chain that samples several scenes —
    some of them twice, with a second pass — is one anonymous bar. The sampler publishes a
    short phase label ("scene 2/3 · pass 2 of 2") that rides along here. Absent (an older
    sampler, a non-FunPack job) it is simply an empty string and the UI shows the numbers
    alone, exactly as before.
    """
    _install_progress_hook()
    label = ""
    try:
        # Read run_phase's state straight off `sys` — deliberately NO import on this path.
        # The editor polls this every 700ms DURING sampling, on ComfyUI's event loop, so it
        # must do nothing that can block behind the worker thread. run_phase keeps its state
        # there anyway (its writer and any reader can hold different module objects), which
        # makes the direct read both the cheapest and the correct one.
        _phase = getattr(sys, "_funpack_run_phase", None)
        if isinstance(_phase, dict):
            label = str(_phase.get("label") or "")
    except Exception:  # noqa: BLE001 — a readout must never break the progress poll
        pass
    return {"value": _progress["value"], "max": _progress["max"], "label": label}


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


def _queue_item_extra(item: list) -> dict:
    """extra_data dict from a /queue tuple [number, prompt_id, prompt, extra_data, ...]."""
    extra = item[3] if len(item) > 3 else None
    return extra if isinstance(extra, dict) else {}


def _select_active(queue_state: dict) -> dict:
    """Pick the editor's own in-flight generation out of a /queue snapshot.

    Filters by EDITOR_CLIENT_ID so unrelated ComfyUI jobs are ignored, and reads back the
    ``funpack`` payload (project id + target scenes) stamped at queue time so a reloaded
    client can re-attach and record the result onto the right scenes."""
    running, pending = [], []
    for item in queue_state.get("queue_running", []) or []:
        if _queue_item_extra(item).get("client_id") == EDITOR_CLIENT_ID:
            running.append(item)
    for item in queue_state.get("queue_pending", []) or []:
        if _queue_item_extra(item).get("client_id") == EDITOR_CLIENT_ID:
            pending.append(item)
    out = {"running": bool(running), "prompt_id": None, "pid": None,
           "scene_ids": [], "only_scene": None, "pending": len(pending)}
    if running:
        item = running[0]
        out["prompt_id"] = item[1] if len(item) > 1 else None
        fp = _queue_item_extra(item).get("funpack") or {}
        if isinstance(fp, dict):
            out["pid"] = fp.get("pid")
            out["scene_ids"] = fp.get("scene_ids") or []
            out["only_scene"] = fp.get("only_scene")
    return out


async def active_generation() -> dict:
    """The editor's own in-flight generation, recovered from ComfyUI's queue.

    Survives a UI reload: the queue lives in the (still-running) ComfyUI process, so the
    reloaded client can re-block Generate, expose Interrupt, and record the result when it
    finishes. See _select_active for the filtering."""
    async with await _session() as s:
        async with s.get(_url("/queue")) as r:
            r.raise_for_status()
            state = await r.json()
    return _select_active(state)


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
