"""Which modules run: the project's own switch, and a module's persistent quarantine.

A module that raises mid-run is dropped for that run (core/patching.py). That alone leaves a
faulty module to fail again on every generation, so a fault is also written down here and the
module stays OFF until it is repaired -- its code changed -- or the person turns it back on.
The project's switch is the person's own: a module they turned off vanishes for that project.

Only modules that MODIFY a run can be switched (they provide `modifier` or `sampler_modifier`).
Loaders, the sampler and the like are structure: a pipeline without them is not a pipeline.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from . import config, log, patching

#: Keys of the settings payload that are not module ids.
RESERVED = frozenset({"_off"})

_TRANSIENT = ("Interrupt", "OutOfMemory", "OOM")        # not the module's fault: say nothing, remember nothing
_lock = threading.Lock()


class Unavailable(RuntimeError):
    """A module cannot run HERE (an older ComfyUI, a missing package): the person's world, not a
    fault in the module, so it is said but never quarantined."""


def controllable(spec) -> bool:
    return bool(spec.provides.get("modifier") or spec.provides.get("sampler_modifier"))


def off_by_project(settings) -> set:
    """Module ids the project turned off, read from the settings payload."""
    raw = settings.get("_off") if isinstance(settings, dict) else None
    ids = raw.get("modules") if isinstance(raw, dict) else None
    return {i for i in ids if isinstance(i, str)} if isinstance(ids, list) else set()


def bad_off(raw) -> str | None:
    """Why a payload's `_off` entry is malformed, or None."""
    if raw is None:
        return None
    ids = raw.get("modules") if isinstance(raw, dict) else None
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        return '"_off" must be {"modules": ["module_id", ...]}'
    return None


# --- the quarantine file ---------------------------------------------------


def _file() -> Path:
    return Path(config.QUARANTINE_FILE)


_mem: Dict[str, dict] = {}       # what could not be written to disk: still off, for this session
_loaded: Dict[str, str] = {}     # module id -> signature of the code this process imported


def _read() -> Dict[str, dict]:
    try:
        raw = json.loads(_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    found = {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, dict)} if isinstance(raw, dict) else {}
    return {**found, **_mem}


def _write(entries: Dict[str, dict]) -> None:
    path = _file()
    tmp = path.with_suffix(".tmp")
    _mem.clear()
    try:
        if not entries:
            path.unlink(missing_ok=True)
            return
        tmp.write_text(json.dumps(entries, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        _mem.update(entries)
        log.warning("module control", f"could not write {path.name}: {exc}. The module is off for this "
                                      f"session only.")


def signature(spec) -> str:
    """Changes when the module's code does: that is what "repaired" means."""
    mod = sys.modules.get(spec.source)
    where = getattr(mod, "__file__", "") or ""
    base = Path(where).parent
    parts = []
    if where and base.is_dir():
        for p in sorted(base.rglob("*.py")):
            if "tests" in p.parts or "__pycache__" in p.parts:
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            parts.append(f"{p.name}:{st.st_mtime_ns}:{st.st_size}")
    return "|".join(parts)


def remember_loaded(spec) -> None:
    """Called when a module is imported: the code this process is actually running."""
    _loaded.setdefault(spec.id, signature(spec))


def quarantined(specs: Iterable = ()) -> Dict[str, dict]:
    """{module id: {reason, when}} for modules still quarantined. One whose code has changed since
    it failed is released here, and said."""
    by_id = {s.id: s for s in specs}
    with _lock:
        entries = _read()
        kept = {}
        for mid, entry in entries.items():
            spec = by_id.get(mid)
            if spec is not None and entry.get("sig"):
                now = signature(spec)
                # Repaired = the code on disk changed AND this process has loaded it. Edited but not
                # restarted still runs the old code, which would only fail again.
                if entry["sig"] != now and _loaded.get(mid, now) == now:
                    log.info("module control", f"{mid} was changed since it failed: trying it again.")
                    continue
                if entry["sig"] != now:
                    entry = {**entry, "restart": True}
            kept[mid] = entry
        if kept != entries:
            _write(kept)
    return kept


def quarantine(spec, reason: str) -> None:
    with _lock:
        entries = _read()
        if spec.id in entries:
            return
        entries[spec.id] = {"reason": str(reason)[:400], "when": time.strftime("%Y-%m-%d %H:%M"),
                            "sig": _loaded.get(spec.id) or signature(spec)}
        _write(entries)
    log.alert("module control", f"{spec.id} failed and is now OFF until it is repaired or you turn it "
                                f"back on (Settings ▸ Modules): {reason}")


def release(module_id: str) -> bool:
    with _lock:
        entries = _read()
        if module_id not in entries:
            return False
        del entries[module_id]
        _write(entries)
    return True


def _transient(exc: BaseException) -> bool:
    return any(t in type(exc).__name__ for t in _TRANSIENT) or "out of memory" in str(exc).lower()


def fault(key: str, exc: BaseException) -> None:
    """Called by core/patching.Dropped when a module's hook raised. Faults that are not the module's
    (an interrupt, running out of memory) are left alone."""
    if _transient(exc):
        return
    mid = key[len("funpack."):].split(".")[0] if key.startswith("funpack.") else None
    if not mid:
        return
    from . import registry
    spec = registry.current().specs.get(mid)
    if spec is not None and controllable(spec):
        quarantine(spec, f"{type(exc).__name__}: {exc}")


patching.on_fault = fault


def start_failed(spec, exc: BaseException) -> None:
    """A module that raised while installing or starting up for a run."""
    if controllable(spec) and not _transient(exc) and not isinstance(exc, Unavailable):
        quarantine(spec, f"{type(exc).__name__}: {exc}")


# --- what a run uses -------------------------------------------------------


def partition(specs: Iterable, settings) -> Tuple[List, List[str]]:
    """(kept, notes): `specs` without the ones the project turned off or that are quarantined."""
    specs = list(specs)
    off = off_by_project(settings)
    held = quarantined(specs)
    kept, notes = [], []
    for spec in specs:
        if controllable(spec):
            if spec.id in off:
                notes.append(f"{spec.id}: turned off for this project")
                continue
            if spec.id in held:
                notes.append(f"{spec.id}: OFF -- it failed on {held[spec.id].get('when', '?')} "
                             f"({held[spec.id].get('reason', '')}). "
                             + ("It has been edited since: restart ComfyUI to try it again, or turn it back on "
                                "in Settings ▸ Modules." if held[spec.id].get("restart")
                                else "Turn it back on in Settings ▸ Modules."))
                continue
        kept.append(spec)
    return kept, notes


def skipped(settings) -> set:
    """Module ids whose own values are not checked or used: off for the project, or quarantined."""
    from . import registry
    specs = registry.current().specs
    held = quarantined(specs.values())
    return {i for i in off_by_project(settings) | set(held) if i in specs and controllable(specs[i])}


def fingerprint() -> str:
    """Part of a node's cache key: a module going into or out of quarantine changes what a run does."""
    from . import registry
    return ",".join(sorted(quarantined(registry.current().specs.values())))


def state(specs: Iterable) -> Dict[str, dict]:
    """What the Modules panel shows: {id: {controllable, quarantine?}} for every module."""
    specs = list(specs)
    held = quarantined(specs)
    return {s.id: {"controllable": controllable(s), **({"quarantine": held[s.id]} if s.id in held else {})}
            for s in specs}
