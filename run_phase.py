"""What the sampler is doing RIGHT NOW, as a short label a UI can poll.

ComfyUI's progress channel carries only (value, max) — no text — so a chain that samples
several scenes, some of them twice, reads as one undifferentiated bar: you can watch the
number climb without knowing which scene it belongs to or whether the second pass has
started. The Chain Sampler publishes a label here as it moves through the run, and the
Movie Editor's ``/api/progress`` hands it to the UI alongside the numbers.

**The state lives on `sys`, not in this module's globals, and that is load-bearing.** The
writer (samplers.py) reaches this file as a package-relative import, so it gets
``ComfyUI-FunPack.run_phase``; the reader (movie_editor/backend/bridge.py) imports FunPack
modules top-level by name after putting the root on sys.path, so it would get a SECOND,
separate ``run_phase`` module object — two copies, two sets of globals, and a label that
never arrives. Hanging the dict off ``sys`` (unambiguously one object per process) makes
both import paths share it. Stateless helpers like `conditioning` do not care which copy
they are; anything holding mutable state across that boundary does.

Everything here is BEST-EFFORT and never affects sampling: it is a live readout, not state
anything is derived from. Nothing reads it back to make a decision, so a label left over
from a crashed run is cosmetic — the next run overwrites it and ``clear()`` empties it —
and it is not the kind of cross-request cache that would violate "the UI is the only source
of truth".
"""
import sys

# Part of this module's contract, not an implementation detail: movie_editor's bridge READS
# THIS ATTRIBUTE DIRECTLY rather than importing this module. That poll runs every 700ms on
# ComfyUI's event loop while the worker thread is sampling, so it must not touch the import
# system or anything else that can block. Renaming this key means updating bridge.py too
# (tests/test_run_phase.py pins the two together).
_SYS_KEY = "_funpack_run_phase"


def _state() -> dict:
    st = getattr(sys, _SYS_KEY, None)
    if not isinstance(st, dict):
        st = {"label": "", "seq": 0}
        setattr(sys, _SYS_KEY, st)
    return st


def set_phase(label: str) -> None:
    """Publish the current phase, e.g. "scene 2/3 · pass 2 of 2"."""
    try:
        st = _state()
        st["label"] = str(label or "")
        st["seq"] += 1
    except Exception:  # noqa: BLE001 — a readout must never break a render
        pass


def clear() -> None:
    """Nothing is sampling. Call from a finally, so an interrupt doesn't leave a label up."""
    set_phase("")


def current() -> dict:
    """{"label": str, "seq": int}. `seq` only ever increases, so a poller can tell "still
    the same phase" from "this phase came round again"."""
    try:
        st = _state()
        return {"label": st["label"], "seq": st["seq"]}
    except Exception:  # noqa: BLE001
        return {"label": "", "seq": 0}
