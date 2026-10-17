"""Optional features that need packages the pack does not force on anyone.

A person switches one on in Settings ▸ Modules; nothing is installed then. The wish is written to
addons.json, and the next ComfyUI start installs what is missing (before any module imports it),
through the same never-moves-anything-installed path an update uses (update.install_lines).

State of an add-on: available (not wanted, not installed), pending (wanted, not installed yet),
installed (wanted and installed), disabled (installed, but switched off). Code that depends on one
asks `enabled(id)`. An add-on this machine cannot run (no NVIDIA GPU) is not offered at all.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import platform
import sys
import threading

from . import config, log

_lock = threading.Lock()

_SPACY_MODEL = ("en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/"
                "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl")


def _nvidia_gpu() -> bool:
    """nvidia-vfx ships wheels for x86-64 Linux and Windows only, and needs a CUDA card."""
    if sys.platform not in ("linux", "win32") or platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.version.cuda and torch.cuda.get_device_capability() >= (7, 5))   # RTX starts at Turing
    except Exception:  # noqa: BLE001
        return False


ADDONS = (
    {"id": "nvidia_vsr", "title": "NVIDIA RTX Super Resolution",
     "hint": "Native upscaling for images and videos. Installs NVIDIA's own package, under its licence.",
     "requires": ("nvidia-vfx>=0.2",), "modules": ("nvvfx",), "offered": _nvidia_gpu},
    {"id": "camera", "title": "Automatic camera and focus",
     "hint": "Lets Shot camera aim camera moves, cuts, transitions and focus at the right things in your prompts, using spaCy.",
     "requires": ("spacy>=3.8,<3.9", _SPACY_MODEL), "modules": ("spacy", "en_core_web_sm"), "offered": lambda: True},
    {"id": "lora_training", "title": "LoRA training",
     "hint": "Train your own LoRA from your media and rated results, with a tag editor and automatic tagging (onnxruntime). Training runs on ComfyUI's own trainer.",
     "requires": ("onnxruntime>=1.17",), "modules": ("onnxruntime",), "offered": lambda: True,
     "legacy": False},          # onnxruntime is common: having it must not switch the feature on by itself
)
_BY_ID = {a["id"]: a for a in ADDONS}


def _present(addon) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in addon["modules"])


def _read() -> dict | None:
    try:
        data = json.loads(config.ADDONS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write(data: dict) -> None:
    tmp = config.ADDONS_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp, config.ADDONS_FILE)
    except OSError as exc:
        log.warning("add-ons", f"could not write {config.ADDONS_FILE.name}: {exc}. The choice is lost on restart.")


def _wanted(data) -> set:
    """What was asked for. With no file yet, whatever is already installed counts as wanted: the packs
    people installed before add-ons existed must not go quiet."""
    if data is not None and isinstance(data.get("wanted"), list):
        return {i for i in data["wanted"] if i in _BY_ID}
    return {a["id"] for a in ADDONS if a.get("legacy", True) and _present(a)}


def enabled(addon_id: str) -> bool:
    """Wanted and installed: the only state in which code may use the add-on."""
    a = _BY_ID.get(addon_id)
    return bool(a) and a["offered"]() and addon_id in _wanted(_read()) and _present(a)


def listing() -> list[dict]:
    """[{id, title, hint, state, failed}] for every add-on this machine can run."""
    data = _read()
    wanted, failed = _wanted(data), (data or {}).get("failed") or {}
    out = []
    for a in ADDONS:
        if not a["offered"]():
            continue
        here, want = _present(a), a["id"] in wanted
        out.append({"id": a["id"], "title": a["title"], "hint": a["hint"],
                    "state": ("installed" if here else "available" if failed.get(a["id"]) else "pending") if want else ("disabled" if here else "available"),   # a failed install is offered again, not "pending"
                    "failed": failed.get(a["id"]) if not here else None})
    return out


def set_wanted(addon_id: str, on: bool) -> bool:
    """Record the wish. False when there is no such add-on here. Turning one on clears its failure, which is how a retry is asked for."""
    a = _BY_ID.get(addon_id)
    if a is None or not a["offered"]():
        return False
    with _lock:
        data = _read() or {}
        wanted = _wanted(_read())
        wanted = wanted | {addon_id} if on else wanted - {addon_id}
        failed = dict(data.get("failed") or {})
        if on:
            failed.pop(addon_id, None)
        _write({"wanted": sorted(wanted), "failed": failed})
    return True


def install_wanted() -> None:
    """At ComfyUI start: install what is wanted and missing. Never raises, and never retries a failure
    on its own (a broken install would add minutes to every start); toggling the add-on retries."""
    try:
        with _lock:
            data = _read() or {}
            failed = dict(data.get("failed") or {})
            wanted = _wanted(_read())
            todo = [a for a in ADDONS if a["id"] in wanted and a["offered"]() and not _present(a) and a["id"] not in failed]
            if not todo:
                return
            from . import update
            for a in todo:
                log.info("add-ons", f"installing {a['title']}…")
                result = update.install_lines(a["requires"])
                importlib.invalidate_caches()
                if _present(a):
                    continue
                failed[a["id"]] = (result.get("detail") or "the install did not finish")[-400:]
                log.alert("add-ons", f"{a['title']} could not be installed: {failed[a['id']]}")
            _write({"wanted": sorted(wanted), "failed": failed})
    except Exception as exc:  # noqa: BLE001
        log.failed("core.addons", exc)
