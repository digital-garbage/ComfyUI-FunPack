"""Which model family a checkpoint file is, without loading it.

Core reads the file -- a safetensors header is a length-prefixed JSON object
whose keys are tensor names, readable in kilobytes regardless of whether the
file itself is 2GB or 40GB -- and that is ALL it knows how to do. Whether a
set of tensor names belongs to MiniMax H3, or anything else, is a model
module's own business: asking one to identify itself is the same "detect"
capability every model module can already offer, the same shape as the
`empty_latent` and `decode` capabilities modules/models/minimax_h3 provides.
Core aggregates whichever module claims a file and says plainly when none
does -- it is never the one deciding what a family IS called.

Only .safetensors today: no loader in this pack can read a GGUF file yet, so
detecting one accurately would be groundwork for a format nothing here can
use.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import List, Optional

from . import registry as registry_mod

#: A header past this size is not a safetensors file -- reading a wild length
#: prefix as a real one would try to allocate however many gigabytes a
#: corrupt or unrelated file's first eight bytes happen to spell out.
MAX_HEADER_BYTES = 256 * 1024 * 1024


def read_safetensors_keys(path: Path) -> Optional[List[str]]:
    """Tensor names out of a safetensors header, or None when this is not one.

    Layout: an 8-byte little-endian length, then that many bytes of JSON whose
    top-level keys are the tensor names (plus `__metadata__`, dropped here).
    Nothing past the header is read.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return None
            (length,) = struct.unpack("<Q", raw)
            if length <= 0 or length > MAX_HEADER_BYTES:
                return None
            header = json.loads(f.read(length).decode("utf-8", errors="replace"))
    except (OSError, ValueError, json.JSONDecodeError, struct.error):
        return None
    if not isinstance(header, dict):
        return None
    return [k for k in header if k != "__metadata__"]


def resolve_diffusion_model(filename: str) -> Optional[Path]:
    """Where `filename` lives, across the folders a FunPack loader can
    actually read a diffusion model from -- `modules/loaders/diffusion_model`
    offers files from "diffusion_models", `modules/loaders/checkpoint` from
    "checkpoints"; nothing in this app reads a model from a "unet" folder,
    so probing one would risk answering for a file no loader here would ever
    actually load under that name. None when it is nowhere findable -- not
    this function's job to say why.
    """
    try:
        import folder_paths
    except ImportError:                                # not inside ComfyUI
        return None
    for folder in ("diffusion_models", "checkpoints"):
        try:
            found = folder_paths.get_full_path(folder, filename)
        except Exception:                              # noqa: BLE001
            found = None
        if found:
            return Path(found)
    return None


def detect(path: Path, registry=None) -> dict:
    """Inspect one checkpoint file.

    Returns ``{"module", "title", "detected", "reason"}``. `module` is the id
    of whichever model module claimed it, or None -- and None is a real
    answer, not a failure to try hard enough: it means this install has no
    module that recognises the file, and the caller should say so rather
    than guess a family. `reason` is plain text for a person, always
    populated.
    """
    p = Path(path)
    if not p.is_file():
        return {"module": None, "title": None, "detected": False,
                "reason": f"{p.name}: file not found"}
    if p.suffix.lower() != ".safetensors":
        return {"module": None, "title": None, "detected": False,
                "reason": f"{p.name}: only .safetensors can be inspected without loading it"}
    keys = read_safetensors_keys(p)
    if keys is None:
        return {"module": None, "title": None, "detected": False,
                "reason": f"{p.name}: not a readable safetensors header"}

    keyset = set(keys)
    reg = registry or registry_mod.current()
    for spec, claims in reg.providers("detect"):
        try:
            matched = claims(keyset)
        except Exception as exc:                       # noqa: BLE001
            from . import log
            log.failed(f"{spec.id}.detect", exc)
            continue
        if matched:
            return {"module": spec.id, "title": spec.title, "detected": True,
                    "reason": f"{p.name}: {spec.title}"}
    return {"module": None, "title": None, "detected": False,
            "reason": f"{p.name}: no installed model module recognises this file"}
