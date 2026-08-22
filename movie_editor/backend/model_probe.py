"""Read a diffusion checkpoint's architecture out of its file, without loading it.

The model family decides which nodes the built-in pipeline uses and which ports the loaders
wire into, so it has to be known BEFORE the graph is built — long before ComfyUI would load
any weights. It used to be a radio button, which meant a user could select LTX, load an H3
checkpoint, and get a graph wired for the wrong model with no error: the mismatch surfaced as
a stray port rather than as a family problem.

Everything needed is in the safetensors header. The architecture signatures ComfyUI's own
`model_detection.py` uses are pure KEY NAMES — no tensor values, no shapes — so reading the
header's key list is enough, and costs the same on a 2 GB file as on a 40 GB one.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterable, Optional

# Prefixes a diffusion model's tensors can sit behind, depending on how the file was packed.
_PREFIXES = ("", "model.diffusion_model.", "diffusion_model.")

# Mirrors comfy/model_detection.py. Key names only — deliberately no shape checks, so this
# cannot disagree with core about what a file IS, only about how much of it we can read.
_H3_KEYS = ("video_patch_proj.weight", "audio_patch_proj.weight")
_LTX_KEY = "adaln_single.emb.timestep_embedder.linear_1.bias"
_LTX_AUDIO_KEY = "audio_adaln_single.linear.weight"

# What FunPack's pipeline wiring understands. `ltxv` (video-only Lightricks) has no wiring of
# its own and runs on the ltxav graph, so it maps there — but the caller is told which it was,
# because an audio-less checkpoint on an AV graph is worth saying out loud.
FAMILY_BY_ARCH = {"minimax_h3": "minimax_h3", "ltxav": "ltxav", "ltxv": "ltxav"}


def read_safetensors_keys(path: str | Path) -> Optional[list[str]]:
    """Tensor names from a safetensors header, or None if this is not a readable one.

    Header layout: 8-byte little-endian length, then that many bytes of JSON whose top-level
    keys are the tensor names (plus `__metadata__`). Nothing after the header is touched.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return None
            (length,) = struct.unpack("<Q", raw)
            # A sane header is kilobytes to a few megabytes; a wild value means this is not a
            # safetensors file and we should not try to allocate it.
            if length <= 0 or length > 256 * 1024 * 1024:
                return None
            header = json.loads(f.read(length).decode("utf-8", errors="replace"))
    except (OSError, ValueError, json.JSONDecodeError, struct.error):
        return None
    if not isinstance(header, dict):
        return None
    return [k for k in header if k != "__metadata__"]


def _is_gguf_container(path: Path) -> bool:
    """A GGUF renamed to .safetensors is common enough to check for: the pickers only
    offered .safetensors until recently, so renaming was the obvious way in. Four bytes."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def read_gguf_keys(path: str | Path) -> Optional[list[str]]:
    """Tensor names from a GGUF file, or None when they cannot be read.

    Delegated to llama.cpp's own `gguf` package rather than hand-parsing the container:
    the header is a typed key-value section of variable-length records, and getting that
    subtly wrong would misidentify a model rather than fail. The package is optional, so
    None here means "cannot tell", never "wrong family".
    """
    try:
        import gguf  # noqa: PLC0415
    except ImportError:
        return None
    try:
        reader = gguf.GGUFReader(str(path))
        return [str(t.name) for t in reader.tensors]
    except Exception:  # noqa: BLE001 — any malformed container is simply unreadable
        return None


def detect_arch(keys: Iterable[str]) -> Optional[str]:
    """`minimax_h3` | `ltxav` | `ltxv`, or None when the keys match no architecture we wire.

    None is a real answer and must not be treated as LTX: it means the file is something this
    pipeline cannot build a graph for, and the caller should say so rather than guess.
    """
    keyset = set(keys or ())
    if not keyset:
        return None
    for prefix in _PREFIXES:
        if all(f"{prefix}{k}" in keyset for k in _H3_KEYS):
            return "minimax_h3"
        if f"{prefix}{_LTX_KEY}" in keyset:
            return "ltxav" if f"{prefix}{_LTX_AUDIO_KEY}" in keyset else "ltxv"
    return None


def detect_family(path: str | Path) -> dict:
    """Inspect one checkpoint.

    Returns ``{"family", "arch", "detected", "reason"}``:
      family   — what to wire the pipeline for, or None when unknown
      arch     — the architecture actually found (`ltxv` is distinct from `ltxav` here)
      detected — whether the file answered at all
      reason   — plain text for the user, always populated
    """
    p = Path(path)
    if not p.is_file():
        return {"family": None, "arch": None, "detected": False,
                "reason": f"{p.name}: file not found"}
    if p.suffix.lower() == ".gguf" or _is_gguf_container(p):
        keys = read_gguf_keys(p)
        if keys is None:
            return {"family": None, "arch": None, "detected": False,
                    "reason": f"{p.name}: a .gguf can only be inspected with the `gguf` package "
                              f"installed — set the family from Models ▸ Model family instead"}
        return _from_keys(p, keys)
    if p.suffix.lower() != ".safetensors":
        # .ckpt / .pth are pickles; reading them means executing them, which is not worth it
        # for a family probe. The declared family stands and the caller is told why.
        return {"family": None, "arch": None, "detected": False,
                "reason": f"{p.name}: only .safetensors can be inspected without loading it"}
    keys = read_safetensors_keys(p)
    if keys is None:
        return {"family": None, "arch": None, "detected": False,
                "reason": f"{p.name}: not a readable safetensors header"}
    return _from_keys(p, keys)


def _from_keys(p: Path, keys: Iterable[str]) -> dict:
    """Same verdict from either container: the signatures are key names, so it does not
    matter which format they were read out of."""
    arch = detect_arch(keys)
    if arch is None:
        return {"family": None, "arch": None, "detected": False,
                "reason": f"{p.name}: no LTX or MiniMax H3 signature — this pipeline has no "
                          f"wiring for it"}
    family = FAMILY_BY_ARCH[arch]
    if arch == "ltxv":
        return {"family": family, "arch": arch, "detected": True,
                "reason": f"{p.name}: Lightricks LTX (video only — the pipeline still wires "
                          f"the AV graph, so the audio branch will have nothing to decode)"}
    label = "MiniMax H3" if arch == "minimax_h3" else "Lightricks LTX-AV"
    return {"family": family, "arch": arch, "detected": True,
            "reason": f"{p.name}: {label}"}


# Widget names that can hold a diffusion model's file, across FunPack's own loader and the
# third-party ones an imported workflow might use. Mirrors pipeline_wiring.SEEDED_FILE_INPUTS.
_FILE_INPUTS = ("model_name", "unet_name", "ckpt_name")


def _default_resolver(filename: str) -> Optional[str]:
    try:
        import folder_paths
    except ImportError:
        return None
    for folder in ("diffusion_models", "unet", "checkpoints"):
        try:
            found = folder_paths.get_full_path(folder, filename)
        except Exception:
            found = None
        if found:
            return found
    return None


def diffusion_model_file(models: dict) -> Optional[str]:
    """The checkpoint filename the pipeline's diffusion-model slot points at."""
    if not isinstance(models, dict):
        return None
    for slot in models.get("slots") or []:
        if not isinstance(slot, dict):
            continue
        cls = str(slot.get("node_class") or "")
        if slot.get("role") != "unet" and "DiffusionModel" not in cls and "UNETLoader" not in cls:
            continue
        inputs = slot.get("inputs") or {}
        for name in _FILE_INPUTS:
            value = inputs.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def probe_models(models: dict, resolve=None) -> dict:
    """Detect the family for a pipeline config's own diffusion model.

    Returns the same shape as `detect_family`, plus `file`. A failure to detect NEVER
    proposes a family: the caller keeps whatever was already set and shows the reason. The
    alternative — guessing LTX — is the bug this replaced.
    """
    resolve = resolve or _default_resolver
    filename = diffusion_model_file(models)
    if not filename:
        return {"family": None, "arch": None, "detected": False, "file": None,
                "reason": "no diffusion model selected yet"}
    path = resolve(filename)
    if not path:
        return {"family": None, "arch": None, "detected": False, "file": filename,
                "reason": f"{filename}: not found in the models folders"}
    out = detect_family(path)
    out["file"] = filename
    return out
