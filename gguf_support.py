"""Load .gguf model files through FunPack's own loaders.

GGUF is two problems, and only the first is small:

1. **Reading the container.** A header, tensor metadata, then quantized blocks. The `gguf`
   package (llama.cpp's own) does this, and ships a reference dequantizer for every quant
   type it defines.

2. **Using the weights without dequantizing them.** The reason people reach for GGUF is that
   a Q4 model occupies a quarter of the VRAM — which only holds if the weights STAY
   quantized and are expanded per-layer during the forward pass. That needs a custom
   torch ops class, and hand-written kernels per quant type.

So this module does not hand-roll (2). It picks the best backend actually present:

  * **ComfyUI-GGUF installed** — use it. It is the mature implementation of (2), and its
    output feeds `comfy.sd.load_diffusion_model_state_dict` exactly like ours does, so
    FunPack's loader stays the only UI while its runtime does the quantized maths.
  * **`gguf` package only** — read and dequantize at LOAD time, then hand over an ordinary
    state dict. Fully native and correct, but the model occupies its dequantized size, so
    the VRAM saving is gone. Said out loud rather than left to be discovered.
  * **Neither** — refuse with the two ways to fix it, instead of a stack trace.

Nothing here downloads or installs anything.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Optional

GGUF_EXT = ".gguf"


def is_gguf(name: str) -> bool:
    return str(name or "").lower().endswith(GGUF_EXT)


# ── finding .gguf files ───────────────────────────────────────────────────────

def gguf_names(folder: str) -> list[str]:
    """`.gguf` files under a ComfyUI model folder, relative-named like get_filename_list().

    Core's `supported_pt_extensions` has no `.gguf`, so `get_filename_list` never returns
    one — the files are on disk and simply invisible to every picker. Scanned directly
    rather than by mutating core's global extension set, which would change what every
    other node in the install offers.
    """
    try:
        import folder_paths
    except ImportError:
        return []
    out: list[str] = []
    seen: set[str] = set()
    roots = list(folder_paths.get_folder_paths(folder) or [])
    # ComfyUI-GGUF registers its own folders; include them under the same picker so a user
    # who already keeps files there does not have to move them.
    for extra in ("unet_gguf" if folder == "diffusion_models" else "clip_gguf",):
        try:
            roots += list(folder_paths.get_folder_paths(extra) or [])
        except Exception:
            pass
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root, followlinks=True):
            for f in files:
                if not is_gguf(f):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, f), root)
                if rel not in seen:
                    seen.add(rel)
                    out.append(rel)
    return sorted(out)


def gguf_path(folder: str, name: str) -> Optional[str]:
    """Absolute path of a `.gguf` picked out of `gguf_names`."""
    try:
        import folder_paths
    except ImportError:
        return None
    roots = list(folder_paths.get_folder_paths(folder) or [])
    for extra in ("unet_gguf" if folder == "diffusion_models" else "clip_gguf",):
        try:
            roots += list(folder_paths.get_folder_paths(extra) or [])
        except Exception:
            pass
    for root in roots:
        cand = os.path.join(root or "", name)
        if os.path.isfile(cand):
            return cand
    return None


# ── backends ──────────────────────────────────────────────────────────────────

def _load_pack_module(dirname: str, mod: str):
    """Import one module out of a custom-node package by path.

    Custom node folders are not importable by name (``ComfyUI-GGUF`` is not an identifier),
    so the file is loaded directly.
    """
    spec = importlib.util.spec_from_file_location(
        f"_funpack_gguf_{mod}", os.path.join(dirname, f"{mod}.py"))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pack_dir() -> Optional[str]:
    """ComfyUI-GGUF's directory, identified by what it CONTAINS rather than by its name —
    the folder gets renamed by Manager, by forks, and by hand."""
    try:
        import folder_paths
        base = os.path.join(folder_paths.base_path, "custom_nodes")
    except Exception:
        return None
    if not os.path.isdir(base):
        return None
    for entry in sorted(os.listdir(base)):
        d = os.path.join(base, entry)
        loader = os.path.join(d, "loader.py")
        if not os.path.isfile(loader) or not os.path.isfile(os.path.join(d, "ops.py")):
            continue
        try:
            with open(loader, "r", encoding="utf-8", errors="replace") as fh:
                if "gguf_sd_loader" in fh.read():
                    return d
        except OSError:
            continue
    return None


def backend() -> Optional[str]:
    """"pack" (quantized in VRAM), "native" (dequantized at load), or None."""
    if _pack_dir():
        return "pack"
    if importlib.util.find_spec("gguf") is not None:
        return "native"
    return None


UNAVAILABLE = (
    "No GGUF backend is available. Either install ComfyUI-GGUF (custom node pack — keeps the "
    "weights quantized in VRAM, which is the point of GGUF), or `pip install gguf` into "
    "ComfyUI's environment for a load-time dequantize that works but gives back no memory."
)


def load_state_dict(path: str) -> tuple[dict, dict, str]:
    """Read one .gguf into (state_dict, model_options, note).

    `model_options` carries the custom operations the quantized path needs; it is empty for
    the native path, whose tensors are ordinary ones by the time they are returned.
    """
    which = backend()
    if which is None:
        raise RuntimeError(UNAVAILABLE)
    if which == "pack":
        d = _pack_dir()
        loader = _load_pack_module(d, "loader")
        ops = _load_pack_module(d, "ops")
        sd = loader.gguf_sd_loader(path)
        options = {}
        ggml_ops = getattr(ops, "GGMLOps", None)
        if ggml_ops is not None:
            options["custom_operations"] = ggml_ops()
        return sd, options, f"gguf: quantized (ComfyUI-GGUF at {os.path.basename(d)})"
    return _load_native(path)


def load_clip_state_dict(path: str) -> tuple[dict, dict, str]:
    """Same as `load_state_dict`, for a text encoder.

    ComfyUI-GGUF reads encoders through a separate entry point: an encoder's GGUF carries
    llama.cpp tensor names that have to be mapped back to the transformer's, which its
    diffusion-model reader does not do.
    """
    which = backend()
    if which is None:
        raise RuntimeError(UNAVAILABLE)
    if which == "pack":
        d = _pack_dir()
        loader = _load_pack_module(d, "loader")
        ops = _load_pack_module(d, "ops")
        read = getattr(loader, "gguf_clip_loader", None) or loader.gguf_sd_loader
        sd = read(path)
        options = {}
        ggml_ops = getattr(ops, "GGMLOps", None)
        if ggml_ops is not None:
            options["custom_operations"] = ggml_ops()
        return sd, options, f"gguf: quantized (ComfyUI-GGUF at {os.path.basename(d)})"
    return _load_native(path)


def _load_native(path: str) -> tuple[dict, dict, str]:
    """Dequantize every tensor to torch at load time using gguf's reference implementation.

    Correct but not thrifty: the model ends up at its dequantized size, so this buys the
    ability to LOAD a .gguf, not GGUF's memory saving. The caller states that.
    """
    import gguf
    import torch

    reader = gguf.GGUFReader(path)
    sd: dict = {}
    quantized = 0
    for tensor in reader.tensors:
        name = str(tensor.name)
        qtype = tensor.tensor_type
        if qtype in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            arr = tensor.data
        else:
            # gguf.quants.dequantize is the reference implementation for every type the
            # package knows — do not reimplement it per quant type.
            arr = gguf.quants.dequantize(tensor.data, qtype)
            quantized += 1
        t = torch.from_numpy(arr.copy())
        # GGUF stores shapes reversed relative to torch.
        shape = tuple(int(d) for d in reversed(tensor.shape))
        if shape and t.numel() == int(torch.tensor(shape).prod()):
            t = t.reshape(shape)
        sd[name] = t.to(torch.float16) if t.dtype == torch.float32 else t
    note = (f"gguf: dequantized at load ({quantized} quantized tensors expanded) — the file "
            f"loads, but it occupies its full size in VRAM. Install ComfyUI-GGUF to keep it "
            f"quantized.")
    return sd, {}, note
