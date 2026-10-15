"""Load .gguf model files through FunPack's own loaders.

Not a module: see common.py's own note -- a plain file at the domain level is a
helper the loaders import, and ships no node of its own.

GGUF is two problems, and only the first is small:

1. **Reading the container.** A header, tensor metadata, then quantized blocks.
   The `gguf` package (llama.cpp's own) does this, and ships a reference
   dequantizer for every quant type it defines.

2. **Using the weights without dequantizing them.** The reason people reach for
   GGUF is that a Q4 model occupies a quarter of the VRAM -- which only holds
   if the weights STAY quantized and are expanded per-layer during the forward
   pass. That needs a custom torch ops class, and hand-written kernels per
   quant type.

So this module does not hand-roll (2). It picks the best backend actually
present:

  * **ComfyUI-GGUF installed** -- use it. It is the mature implementation of
    (2), and its output feeds `comfy.sd.load_diffusion_model_state_dict`
    exactly like ours does, so FunPack's loader stays the only UI while its
    runtime does the quantized maths.
  * **`gguf` package only** -- read and dequantize at LOAD time, then hand over
    an ordinary state dict. Fully native and correct, but the model occupies
    its dequantized size, so the VRAM saving is gone. Said out loud rather than
    left to be discovered.
  * **Neither** -- refuse with the two ways to fix it, instead of a stack trace.

Nothing here downloads or installs anything.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from typing import Optional

GGUF_EXT = ".gguf"


def is_gguf(name: str) -> bool:
    return str(name or "").lower().endswith(GGUF_EXT)


GGUF_MAGIC = b"GGUF"


def has_gguf_magic(path: str) -> bool:
    """True when the FILE is a GGUF container, whatever it is called.

    Renaming a .gguf to .safetensors is an easy mistake to make, and the
    result is an unreadable error from the safetensors parser trying to read a
    binary header as UTF-8 JSON. Four bytes settle it, so the content decides.
    """
    try:
        with open(path, "rb") as f:
            return f.read(4) == GGUF_MAGIC
    except OSError:
        return False


# ── finding .gguf files ───────────────────────────────────────────────────────

def _walk(root: str, limit: int = 5000, depth: int = 12):
    """Every .gguf under `root` as a relative name, following links.

    A folder is not entered again while it is one of the folders above it (same device and inode, so case, roots and
    cycles through a sibling are all caught), and the walk stops after `limit` folders: a link to a whole disk is slow, not endless.
    A link to a sibling folder is not a loop; it is a second name for the same files.
    """
    seen = [0]
    cut = []

    def rec(path, rel, above):
        seen[0] += 1
        try:
            entries = sorted(os.scandir(path), key=lambda e: e.name)
        except OSError:
            return
        for e in entries:
            try:
                if e.is_dir():
                    st = os.stat(e.path)
                    key = (st.st_dev, st.st_ino)
                    if key in above:
                        continue
                    if len(above) >= depth or seen[0] >= limit:
                        cut.append(e.path)
                        continue
                    yield from rec(e.path, rel + e.name + "/", above | {key})
                elif is_gguf(e.name) and e.is_file():              # is_file: a dangling link is not a model
                    yield rel + e.name
            except OSError:
                continue

    st = os.stat(root)
    yield from rec(root, "", frozenset({(st.st_dev, st.st_ino)}))
    if cut:
        from .._core import log
        log.once(f"gguf_walk_cut:{root}", log.WARNING, "FunPack .gguf listing",
                 f"{root}: stopped looking inside {len(cut)} folder(s) (deeper than {depth} levels or more than {limit} folders in all), so .gguf files there are not listed")


def gguf_names(folder: str) -> list[str]:
    """`.gguf` files under a ComfyUI model folder, relative-named like get_filename_list().

    Core's `supported_pt_extensions` has no `.gguf`, so `get_filename_list`
    never returns one -- the files are on disk and simply invisible to every
    picker. Scanned directly rather than by mutating core's global extension
    set, which would change what every other node in the install offers.
    """
    try:
        import folder_paths
    except ImportError:
        return []
    out: list[str] = []
    seen: set[str] = set()
    roots = list(folder_paths.get_folder_paths(folder) or [])
    # ComfyUI-GGUF registers its own folders; include them under the same
    # picker so a user who already keeps files there does not have to move them.
    for extra in ("unet_gguf" if folder == "diffusion_models" else "clip_gguf",):
        try:
            roots += list(folder_paths.get_folder_paths(extra) or [])
        except Exception:
            pass
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for rel in _walk(root):
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
    if ".." in name.replace("\\", "/").split("/"):
        return None
    for root in roots:
        cand = os.path.join(root or "", name)
        if not os.path.normpath(cand).startswith(os.path.normpath(root or ".") + os.sep):
            continue                                        # a name that climbs out of the model folder (symlinks inside it are fine)
        if os.path.isfile(cand):
            return cand
    return None


# ── backends ──────────────────────────────────────────────────────────────────

_PKG_NAME = "_funpack_gguf_pack"


def _ensure_pack_package(dirname: str) -> str:
    """Register the pack's directory as an importable package, once.

    Loading `loader.py` as a standalone module by file path fails the moment
    it does `from .ops import ...` -- "attempted relative import with no known
    parent package", because a module loaded that way has no package to be
    relative TO. So a parent package is synthesized with `__path__` pointing at
    the directory, and submodules are imported underneath it, which is what
    makes relative imports resolve.

    The pack's own `__init__.py` is deliberately NOT executed: it registers
    the pack's nodes, and we want its loader, not a second copy of its node
    list.
    """
    pkg = sys.modules.get(_PKG_NAME)
    if pkg is not None and getattr(pkg, "__path__", [None])[0] == dirname:
        return _PKG_NAME
    # Pointing at a different directory: drop the cached SUBMODULES too.
    # Replacing only the parent leaves `<pkg>.loader` in sys.modules, and
    # import_module would hand back the old pack's code from the new path's name.
    for name in [m for m in sys.modules if m == _PKG_NAME or m.startswith(_PKG_NAME + ".")]:
        del sys.modules[name]
    pkg = types.ModuleType(_PKG_NAME)
    pkg.__path__ = [dirname]
    pkg.__package__ = _PKG_NAME
    sys.modules[_PKG_NAME] = pkg
    return _PKG_NAME


def _load_pack_module(dirname: str, mod: str):
    """Import one module out of a custom-node package by path.

    Custom node folders are not importable by name (``ComfyUI-GGUF`` is not an
    identifier), so the directory is mounted as a package first and the
    submodule imported from it.
    """
    pkg = _ensure_pack_package(dirname)
    return importlib.import_module(f"{pkg}.{mod}")


def _pack_dir() -> Optional[str]:
    """ComfyUI-GGUF's directory, identified by what it CONTAINS rather than by
    its name -- the folder gets renamed by Manager, by forks, and by hand."""
    try:
        import folder_paths
        base = os.path.join(folder_paths.base_path, "custom_nodes")
    except Exception:
        return None
    if not os.path.isdir(base):
        return None
    for entry in sorted(os.listdir(base)):
        if entry.endswith(".disabled"):                             # Manager's way of switching a pack off
            continue
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
    "No GGUF backend is available. The `gguf` package is one of FunPack's requirements, "
    "so an install that predates .gguf support just needs them re-run:\n"
    "    pip install -r custom_nodes/ComfyUI-FunPack/requirements.txt\n"
    "That gives a load-time dequantize: the file loads, but at its full size. To keep the "
    "weights quantized in VRAM -- which is the point of GGUF -- install the ComfyUI-GGUF "
    "node pack as well; FunPack uses it as the runtime and stays the only loader you wire."
)


PACK_NEEDS_GGUF = (
    "ComfyUI-GGUF is installed, but the `gguf` package it needs is not. Install it in ComfyUI's Python "
    "(pip install -r custom_nodes/ComfyUI-FunPack/requirements.txt) and restart ComfyUI."
)


def load_state_dict(path: str) -> tuple[dict, dict, str]:
    """Read one .gguf into (state_dict, model_options, note).

    `model_options` carries the custom operations the quantized path needs; it
    is empty for the native path, whose tensors are ordinary ones by the time
    they are returned.
    """
    return _load(path, clip=False)


def load_clip_state_dict(path: str) -> tuple[dict, dict, str]:
    """Same as `load_state_dict`, for a text encoder.

    ComfyUI-GGUF reads encoders through a separate entry point: an encoder's
    GGUF carries llama.cpp tensor names that have to be mapped back to the
    transformer's, which its diffusion-model reader does not do.
    """
    return _load(path, clip=True)


def _load(path: str, clip: bool) -> tuple[dict, dict, str]:
    """Try the quantized backend, fall back to dequantizing, in that order.

    The pack refuses architectures it has no handling for -- "Unexpected
    architecture type in GGUF file: 'minimax_h3'" -- which is a reasonable
    refusal on its part and a dead end for us. Since the pack DEPENDS on the
    `gguf` package, its presence guarantees the fallback is available, so a
    refusal costs memory rather than the load.
    """
    which = backend()
    if clip and which != "pack":
        # Only the pack maps llama.cpp's tensor names back to a text encoder's; the native reader would hand over
        # a state dict that looks loaded and is the wrong model.
        raise RuntimeError("a text-encoder .gguf needs the ComfyUI-GGUF node pack to be read, and it is not installed. "
                           "Installing it helps only for architectures it handles (not Gemma 4, not vision files); "
                           "otherwise use a .safetensors encoder of the same model.")
    if which is None:
        raise RuntimeError(UNAVAILABLE)
    if which == "pack":
        d = _pack_dir()
        try:
            loader = _load_pack_module(d, "loader")
            ops = _load_pack_module(d, "ops")
            read = (getattr(loader, "gguf_clip_loader", None) if clip else None) or loader.gguf_sd_loader
            sd = read(path)
            if isinstance(sd, tuple):        # newer ComfyUI-GGUF: (state dict, {arch_str, metadata})
                sd = sd[0]
            if clip:
                # A vision projector (mmproj) comes back from the pack with llama.cpp's own v./mm. names and nothing
                # else; built as a text encoder it would be an empty shell that "loaded". Vision tensors beside mapped
                # text ones are harmless (ComfyUI loads encoders non-strictly), so only "no text keys" or unmapped
                # text names (blk./token_embd) are refused.
                text = [str(k) for k in sd if not str(k).startswith(("v.", "mm."))]
                if not text or any(k.startswith(("blk.", "token_embd")) for k in text):
                    raise ValueError("it holds no text-encoder weights ComfyUI-GGUF can map (a vision projector / mmproj file?)")
            options = {}
            ggml_ops = getattr(ops, "GGMLOps", None)
            if ggml_ops is not None:
                options["custom_operations"] = ggml_ops()
            return sd, options, f"gguf: quantized (ComfyUI-GGUF at {os.path.basename(d)})"
        except Exception as e:  # noqa: BLE001 -- any refusal is a reason to fall back
            no_gguf = importlib.util.find_spec("gguf") is None
            if no_gguf and isinstance(e, ModuleNotFoundError) and "gguf" in str(e):
                raise RuntimeError(PACK_NEEDS_GGUF) from e
            if clip:
                # The native reader keeps llama.cpp's tensor names; only the pack maps them back to a text encoder's.
                # Loading them anyway gives a model that looks loaded and is the wrong one (a Gemma 4 .gguf did).
                raise RuntimeError(f"this text-encoder .gguf could not be loaded as one: {e}. "
                                   "If the file is intact, its architecture is not one ComfyUI-GGUF handles (Gemma 4, vision files) "
                                   "or ComfyUI-GGUF is out of date. Use a .safetensors encoder of the same model.") from e
            if no_gguf:
                raise
            sd, options, note = _load_native(path, pack=True)
            return sd, options, (f"gguf: ComfyUI-GGUF could not read this file ({e}) -- "
                                 f"fell back. {note}")
    return _load_native(path)


def _load_native(path: str, pack: bool = False) -> tuple[dict, dict, str]:
    """Dequantize every tensor to torch at load time using gguf's reference implementation.

    Correct but not thrifty: the model ends up at its dequantized size, so
    this buys the ability to LOAD a .gguf, not GGUF's memory saving. The
    caller states that.
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
            # gguf.quants.dequantize is the reference implementation for every
            # type the package knows -- do not reimplement it per quant type.
            try:
                arr = gguf.quants.dequantize(tensor.data, qtype)
            except (NotImplementedError, ValueError) as e:
                raise RuntimeError(f"{os.path.basename(path)}: tensor {name!r} uses a storage type the gguf package cannot "
                                   f"dequantize or is damaged ({e}).") from e
            quantized += 1
        t = torch.from_numpy(arr.copy())
        # GGUF stores dimensions in the opposite order to torch.
        shape = tuple(int(d) for d in reversed(tensor.shape))
        expected = 1
        for d in shape:
            expected *= d
        if shape and t.numel() != expected:
            # Loading the tensor unshaped would hand the model a weight of the
            # wrong rank and fail somewhere far from here, so stop with the
            # name of the culprit.
            raise RuntimeError(
                f"{os.path.basename(path)}: tensor {name!r} dequantized to {t.numel()} "
                f"elements but its header declares {shape} ({expected}). This container "
                f"is not laid out the way the gguf package describes it.")
        if shape:
            t = t.reshape(shape)
        if t.dtype == torch.float32 and t.numel():       # fp16 halves the size; a tensor too big for its range goes bf16 instead of inf
            t = t.to(torch.float16) if t.abs().max() < 60000 else t.to(torch.bfloat16)
        sd[name] = t
    note = (f"gguf: dequantized at load ({quantized} quantized tensors expanded) -- the "
            f"file loads, but it occupies its full size in VRAM."
            + ("" if pack else " Install ComfyUI-GGUF to keep it quantized."))
    return sd, {}, note
