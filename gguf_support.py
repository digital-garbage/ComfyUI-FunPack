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

import importlib
import importlib.util
import itertools
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

    Renaming a .gguf to .safetensors is an easy mistake to make — for most of this project's
    life the pickers only offered .safetensors, so renaming was the obvious way to get a file
    to show up. The result is an unreadable error from the safetensors parser trying to read
    a binary header as UTF-8 JSON. Four bytes settle it, so the content decides.
    """
    try:
        with open(path, "rb") as f:
            return f.read(4) == GGUF_MAGIC
    except OSError:
        return False


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

_PKG_NAME = "_funpack_gguf_pack"


def _ensure_pack_package(dirname: str) -> str:
    """Register the pack's directory as an importable package, once.

    Loading `loader.py` as a standalone module by file path fails the moment it does
    `from .ops import ...` — "attempted relative import with no known parent package",
    because a module loaded that way has no package to be relative TO. So a parent package
    is synthesized with `__path__` pointing at the directory, and submodules are imported
    underneath it, which is what makes relative imports resolve.

    The pack's own `__init__.py` is deliberately NOT executed: it registers the pack's nodes,
    and we want its loader, not a second copy of its node list.
    """
    pkg = sys.modules.get(_PKG_NAME)
    if pkg is not None and getattr(pkg, "__path__", [None])[0] == dirname:
        return _PKG_NAME
    # Pointing at a different directory: drop the cached SUBMODULES too. Replacing only the
    # parent leaves `<pkg>.loader` in sys.modules, and import_module would hand back the old
    # pack's code from the new path's name.
    for name in [m for m in sys.modules if m == _PKG_NAME or m.startswith(_PKG_NAME + ".")]:
        del sys.modules[name]
    pkg = types.ModuleType(_PKG_NAME)
    pkg.__path__ = [dirname]
    pkg.__package__ = _PKG_NAME
    sys.modules[_PKG_NAME] = pkg
    return _PKG_NAME


def _load_pack_module(dirname: str, mod: str):
    """Import one module out of a custom-node package by path.

    Custom node folders are not importable by name (``ComfyUI-GGUF`` is not an identifier),
    so the directory is mounted as a package first and the submodule imported from it.
    """
    pkg = _ensure_pack_package(dirname)
    return importlib.import_module(f"{pkg}.{mod}")


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
    "No GGUF backend is available. The `gguf` package is one of FunPack's requirements, so an "
    "install that predates .gguf support just needs them re-run:\n"
    "    pip install -r custom_nodes/ComfyUI-FunPack/requirements.txt\n"
    "That gives a load-time dequantize: the file loads, but at its full size. To keep the "
    "weights quantized in VRAM — which is the point of GGUF — install the ComfyUI-GGUF node "
    "pack as well; FunPack uses it as the runtime and stays the only loader you wire."
)


def load_state_dict(path: str) -> tuple[dict, dict, str]:
    """Read one .gguf into (state_dict, model_options, note).

    `model_options` carries the custom operations the quantized path needs; it is empty for
    the native path, whose tensors are ordinary ones by the time they are returned.
    """
    return _load(path, clip=False)


def load_clip_state_dict(path: str) -> tuple[dict, dict, str]:
    """Same as `load_state_dict`, for a text encoder.

    ComfyUI-GGUF reads encoders through a separate entry point: an encoder's GGUF carries
    llama.cpp tensor names that have to be mapped back to the transformer's, which its
    diffusion-model reader does not do.
    """
    return _load(path, clip=True)


def _load(path: str, clip: bool) -> tuple[dict, dict, str]:
    """Try the quantized backend, fall back to dequantizing, in that order.

    The pack refuses architectures it has no handling for — "Unexpected architecture type in
    GGUF file: 'minimax_h3'" — which is a reasonable refusal on its part and a dead end for
    us. Since the pack DEPENDS on the `gguf` package, its presence guarantees the fallback
    is available, so a refusal costs memory rather than the load.
    """
    which = backend()
    if which is None:
        raise RuntimeError(UNAVAILABLE)
    leftover = note_leftover_cache(path)
    if which == "pack":
        d = _pack_dir()
        try:
            loader = _load_pack_module(d, "loader")
            ops = _load_pack_module(d, "ops")
            allowed = _allow_architectures(loader, path)
            read = loader.gguf_sd_loader
            if clip:
                read = getattr(loader, "gguf_clip_loader", None) or read
            sd = read(path)
            # Some versions hand back (state_dict, architecture) rather than the dict alone.
            # Passed on as-is, ComfyUI iterates the TUPLE, gets the dict as its first element
            # and fails with "'dict' object has no attribute 'startswith'" — an error with
            # nothing in it that points back here.
            if isinstance(sd, (tuple, list)) and len(sd) == 2 and isinstance(sd[0], dict):
                sd = sd[0]
            # Checked on every path, not just when the architecture list was overridden: the
            # shape of what the pack returns is a moving target across its versions, and
            # anything unexpected fails deep inside core where the message makes no sense out
            # here. Raising turns it into the fallback below instead.
            _assert_state_dict(sd, path)
            options = {}
            ggml_ops = getattr(ops, "GGMLOps", None)
            if ggml_ops is not None:
                options["custom_operations"] = ggml_ops()
            return sd, options, (f"gguf: quantized (ComfyUI-GGUF at {os.path.basename(d)})"
                                 + (f"; {allowed}" if allowed else "")
                                 + (f"; {leftover}" if leftover else ""))
        except Exception as e:  # noqa: BLE001 — any refusal is a reason to fall back
            if importlib.util.find_spec("gguf") is None:
                raise
            sd, options, note = _load_native(path)
            return sd, options, (f"gguf: ComfyUI-GGUF could not read this file ({e}) — "
                                 f"fell back. {note}"
                                 + (f" {leftover}" if leftover else ""))
    sd, options, note = _load_native(path)
    return sd, options, note + (f" {leftover}" if leftover else "")


LEFTOVER_SUFFIX = ".dequantized.safetensors"


def note_leftover_cache(path: str) -> str:
    """FunPack used to keep an expanded copy of a .gguf beside it. It no longer does — on a
    rented box disk is billed, and the copy is the size of the whole model. Any file left
    over from that is dead weight the user has no reason to guess at, so name it and its
    size once. Never deleted here: it is a big file in the user's model folder."""
    leftover = path + LEFTOVER_SUFFIX
    try:
        if not os.path.isfile(leftover):
            return ""
        gb = os.path.getsize(leftover) / 1024 ** 3
    except OSError:
        return ""
    return (f"an old FunPack expansion cache is still beside this model "
            f"({os.path.basename(leftover)}, {gb:.1f} GB) — nothing reads it any more, "
            f"delete it to reclaim the space")


def _assert_state_dict(sd, path: str) -> None:
    """Raise unless `sd` is a usable state dict. Cheap, and only shape — nothing here can
    tell whether the NUMBERS are right, which is why the status line says to suspect this
    first if a generation comes out wrong."""
    if not isinstance(sd, dict) or not sd:
        raise RuntimeError(
            f"{os.path.basename(path)}: the GGUF backend returned "
            f"{type(sd).__name__} instead of a state dict")
    for key, value in sd.items():
        if not isinstance(key, str):
            raise RuntimeError(
                f"{os.path.basename(path)}: the GGUF backend returned a state dict keyed by "
                f"{type(key).__name__}, not by tensor name")
        if not hasattr(value, "shape"):
            raise RuntimeError(
                f"{os.path.basename(path)}: {key!r} came back as {type(value).__name__}, "
                f"which is not a tensor")


def read_architecture(path: str) -> Optional[str]:
    """`general.architecture` out of a GGUF header, or None. Header only — no tensor data."""
    try:
        import gguf
        reader = gguf.GGUFReader(path)
        field = reader.fields.get("general.architecture")
        if field is None:
            return None
        return str(bytes(field.parts[field.data[0]]), encoding="utf-8")
    except Exception:  # noqa: BLE001
        return None


def _allow_architectures(loader, path: str) -> str:
    """Let ComfyUI-GGUF read an architecture its allowlist does not name. Returns a note.

    The pack keeps a list of architectures it has been tested against and refuses anything
    else. That is the right default for the pack and a dead end here: the refusal costs the
    QUANTIZED path, and the fallback expands the whole checkpoint at load — minutes of
    dequantizing, and the file's full size in VRAM, which is the entire reason someone chose
    a GGUF.

    The reading itself is generic: the pack walks the tensors and wraps them, and the
    architecture only steers the key renaming that image models do not use. So the name is
    added to the IMAGE list, never the text one, and only for a file whose header actually
    declares it. Announced, because it is an override of another project's own guard and a
    wrong result here should be traceable to this line.
    """
    arch = read_architecture(path)
    if not arch:
        return ""
    added = []
    for attr in ("IMG_ARCH_LIST", "ARCH_LIST"):
        lst = getattr(loader, attr, None)
        if lst is None:
            continue
        try:
            if arch in lst:
                return ""              # the pack already knows it; nothing to override
            if isinstance(lst, set):
                lst.add(arch)
            elif isinstance(lst, list):
                lst.append(arch)
            else:
                continue
            added.append(attr)
        except Exception:  # noqa: BLE001
            continue
    if not added:
        return ""
    return (f"architecture {arch!r} is not on ComfyUI-GGUF's tested list and FunPack added "
            f"it, so the file loads quantized instead of being expanded — if the weights come "
            f"out wrong, this is the first thing to suspect")


def _load_native(path: str) -> tuple[dict, dict, str]:
    """Dequantize every tensor to torch at load time using gguf's reference implementation.

    Correct but not thrifty: the model ends up at its dequantized size, so this buys the
    ability to LOAD a .gguf, not GGUF's memory saving. The caller states that.

    Expansion is STREAMED. Dequantizing produces float32 — four times the size the weight
    will occupy once it is cast — so collecting every tensor before converting any of them
    peaks at several times the finished model in host RAM. On a video checkpoint that is
    tens of gigabytes of transient allocation, which is enough to take the machine down
    rather than merely the load. Each tensor is therefore cast and released as it is
    produced, and only a small window is ever in flight.
    """
    import gguf
    import numpy as np
    import torch

    reader = gguf.GGUFReader(path)
    tensors = list(reader.tensors)
    raw = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
    quantized = sum(1 for t in tensors if t.tensor_type not in raw)
    # Bounded, and deliberately small: the window below keeps `workers * 2` float32 tensors
    # alive at once, and the largest weight in a video model is around a gigabyte at that
    # width. More threads would expand faster and spike higher; the durable speed win is the
    # ComfyUI-GGUF pack's quantized path, not this pool.
    workers = max(1, min(4, (os.cpu_count() or 2) - 1))
    print(f"[FunPack] gguf: expanding {quantized} of {len(tensors)} tensors from "
          f"{os.path.basename(path)} across {workers} thread(s). This is the slow path — "
          f"minutes on a video checkpoint. Install ComfyUI-GGUF to skip it entirely.",
          flush=True)

    def _expand(tensor):
        """One tensor, fully finished: nothing float32 outlives this call."""
        qtype = tensor.tensor_type
        arr = tensor.data if qtype in raw else gguf.quants.dequantize(tensor.data, qtype)
        if arr.dtype == np.float32:
            # `.to` allocates the half-width copy and the float32 array is freed on return,
            # so the wide form never accumulates.
            t = torch.from_numpy(arr).to(torch.float16)
        else:
            # Detach from the memory-mapped file; the reader's mapping does not outlive us.
            t = torch.from_numpy(arr.copy())
        name = str(tensor.name)
        # GGUF stores dimensions in the opposite order to torch.
        shape = tuple(int(d) for d in reversed(tensor.shape))
        expected = 1
        for d in shape:
            expected *= d
        if shape and t.numel() != expected:
            # Loading the tensor unshaped would hand the model a weight of the wrong rank
            # and fail somewhere far from here, so stop with the name of the culprit.
            raise RuntimeError(
                f"{os.path.basename(path)}: tensor {name!r} dequantized to {t.numel()} "
                f"elements but its header declares {shape} ({expected}). This container is "
                f"not laid out the way the gguf package describes it.")
        return name, t.reshape(shape) if shape else t

    sd: dict = {}
    if workers > 1 and len(tensors) > 1:
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            remaining = iter(tensors)
            # A fixed window rather than `pool.map`, which submits every tensor at once and
            # buffers each result until the one before it is consumed — the whole model in
            # float32, which is the allocation this function exists to avoid.
            flight = deque(pool.submit(_expand, t)
                           for t in itertools.islice(remaining, workers * 2))
            while flight:
                name, t = flight.popleft().result()
                sd[name] = t
                nxt = next(remaining, None)
                if nxt is not None:
                    flight.append(pool.submit(_expand, nxt))
    else:
        for tensor in tensors:
            name, t = _expand(tensor)
            sd[name] = t

    note = (f"gguf: dequantized at load ({quantized} quantized tensors expanded across "
            f"{workers} thread(s)) — the file loads, but it occupies its full size in VRAM, "
            f"and expanding it is why this is slower than a .safetensors. Install "
            f"ComfyUI-GGUF for a lazy quantized load.")
    return sd, {}, note
