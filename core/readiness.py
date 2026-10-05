"""Is this machine ready to generate? Plain checks of what a first run on a fresh GPU box trips over.

Each check is one row: {"level": "ok" | "warn" | "fail", "text": a sentence that says what to do}.
`fail` = a generation or render will not work; `warn` = it works but not as it should. Nothing here raises:
a check that cannot run reports itself as a warning, because this is the screen to trust when things are off.
"""

import importlib.util
import shutil
from typing import Callable, Iterable, List

LOW_DISK_GB = 20
MODEL_FOLDERS = (("diffusion_models", "diffusion models"), ("checkpoints", "checkpoints"), ("vae", "VAEs"),
                 ("text_encoders", "text encoders"), ("loras", "LoRAs"))


def row(level, text):
    return {"level": level, "text": text}


def have(module) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def device() -> str:
    """"cuda", "mps" or "cpu": what ComfyUI generates on (it can be started with --cpu), else what torch offers."""
    try:
        import comfy.model_management as mm
        return mm.get_torch_device().type
    except Exception:                                         # noqa: BLE001 -- not inside ComfyUI
        import torch
        mps = getattr(torch.backends, "mps", None)
        return "cuda" if torch.cuda.is_available() else "mps" if mps is not None and mps.is_available() else "cpu"


def machine(disk_free_gb=None) -> List[dict]:
    out = []
    for tool, why in (("ffmpeg", "the final render and previews"), ("ffprobe", "reading clip lengths")):
        out.append(row("ok", f"{tool} found") if shutil.which(tool) else row("fail", f"{tool} is not installed: it is needed for {why}. Install ffmpeg on this machine."))
    out.append(row("ok", "Pillow found") if have("PIL") else row("fail", "Pillow is not installed: 'continue from the previous clip' cannot read the last frame. pip install pillow"))
    try:
        import torch
        kind = device()
        if kind == "mps":
            out.append(row("ok", "Apple GPU (MPS): generates, slowly; large video models may not fit in its memory."))
        elif kind == "cpu":
            out.append(row("warn", "ComfyUI runs on the CPU: generating is very slow, and only small models fit."))
        elif kind != "cuda":
            out.append(row("warn", f"ComfyUI generates on a '{kind}' device, which FunPack is not tested on: expect slow runs, and some features may fail."))
        else:
            props = torch.cuda.get_device_properties(0)
            out.append(row("ok", f"{props.name}, {props.total_memory / 1024 ** 3:.0f} GB, sm_{props.major}{props.minor}"))
            if not torch.cuda.is_bf16_supported():
                out.append(row("fail", "This GPU does not support bf16, which the H3 VAEs need."))
            if props.major >= 12 and have("xformers"):
                out.append(row("warn", "xformers has no masked-attention kernel for this GPU: start ComfyUI with --disable-xformers --use-sage-attention."))
            if not have("sageattention"):                     # a CUDA-only library: nothing to install on a Mac
                out.append(row("warn", "sageattention is not installed: attention runs slower (and the H3 SLA setting has no fast path)."))
    except Exception as exc:                                  # noqa: BLE001
        out.append(row("warn", f"torch could not be inspected ({exc})."))
    if disk_free_gb is not None and disk_free_gb < LOW_DISK_GB:
        out.append(row("warn", f"Only {disk_free_gb} GB free: renders and previews fill a disk quickly (want {LOW_DISK_GB}+)."))
    return out


def models(counts: dict) -> List[dict]:
    """`counts`: model folder -> number of files ComfyUI lists there."""
    out = []
    if not counts.get("diffusion_models") and not counts.get("checkpoints"):
        out.append(row("fail", "No model files: put a diffusion model (or checkpoint) in ComfyUI's models folder."))
    for key, label in MODEL_FOLDERS:
        n = counts.get(key)
        if n is None:
            continue
        out.append(row("ok" if n else "warn", f"{n} {label}" if n else f"No {label} found."))
    return out


def pipelines(presets: Iterable[dict], installed: Callable[[str], bool]) -> List[dict]:
    """Nodes a starting point needs that this ComfyUI does not have (a custom-node pack is missing)."""
    out = []
    for preset in presets:
        missing = sorted({s["node"] for s in preset.get("slots", []) if not installed(s["node"])})
        if missing:
            out.append(row("warn", f"“{preset.get('title') or preset.get('id')}” needs nodes that are not installed: {', '.join(missing)}. Install the pack that provides them (Settings ▸ Custom nodes)."))
    return out or [row("ok", "Every starting point's nodes are installed")]


def modules(failed, quarantined) -> List[dict]:
    out = [row("warn", f"{where} did not load: {why}") for where, why in failed]
    out += [row("warn", f"Module {mid} is switched off after a failure: {e.get('reason', '')}. Release it in Settings ▸ Modules.") for mid, e in quarantined.items()]
    return out or [row("ok", "Every module loaded")]
