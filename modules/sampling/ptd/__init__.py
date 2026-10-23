"""PTD few-step adapters for MiniMax-H3: 4 to 8 hops instead of a long schedule, with no CFG.

A PTD file is not a LoRA: it holds one set of adapters per hop, and the sampler must visit exactly the
times it was trained at. Switching this on loads the file, puts the adapters on the model and runs PTD's
own sampler over the file's hop times, whatever sampler and steps are chosen. Ordinary LoRAs stack with it.

Loads the EPFL fl2va releases and the taobao-content-ai Ref2VA releases, as bf16 or as the smaller int8
files, on the full or the pruned base. Ported from ComfyUI-PTD-Compact (Nynxz) over ComfyUI-PTD (EPFL VITA
Lab) and ComfyUI-MiniMaxH3-PTD (taobao-content-ai), all Apache-2.0; see the README Thank you section.
Not yet run on a real H3 here.
"""

import os
from pathlib import Path

import folder_paths
from comfy.patcher_extension import WrappersMP

from ..._core import control, log, patching

from . import adapters
from .controller import CONTROLLER_KEY, Controller, attach
from .sampler import sample_ptd

ID = "ptd"
TITLE = "PTD few-step"
MOUNT = "generation.sampling"
STAGE = "load"
CATEGORY = "sampling"
STATUS = "experimental"
REQUIRES = ["audio_stream", "adaln_modalities"]

if "ptd" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("ptd", str(Path(folder_paths.models_dir) / "ptd"))

KERNELS = {"exact": False, "fast": True}


def _files():
    try:
        return [f for f in folder_paths.get_filename_list("ptd") if f.endswith(".safetensors")]
    except Exception:          # noqa: BLE001 -- no folder yet is the same as no files
        return []


def _options():
    return [{"value": f, "label": f} for f in _files()] or [{"value": "", "label": "No file in models/ptd"}]


_FOUND = _files()

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Few-step PTD",
        "hint": "Renders in the adapter file's own 4 to 8 steps with no CFG. The sampler, steps and scheduler "
                "are replaced; PTD cannot start from a picture or a mask. Use the base the file was trained on, "
                "and switch off SLA, turbo LoRAs and other enhancements, which it was not trained with.",
    },
    "adapter": {
        "type": "enum", "default": _FOUND[0] if _FOUND else "",
        "options": _options(),
        "label": "PTD adapter file",
        "hint": "A PTD file from ComfyUI/models/ptd. The list is read again each time Settings opens.",
        "when": {"enabled": True},
    },
    "kernels": {
        "type": "enum", "default": "exact",
        "options": [{"value": "exact", "label": "Exact"}, {"value": "fast", "label": "Fast"}],
        "label": "Block maths",
        "hint": "Exact follows the arithmetic the adapters were trained with. Fast uses ComfyUI's own blocks, "
                "about 4 to 8 percent quicker per step; only the rounding differs.",
        "when": {"enabled": True},
    },
}

# One parsed file is kept between runs, so changing a seed or a prompt does not read gigabytes again.
# It holds no model, so nothing here keeps a diffusion model alive.
_kept = {}


def _adapter(path):
    stamp = (str(path), os.path.getmtime(path), os.path.getsize(path))
    if _kept.get("stamp") != stamp:
        _kept.clear()
        _kept.update(stamp=stamp, adapter=adapters.parse(path, label=Path(path).name))
    return _kept["adapter"]


def _swap(controller):
    """Run PTD's sampler over the adapter's own hop times, in place of the sampler that was chosen."""
    import comfy.samplers
    ptd = comfy.samplers.KSAMPLER(sample_ptd, extra_options={"controller": controller})

    def run(executor, guider, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar):
        if abs(float(getattr(guider, "cfg", 1.0)) - 1.0) > 1e-6:
            log.once("ptd:cfg", log.ALERT, "FunPack PTD",
                     f"CFG is {guider.cfg:g}: PTD was trained without guidance, so set CFG to 1")
        try:
            import latent_preview
            callback = latent_preview.prepare_callback(guider.model_patcher, controller.hops)
        except Exception:      # noqa: BLE001 -- only the progress bar and previews are lost
            pass
        # Step-aware modules (SLA, momentum, velocity bias) read the schedule from here: give them the real one.
        # On the device and dtype the sampler was given: ComfyUI's own schedule sits on the GPU, and a model that
        # compares it with its call's sigma (H3's head bank, SLA) fails on a CPU copy.
        grid = controller.grid.to(device=sigmas.device, dtype=sigmas.dtype)
        options = extra_args.get("model_options", {}).get("transformer_options")
        if options is not None:
            options["sample_sigmas"] = grid
        log.once("ptd:swap", log.INFO, "FunPack PTD",
                 f"sampling {controller.hops} hops from {controller.adapter.label}; the chosen sampler and steps "
                 f"({len(sigmas) - 1}) are not used")
        return ptd.sample(guider, grid, extra_args, callback, noise, latent_image,
                          denoise_mask, disable_pbar)

    return run


def install(patcher, values, key):
    if not values["enabled"]:
        _kept.clear()
        return None
    raw = getattr(patcher, "_patcher", patcher)
    # PTD installs on the real patcher, not the guarded one. A guard would drop a failing hook and carry on
    # with the plain sampler on a model whose layers expect PTD's hops, which renders noise; here it ends the run.
    path = folder_paths.get_full_path("ptd", values["adapter"]) if values["adapter"] else None
    if path is None:
        raise control.Unavailable("no PTD adapter file is chosen: put one in ComfyUI/models/ptd, restart, "
                                  "and pick it in Settings")
    native = raw.get_model_object("diffusion_model")
    adapter = _adapter(path)
    adapters.verify(adapter, native)
    if CONTROLLER_KEY in raw.model_options:
        raise ValueError("This model already carries PTD adapters")
    controller = Controller(native, adapter)
    controller.fuse_blocks = KERNELS[values["kernels"]]
    attach(raw, controller, key, patching.tag)
    raw.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, key, patching.tag(_swap(controller), key))
    size = adapter.size / 2**30
    return (f"{adapter.label}, {adapter.hops} hops, {'int8' if adapter.int8 else 'float'} {size:.1f} GB, "
            f"{values['kernels']} blocks")


def patcher_has_ptd(patcher):
    from .controller import CONTROLLER_KEY
    return CONTROLLER_KEY in patcher.model_options

PROVIDES = {"modifier": install, "setting_options": lambda: {"adapter": _options()}}
