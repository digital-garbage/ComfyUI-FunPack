"""Temporal style: how motion and time feel, on an LTX model. One mechanism, chosen by name.

Everything but `loop` tells the model a different frame rate than the clip's (see style.py);
`loop` rolls the latent in time so the clip's end joins its start (loop.py).

Not here: v4's `auto`, which read each scene's prompt and chose per scene. That choice is a
conditioning-side decision (the prompt lives there); it is still to be ported with the scene-chain
work, and a style set here applies to every scene of the run alike.
"""

from ..._core import patching
from . import loop as _loop
from . import style

ID = "temporal_style"
TITLE = "Temporal style"
MOUNT = "generation.sampling"
STAGE = "sampling"
CATEGORY = "sampling"
STATUS = "experimental"
REQUIRES = ["ltx_av"]

STYLES = ["natural", "accelerate", "decelerate", "freeze", "pulse",
          "rapid_start", "rapid_end", "rapid_start_end", "loop"]

SETTINGS = {
    "style": {
        "type": "enum", "default": "natural", "label": "Temporal style",
        "options": [{"value": v, "label": v.replace("_", " ")} for v in STYLES],
        "hint": "How motion and time feel. accelerate/freeze hold the motion, decelerate/rapid make it punchier, "
                "pulse repeats ease-downs, loop joins the end to the start.",
    },
}


def install(patcher, values, key):
    chosen = values.get("style", "natural")
    old = patcher.model_options.get("model_function_wrapper")
    state = {}
    if chosen in style.MULT:
        mult = style.MULT[chosen]
        wrapper = style.make_wrapper(old, lambda args: mult)
        note = f"{chosen}: frame rate x{mult:g}"
    elif chosen == "pulse":
        wrapper = style.make_wrapper(old, lambda args: style.pulse_mult(style.progress_of(args, state)))
        note = "pulse"
    elif chosen in ("rapid_start", "rapid_end", "rapid_start_end"):
        wrapper = style.make_wrapper(old, lambda args: style.rapid_mult(style.progress_of(args, state), chosen))
        note = chosen.replace("_", " ")
    elif chosen == "loop":
        wrapper = _loop.make_loop_temporal_wrapper(old)
        note = "loop: the clip's end joins its start"
        if "context_handler" in patcher.model_options:
            note += " (context windows are on: each WINDOW is looped, not the clip -- turn them off for a seamless clip)"
    else:
        return None
    patcher.set_model_unet_function_wrapper(patching.tag(wrapper, key))
    installed = patcher.model_options["model_function_wrapper"]
    installed._funpack_prev = old                       # so stripping ours puts the earlier one back
    return note


PROVIDES = {"modifier": install}
