"""Shadow negative: give the negative prompt a job on a model that ignores it.

MiniMax H3 samples at CFG 1, so ComfyUI never evaluates the negative branch and
the negative prompt is computed and thrown away. This runs the negative text
through the SAME blocks as a second, "shadow" stream and pushes the real
stream's attention output away from it, NAG-style: normalised and norm-capped,
so it cannot run away.

Heavier than a one-off embedding edit: it re-derives the push at every block,
every step. Roughly one extra attention pass per block while it is active.

v4 shipped this and it never ran (see shadow.py for why). Anything heard or seen
from v4's version was something else; this one needs judging fresh.

A block this replaces is not run through any other module's hook installed
UNDERNEATH it on that block -- the forward is this module's own. "Leave other
block features alone" skips those blocks instead, so both keep working, each on
its own blocks.
"""

from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, log
from . import shadow

ID = "shadow_negative"
TITLE = "Shadow negative"
MOUNT = "generation.sampling"
STAGE = "guidance"
CATEGORY = "guidance"
STATUS = "experimental"
REQUIRES = ["dit_block_hooks", "adaln_modalities", "audio_stream"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Use the negative prompt",
        "hint": "This model ignores negative prompts; this makes it steer away from yours.",
    },
    "video_scale": {
        "type": "float", "default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1,
        "label": "Picture push", "ui": "slider",
        "hint": "How hard the picture moves away from the negative. 1 = off.",
        "when": {"enabled": True},
    },
    "audio_scale": {
        "type": "float", "default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1,
        "label": "Sound push", "ui": "slider",
        "hint": "Same for the sound. 1 = off.",
        "when": {"enabled": True},
    },
    "tau": {
        "type": "float", "default": 2.5, "min": 1.0, "max": 8.0, "step": 0.05,
        "label": "Limit", "ui": "slider",
        "hint": "Caps how far any one spot can be pushed. Lower is safer.",
        "when": {"enabled": True},
    },
    "alpha": {
        "type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
        "label": "Mix", "ui": "slider",
        "hint": "How much of the pushed result is kept.",
        "when": {"enabled": True},
    },
    "start": {
        "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
        "label": "From", "ui": "slider",
        "hint": "Where in the steps it starts, 0 = first step.",
        "when": {"enabled": True},
    },
    "end": {
        "type": "float", "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01,
        "label": "Until", "ui": "slider",
        "hint": "Where it stops, 1 = last step.",
        "when": {"enabled": True},
    },
    "compose": {
        "type": "bool", "default": False,
        "label": "Leave other block features alone",
        "hint": "Skips blocks another feature is already using, so both keep working.",
        "when": {"enabled": True},
    },
}


def _say(message):
    log.once(f"{ID}:{message}", log.ALERT, "FunPack Shadow negative", message)


def _negative_of(guider):
    """The first negative entry's text states, from the guider about to sample.

    CFGGuider.outer_sample is wrapped with the guider as `class_obj`, and its
    `conds` are already the per-run copies by then (comfy/samplers.py).
    """
    for entry in (getattr(guider, "conds", None) or {}).get("negative") or ():
        cross = entry.get("cross_attn") if isinstance(entry, dict) else None
        if cross is not None and getattr(cross, "numel", lambda: 0)():
            return cross
    return None


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    video_scale = float(values.get("video_scale", 3.0))
    audio_scale = float(values.get("audio_scale", 1.0))
    alpha = float(values.get("alpha", 0.35))
    if (video_scale == 1.0 and audio_scale == 1.0) or alpha == 0.0:
        return None
    start, end = sorted((float(values.get("start", 0.0)), float(values.get("end", 0.6))))
    compose = bool(values.get("compose"))

    dm = patcher.model.diffusion_model
    state = shadow.State(dm, video_scale, audio_scale, float(values.get("tau", 2.5)),
                         alpha, start, end)

    def outer(executor, *args, **kwargs):
        state.negative = _negative_of(getattr(executor, "class_obj", None))
        if state.negative is None:
            _say("off this run: there is no negative prompt to push away from")
        return executor(*args, **kwargs)

    def per_call(executor, x, timestep, context, *args, **kwargs):
        state.text_len = int(context.shape[1]) if context is not None else None
        state.neg_h = None
        return executor(x, timestep, context, *args, **kwargs)

    patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, outer)
    patcher.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, key, per_call)

    def in_window(args):
        where = dit_hooks.current_step(args.get("transformer_options"))
        if where is None:
            return True
        index, total = where
        progress = index / max(1, total - 1)
        return start <= progress <= end

    def make(block):
        def hook(args, extra):
            if state.negative is None or (compose and extra["funpack_below"]) \
                    or not in_window(args):
                return extra["original_block"](args)
            out, why = shadow.forward(state, block, args)
            if out is None:
                _say(f"skipped a block: {why}")
                return extra["original_block"](args)
            return out
        return hook

    dit = patcher.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
    taken = sorted(i for (kind, i, *_rest) in dit if kind == dit_hooks.BLOCK)
    for i, block in enumerate(dm.blocks):
        dit_hooks.add_block_hook(patcher, key, i, make(block))

    # Replaces what the hooked blocks allocate: see dit_hooks.without_compiler.
    dit_hooks.without_compiler(patcher, key)
    note = f"picture {video_scale:g}, sound {audio_scale:g}, steps {start:.2f}-{end:.2f}"
    if taken:
        note += (f"; leaves block(s) {','.join(map(str, taken))} to the feature already there"
                 if compose else
                 f"; takes over block(s) {','.join(map(str, taken))} from the feature already there")
    return note


PROVIDES = {"modifier": install}
