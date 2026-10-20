"""Sharpness: win back fine detail late in the run, with any sampler.

Each step's prediction of the finished picture is pushed away from the previous step's, which keeps the
detail the smoothing steps wear off. It costs no extra model call (one picture-sized subtraction per
step, only in the chosen last part of the run). It is a different thing from Detail, which filters the
picture in space; this one compares steps in time.

The edit lands on the model's own answer, the last step included, as in 4.x. Samplers that call the model
twice per step compare each call with the one before it. Only the picture is touched, never the sound.
"""

from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, streams

ID = "sharpness"
TITLE = "Sharpness"
MOUNT = "generation.sampling"
STAGE = "guidance"
CATEGORY = "sampling"
STATUS = "experimental"
REQUIRES = ["temporal_latent"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Restore fine detail",
        "hint": "Wins back detail late in the run. No extra model calls.",
    },
    "amount": {
        "type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
        "label": "How much", "ui": "slider",
        "hint": "0 = nothing. 0.2 to 0.4 is typical.",
        "when": {"enabled": True},
    },
    "last_part": {
        "type": "float", "default": 0.35, "min": 0.05, "max": 1.0, "step": 0.05,
        "label": "Last part of the run", "ui": "slider",
        "hint": "The share of the steps, counted from the end, that are sharpened.",
        "when": {"enabled": True},
    },
}


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    amount, last = float(values.get("amount", 0.3)), float(values.get("last_part", 0.35))
    if amount <= 0.0:
        return "amount 0: nothing to do"
    seen = {}                                        # by cond/uncond call: (step, picture)

    def forget(executor, *args, **kwargs):
        seen.clear()
        try:
            return executor(*args, **kwargs)
        finally:
            seen.clear()                             # a latent-sized tensor must not outlive the run

    patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, f"{key}.forget", forget)

    def apply_model(executor, x, t, *args, **kwargs):
        named = streams.model_args(args, kwargs)
        to = named.get("transformer_options")
        out = executor(x, t, *args, **kwargs)
        where, split = dit_hooks.current_step(to), streams.video_of(out, named)
        if where is None or split is None or dit_hooks.probing(to):
            return out
        video, rebuild = split
        which = tuple(int(v) for v in ((to or {}).get("cond_or_uncond") or ()))
        before = seen.get(which)
        edited = (before is not None and where[0] >= before[0] and before[1].shape == video.shape    # else nothing to compare, or a new pass
                  and where[0] >= where[1] * (1.0 - last))
        if edited:
            video = video + 0.5 * amount * (video - before[1].to(video.dtype))
        seen[which] = (where[0], video.detach())             # the sharpened answer, so the boost does not compound
        return rebuild(video) if edited else out

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)
    return f"amount {amount:g} on the last {last:.0%} of the steps"


PROVIDES = {"modifier": install}
