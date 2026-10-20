"""Sharpness: win back fine detail late in the run, with any sampler.

Each step's prediction of the finished picture is pushed away from the previous step's, which keeps the
detail the smoothing steps wear off. It costs no extra model call (one picture-sized subtraction per
step, only in the chosen last part of the run). It is a different thing from Detail, which filters the
picture in space; this one compares steps in time.

The edit rides into the next step's input (core/input_steer.py), so the last step is never edited on
packed models. Samplers that call the model twice per step (heun, dpm_2) are not supported: it says so
in the Log and stands down. Only the picture is touched, never the sound.
"""

from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, input_steer, log, streams

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
        "hint": "Wins back detail late in the run. No extra model calls; not for heun-style samplers.",
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
    steer = input_steer.Steer("Sharpness")
    steer.attach(patcher, key)

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
        step = steer.begin(x, t, to)
        out = executor(step.x, t, *args, **kwargs)
        where, split = dit_hooks.current_step(to), streams.video_of(out, named)
        if where is None or split is None or dit_hooks.probing(to):
            return out
        video, rebuild = split
        which = tuple(int(v) for v in ((to or {}).get("cond_or_uncond") or ()))
        before = seen.get(which)
        seen[which] = (where[0], video.detach())
        if (before is None or where[0] < before[0] or before[1].shape != video.shape      # nothing to compare, or a new pass
                or not step.steering or where[0] < where[1] * (1.0 - last)):
            return out
        edit = 0.5 * amount * (video - before[1].to(video.dtype))
        if x.dim() == 3:                             # packed: arrives scaled by (1 - next sigma), sized to move the input as a direct edit would
            sigma, nxt = float(t.max()), float(to["sample_sigmas"][where[0] + 1])
            edit = edit * (sigma - nxt) / max(sigma * (1.0 - nxt), 1e-6)
        return step.keep(out, rebuild(video + edit))

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)
    return f"amount {amount:g} on the last {last:.0%} of the steps"


PROVIDES = {"modifier": install}
