"""Taste guidance: nudge each late step's predicted picture toward what you rated up.

Every run banks a small fingerprint of its final predicted picture on the taste
key. Once 10+ rated clips exist, a tiny judge learns to rank them (see the taste
module's value.py), and over the last half of the steps the prediction is moved
along the judge's gradient. No extra model pass: one backward through a few
hundred thousand weights.

`strength` is a fraction of the picture's own size per fully-open step (0.02 =
2%): the raw gradient reaches each latent value through a pooling window, so
used as-is it would be far too small to matter.

Sound untouched. Refuses batched calls (CFG > 1 puts the unconditioned
prediction in the same batch; steering their blend steers toward a prediction
averaged with its own opposite).
"""

from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, input_steer, log, registry, streams

ID = "output_guidance"
TITLE = "Taste guidance"
MOUNT = "generation.sampling"
STAGE = "guidance"
CATEGORY = "guidance"
STATUS = "experimental"
REQUIRES = ["temporal_latent"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Taste guidance",
        "hint": "After 10+ ratings, nudges the finish of each clip toward what you rated up.",
    },
    "strength": {
        "type": "float", "default": 0.02, "min": 0.0, "max": 0.2, "step": 0.005,
        "label": "Strength", "ui": "slider",
        "hint": "0 = only learn. 0.02 is a gentle start.",
        "when": {"enabled": True},
    },
}

KIND = "x0_final"
NAME = "final"


def _say(message):
    log.once(f"{ID}:{message}", log.ALERT, "FunPack Taste guidance", message)


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    taste = registry.current().ask("taste_store", patcher)
    if taste is None:
        _say("off: set a Taste key first -- it is what your ratings teach")
        return None
    strength = float(values.get("strength", 0.02))
    live = {"judge": None}

    def fresh():
        live["judge"] = taste.judge(KIND, NAME) if strength > 0 else None
        liked, disliked = taste.counts(KIND, NAME)
        log.once(f"{ID}:state", log.INFO, "FunPack Taste guidance",
                 f"key {taste.key!r}: " + (f"steering the last half from {liked} liked / {disliked} disliked"
                                            if live["judge"] is not None else
                                            f"learning ({liked} liked / {disliked} disliked; "
                                            "steers from 10 rated clips with both kinds)"))

    captured = taste.collect(patcher, key, KIND, fresh=fresh)
    steer = input_steer.Steer("Taste guidance")

    def apply_model(executor, x, t, *args, **kwargs):
        named = streams.model_args(args, kwargs)
        to = named.get("transformer_options")
        step = steer.begin(x, t, to)
        out = executor(step.x, t, *args, **kwargs)
        split = streams.video_of(out, named)
        if split is None:
            _say("off this run: could not find the picture in this model's latent")
            return out
        video, rebuild = split
        if video.shape[0] != 1:
            _say("off this run: batched predictions (CFG above 1) can't be steered honestly")
            return out
        if dit_hooks.last_step(to):
            captured[NAME] = taste.describe(video.detach())
        amount = strength * step.gate
        if live["judge"] is None or amount <= 0.0:
            return out
        return step.keep(out, rebuild(live["judge"].nudge(video, amount)))

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)
    return (f"strength {strength:g}, last half of steps; learns every run, "
            "steers from 10 rated clips (read fresh every run)")


PROVIDES = {"modifier": install}
