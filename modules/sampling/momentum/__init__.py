"""Momentum Guidance: smooth the fine-motion steps with a running average of past directions.

Each step's direction (where the model says the picture should move) is averaged into a running
memory. Once the noise has fallen below a level, the current direction is blended toward that memory,
which calms jitter in the refinement steps. It is the late-step partner of ALG, which acts on the early
ones (arXiv:2602.20360). It can damp motion; that is the price of averaging.

The edit is carried into the next step's input (core/input_steer.py), never onto the last answer, and
only the picture is touched, never the sound.
"""

import torch
from comfy.patcher_extension import WrappersMP

from ..._core import input_steer, log, streams

ID = "momentum"
TITLE = "Momentum guidance"
MOUNT = "generation.sampling"
STAGE = "guidance"
CATEGORY = "guidance"
STATUS = "experimental"
REQUIRES = ["temporal_latent"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Smooth the fine motion",
        "hint": "Calms jitter in the last steps. May also soften motion.",
    },
    "strength": {
        "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
        "label": "How much", "ui": "slider",
        "hint": "0 = nothing, 1 = follow the running average completely.",
        "when": {"enabled": True},
    },
    "decay": {
        "type": "float", "default": 0.5, "min": 0.0, "max": 0.95, "step": 0.05,
        "label": "How long it remembers", "ui": "slider",
        "hint": "High values hold on to the noisy first steps and make garbage on short schedules.",
        "when": {"enabled": True},
    },
    "below_sigma": {
        "type": "float", "default": 0.975, "min": 0.0, "max": 1.0, "step": 0.025,
        "label": "Starts at noise level", "ui": "slider",
        "hint": "Acts once the noise has dropped below this.",
        "when": {"enabled": True},
    },
}


def blend(x, out, sigma, ema, decay, strength):
    """(new answer, new memory) for one step on a video tensor. The direction is
    d = (x - answer) / sigma; the answer is rebuilt from d blended toward the memory."""
    d = ((x - out) / max(sigma, 1e-6)).detach()
    ema = d if ema is None else decay * ema + (1.0 - decay) * d
    mixed = d + strength * (ema - d)
    return x - sigma * mixed, ema


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    strength = max(0.0, min(1.0, float(values.get("strength", 0.5))))
    decay = max(0.0, min(0.999, float(values.get("decay", 0.5))))
    below = float(values.get("below_sigma", 0.975))
    if strength <= 0.0:
        return None
    state = {"ema": None, "last": None}
    steer = input_steer.Steer("Momentum")
    steer.attach(patcher, key)

    def apply_model(executor, x, t, *args, **kwargs):
        named = streams.model_args(args, kwargs)
        step = steer.begin(x, t, named.get("transformer_options"))
        out = executor(step.x, t, *args, **kwargs)
        sigma = float(t.max())
        if state["last"] is not None and sigma > state["last"] + 1e-4:
            state["ema"] = None                                  # sigma went back up: a new run
        state["last"] = sigma
        seen_out, seen_in = streams.video_of(out, named), streams.video_of(step.x, named)
        if seen_out is None or seen_in is None:
            log.once(f"{ID}:layout", log.ALERT, "FunPack Momentum",
                     "off this run: could not find the picture in this model's latent")
            return out
        video, rebuild = seen_out
        # The memory always accumulates, so the window opens on real history, not a cold start.
        new, state["ema"] = blend(seen_in[0].float(), video.float(), sigma, state["ema"], decay,
                                  strength if sigma < below else 0.0)
        if sigma >= below or not step.steering:
            return out
        return step.keep(out, rebuild(new.to(video.dtype)))

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)
    return f"strength {strength:g}, memory {decay:g}, from noise level {below:g}"


PROVIDES = {"modifier": install}
