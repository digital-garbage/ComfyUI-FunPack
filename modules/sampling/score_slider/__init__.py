"""Taste slider: over the last half of the steps, lean the picture toward what your
liked prompts have in common.

FreeSliders (arXiv 2511.00103), in score space: the step is predicted twice more,
with the prompt's words nudged a little TOWARD and a little AWAY from your
taste, and the picture moves by the difference -- `base + eta * (plus - minus)`.
The taste direction is where your liked clips' prompts sit relative to the
average of everything you rated (v4's liked_dir: liked minus the session mean).

"Similar prompts only" (v4's taste_nearest_prompt): the direction comes from
liked clips whose prompts resemble this one, when any do; otherwise the whole
key's.

Costs two extra model passes per step while it acts (the last half). Words only:
reference-image rows in the prompt are left alone. Picture only: sound is the
base prediction's.
"""

import torch
import torch.nn.functional as F
from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, input_steer, log, registry, streams
from ..prompt_taste import KIND, MIN_LIKED, NAME, cond_row, direction      # shared with prompt_nudge and velocity_bias: a plain file, not a module

ID = "score_slider"
TITLE = "Taste slider"
MOUNT = "generation.sampling"
STAGE = "sampling"          # innermost APPLY_MODEL: outer features see one combined result
CATEGORY = "guidance"
STATUS = "experimental"
REQUIRES = ["temporal_latent"]
USES = ["taste_store"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Taste slider",
        "hint": "Leans the finish toward what your liked prompts share. Needs 3 liked clips. Costs 2 extra passes per late step.",
    },
    "strength": {
        "type": "float", "default": 1.0, "min": -3.0, "max": 3.0, "step": 0.25,
        "label": "Slider", "ui": "slider",
        "hint": "Positive leans toward your taste, negative away. 0 = only learn.",
        "when": {"enabled": True},
    },
    "similar": {
        "type": "bool", "default": False,
        "label": "Similar prompts only",
        "hint": "Learn only from liked clips whose prompt resembles this one, when there are any.",
        "when": {"enabled": True},
    },
}

STEP = 0.15                 # the nudge, as a fraction of a word row's size


def _say(message):
    log.once(f"{ID}:{message}", log.ALERT, "FunPack Taste slider", message)


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    taste = registry.current().ask("taste_store", patcher)
    if taste is None:
        _say("off: set a Taste key first -- it is what your ratings teach")
        return None
    eta = float(values.get("strength", 1.0))
    similar = bool(values.get("similar"))
    live = {}

    def fresh():
        live.clear()

    captured = taste.collect(patcher, key, KIND, fresh=fresh)
    steer = input_steer.Steer("Taste slider")
    steer.attach(patcher, key)

    def apply_model(executor, x, t, *args, **kwargs):
        named = streams.model_args(args, kwargs)
        step = steer.begin(x, t, named.get("transformer_options"))
        x = step.x
        c = named.get("c_crossattn")
        if not torch.is_tensor(c) or c.dim() != 3:
            _say("off this run: no text conditioning reached the model")
            return executor(x, t, *args, **kwargs)
        words = registry.current().ask("text_rows", named, int(c.shape[1]))
        if words is None:
            words = torch.ones(c.shape[1], dtype=torch.bool, device=c.device)
        words = words.to(c.device)
        row = cond_row(c, named.get("transformer_options"))
        pooled = c[row or 0][words].float().mean(0).detach()
        if not dit_hooks.probing(named.get("transformer_options")) and row is not None:
            captured[NAME] = pooled
        if "dir" not in live and (row is not None or not similar):
            live["dir"], how = direction(taste.rows(KIND), pooled, similar)
            log.once(f"{ID}:state", log.INFO, "FunPack Taste slider", f"key {taste.key!r}: {how}")
        base = executor(x, t, *args, **kwargs)
        amount = eta * step.gate
        d = live.get("dir")
        if d is None or amount == 0.0 or d.numel() != c.shape[-1]:
            return base
        size = torch.linalg.vector_norm(c[:, words], dim=-1, dtype=torch.float32).mean()
        nudge = (d.to(c.device) * STEP * size).to(c.dtype) * words.view(1, -1, 1)
        split = streams.video_of(base, named)
        if split is None:
            _say("off this run: could not find the picture in this model's latent")
            return base
        # Perturbed-text passes are not the clip's own forward: measuring probes skip them.
        aside = {**named, "transformer_options": {**(named.get("transformer_options") or {}),
                                                  dit_hooks.PROBE: True}}
        plus = streams.video_of(executor(x, t, **{**aside, "c_crossattn": c + nudge}), named)
        minus = streams.video_of(executor(x, t, **{**aside, "c_crossattn": c - nudge}), named)
        video, rebuild = split
        return step.keep(base, rebuild(video + (plus[0] - minus[0]) * amount))

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)
    if eta == 0.0:
        return "strength 0: learning only"
    return (f"eta {eta:g}, last half of steps{', similar prompts first' if similar else ''}; "
            f"steers from {MIN_LIKED} liked clips (read fresh every run)")


PROVIDES = {"modifier": install}
