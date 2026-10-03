"""Taste prompt nudge: over the last half of the steps, shift the prompt's words toward what your liked
prompts have in common.

The same learned direction as the Taste slider (liked prompts relative to everything rated), applied the
cheap way: the words the model reads are nudged in place, once per step, with no extra model passes. It
grows from nothing at the halfway point to full strength at the last step. Words only: reference-image
rows are left alone. Needs 3 liked clips on the taste key.

Shares the slider's learning (`prompt_taste`), so turning on either teaches both.
"""

import torch
from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, log, registry, streams
from ..score_slider import KIND, MIN_LIKED, NAME, direction

ID = "prompt_nudge"
TITLE = "Taste prompt nudge"
MOUNT = "generation.sampling"
STAGE = "sampling"
CATEGORY = "guidance"
STATUS = "experimental"
REQUIRES = ["temporal_latent"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Taste prompt nudge",
        "hint": "Shifts the prompt toward what your liked prompts share. Free: no extra passes.",
    },
    "strength": {
        "type": "float", "default": 0.02, "min": 0.0, "max": 0.1, "step": 0.005,
        "label": "How much", "ui": "slider",
        "hint": "Small is plenty: it adds up over the steps. 0 = only learn.",
        "when": {"enabled": True},
    },
    "similar": {
        "type": "bool", "default": False,
        "label": "Similar prompts only",
        "hint": "Learn only from liked clips whose prompt resembles this one, when there are any.",
        "when": {"enabled": True},
    },
}


def _say(message):
    log.once(f"{ID}:{message}", log.ALERT, "FunPack Taste prompt nudge", message)


def nudged(c, words, d, amount):
    """`c` with each word row moved along unit direction `d` by `amount` x the words' own size."""
    size = torch.linalg.vector_norm(c[:, words], dim=-1, dtype=torch.float32).mean()
    step = (d.to(c.device) * amount * size).to(c.dtype)
    return c + step * words.view(1, -1, 1).to(c.dtype)


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    taste = registry.current().ask("taste_store", patcher)
    if taste is None:
        _say("off: set a Taste key first -- it is what your ratings teach")
        return None
    strength = float(values.get("strength", 0.02))
    similar = bool(values.get("similar"))
    live = {}

    def fresh():
        live.clear()

    captured = taste.collect(patcher, key, KIND, fresh=fresh)

    def apply_model(executor, x, t, *args, **kwargs):
        named = streams.model_args(args, kwargs)
        c = named.get("c_crossattn")
        if not torch.is_tensor(c) or c.dim() != 3:
            _say("off this run: no text conditioning reached the model")
            return executor(x, t, *args, **kwargs)
        words = registry.current().ask("text_rows", named, int(c.shape[1]))
        if words is None:
            words = torch.ones(c.shape[1], dtype=torch.bool, device=c.device)
        words = words.to(c.device)
        to = named.get("transformer_options")
        if not dit_hooks.probing(to):
            captured[NAME] = c[0][words].float().mean(0).detach()
        if "dir" not in live:
            live["dir"], how = direction(taste.rows(KIND), c[0][words].float().mean(0), similar)
            log.once(f"{ID}:state", log.INFO, "FunPack Taste prompt nudge", f"key {taste.key!r}: {how}")
        amount = strength * dit_hooks.late_half(to)
        d = live["dir"]
        if d is None or amount <= 0.0:
            return executor(x, t, *args, **kwargs)
        if d.numel() != c.shape[-1]:
            _say("off this run: the learned direction was taught on a different model's text width; "
                 "rate a few clips on this model")
            return executor(x, t, *args, **kwargs)
        named = {**named, "c_crossattn": nudged(c, words, d, amount)}
        return executor(x, t, **named)

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)
    if strength == 0.0:
        return "strength 0: learning only"
    return (f"strength {strength:g}, last half of steps{', similar prompts first' if similar else ''}; "
            f"steers from {MIN_LIKED} liked clips (read fresh every run)")


PROVIDES = {"modifier": install}
