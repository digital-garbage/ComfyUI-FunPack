"""Taste prompt nudge: over the last half of the steps, shift the prompt's words toward what your liked
prompts have in common.

The same learned direction as the Taste slider (liked prompts relative to everything rated), applied the
cheap way: the words the model reads are nudged in place, once per step, with no extra model passes. It
grows from nothing at the halfway point as the steps go on (the last step reaches half to three
quarters of it, depending on the schedule length). Words only: reference-image
rows are left alone. Needs 3 liked clips on the taste key.

Shares the slider's learning (`prompt_taste`), so turning on either teaches both.
"""

import torch
from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, log, registry, streams
from ..score_slider import KIND, MIN_LIKED, NAME, cond_row, direction

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


def nudged(c, words, d, amount, width=None):
    """`c` with each word row moved by `amount` along unit direction `d` (v4's absolute size: the
    strength is the row shift itself). `width` limits it to the first channels: an audio+video model's
    words carry the sound's channels after the picture's, and those stay as they were."""
    step = (d.to(c.device) * amount).to(c.dtype)
    if width is not None:
        step = torch.cat([step[:width], torch.zeros_like(step[width:])])
    return c + step * words.view(1, -1, 1).to(c.dtype)


def picture_width(patcher, c):
    """How many leading channels of the words belong to the picture, or None when all do."""
    dm = getattr(getattr(patcher, "model", None), "diffusion_model", None)
    video, audio = getattr(dm, "cross_attention_dim", None), getattr(dm, "audio_cross_attention_dim", None)
    return int(video) if video and audio and int(c.shape[-1]) == int(video) + int(audio) else None


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    taste = registry.current().ask("taste_store", patcher)
    if taste is None:
        _say("off: set a Taste key first -- it is what your ratings teach")
        return None
    strength = max(0.0, min(0.1, float(values.get("strength", 0.02))))
    similar = bool(values.get("similar"))
    live = {}

    def fresh():
        live.clear()
        acted.clear()

    captured = taste.collect(patcher, key, KIND, fresh=fresh)
    acted = {"yes": False}

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
        row = cond_row(c, to)
        if not dit_hooks.probing(to) and row is not None:
            captured[NAME] = c[row][words].float().mean(0).detach()
        if "dir" not in live and (row is not None or not similar):     # "similar" needs the positive prompt
            live["dir"], how = direction(taste.rows(KIND), c[row or 0][words].float().mean(0), similar)
            log.once(f"{ID}:state", log.INFO, "FunPack Taste prompt nudge", f"key {taste.key!r}: {how}")
        if "dir" not in live:
            return executor(x, t, *args, **kwargs)
        amount = strength * dit_hooks.late_half(to)
        d = live["dir"]
        if dit_hooks.last_step(to) and not acted.get("yes") and not dit_hooks.probing(to):
            acted["yes"] = d is not None and amount > 0.0
            if not acted.get("yes"):
                _say("Inactive | nothing was nudged this run: " + (
                    "no direction learned yet (needs liked clips)" if d is None else
                    "the sampler did not report its steps, so the late-step gate cannot open"
                    if dit_hooks.current_step(to) is None else
                    "the schedule is too short for the late-step gate to open"))
        if d is None or amount <= 0.0:
            return executor(x, t, *args, **kwargs)
        acted["yes"] = True
        if d.numel() != c.shape[-1]:
            _say("off this run: the learned direction was taught on a different model's text width; "
                 "rate a few clips on this model")
            return executor(x, t, *args, **kwargs)
        new_c = nudged(c, words, d, amount, picture_width(patcher, c))
        if not acted.get("checked"):
            acted["checked"] = True
            moved = float((new_c - c).float().norm())
            wanted = amount * float(words.sum()) ** 0.5 * c.shape[0] ** 0.5
            if moved < 0.5 * wanted:
                _say(f"the nudge is mostly lost to {str(c.dtype).split('.')[-1]} rounding "
                     f"({moved / max(wanted, 1e-12):.0%} of it arrived); raise the strength")
        named = {**named, "c_crossattn": new_c}
        return executor(x, t, **named)

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)
    if strength == 0.0:
        return "strength 0: learning only"
    return (f"strength {strength:g}, last half of steps{', similar prompts first' if similar else ''}; "
            f"steers from {MIN_LIKED} liked clips (read fresh every run)")


PROVIDES = {"modifier": install}
