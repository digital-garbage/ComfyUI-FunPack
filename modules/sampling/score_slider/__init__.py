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

from ..._core import dit_hooks, log, registry, streams

ID = "score_slider"
TITLE = "Taste slider"
MOUNT = "generation.sampling"
STAGE = "sampling"          # innermost APPLY_MODEL: outer features see one combined result
CATEGORY = "guidance"
STATUS = "experimental"
REQUIRES = ["temporal_latent"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Taste slider",
        "hint": "Leans the finish toward what your liked prompts share. Costs 2 extra passes per late step.",
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

KIND = "prompt_taste"
NAME = "pooled"
MIN_LIKED = 3               # v4: direction_count >= 3
STEP = 0.15                 # the nudge, as a fraction of a word row's size
NEAREST, MIN_SIM = 3, 0.5   # v4's taste_nearest_prompt


def _say(message):
    log.once(f"{ID}:{message}", log.ALERT, "FunPack Taste slider", message)


def direction(rows, current=None, similar=False):
    """-> (unit vector | None, how it was found)."""
    pooled = [(r["rows"][NAME].float(), r["reward"]) for r in rows if NAME in r["rows"]]
    liked = [p for p, w in pooled if w > 0]
    if len(liked) < MIN_LIKED:
        return None, f"needs {MIN_LIKED} liked clips ({len(liked)} so far)"
    mean = torch.stack([p for p, _w in pooled]).mean(0)
    if similar and current is not None:
        near = sorted(((float(F.cosine_similarity(p, current.float(), dim=0)), p) for p in liked),
                      key=lambda s: -s[0])[:NEAREST]
        near = [(s, p) for s, p in near if s >= MIN_SIM]
        if near:
            d = sum(s * F.normalize(p - mean, dim=0) for s, p in near)
            if d.norm() > 1e-8:
                return F.normalize(d, dim=0), f"from {len(near)} similar liked prompt(s)"
    d = torch.stack(liked).mean(0) - mean
    if d.norm() <= 1e-8:
        return None, "liked prompts don't differ from the rest yet"
    return F.normalize(d, dim=0), f"from {len(liked)} liked of {len(pooled)} rated"


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
        pooled = c[0][words].float().mean(0).detach()
        captured[NAME] = pooled
        if "dir" not in live:
            live["dir"], how = direction(taste.rows(KIND), pooled, similar)
            log.once(f"{ID}:state", log.INFO, "FunPack Taste slider", f"key {taste.key!r}: {how}")
        base = executor(x, t, *args, **kwargs)
        amount = eta * dit_hooks.late_half(named.get("transformer_options"))
        d = live["dir"]
        if d is None or amount == 0.0 or d.numel() != c.shape[-1]:
            return base
        size = torch.linalg.vector_norm(c[:, words], dim=-1, dtype=torch.float32).mean()
        step = (d.to(c.device) * STEP * size).to(c.dtype) * words.view(1, -1, 1)
        split = streams.video_of(base, named)
        if split is None:
            _say("off this run: could not find the picture in this model's latent")
            return base
        plus = streams.video_of(executor(x, t, **{**named, "c_crossattn": c + step}), named)
        minus = streams.video_of(executor(x, t, **{**named, "c_crossattn": c - step}), named)
        video, rebuild = split
        return rebuild(video + (plus[0] - minus[0]) * amount)

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)
    if eta == 0.0:
        return "strength 0: learning only"
    return (f"eta {eta:g}, last half of steps{', similar prompts first' if similar else ''}; "
            f"steers from {MIN_LIKED} liked clips (read fresh every run)")


PROVIDES = {"modifier": install}
