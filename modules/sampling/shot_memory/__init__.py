"""Shot memory: the starting noise remembers the shots you liked.

With the prompt fixed, what a like or dislike reacts to is mostly the SHOT -- where
things sit, how the frame is split. The model decides that from the coarse part of
the starting noise, so a rating lands on the thing that caused it instead of on an
averaged direction in a space thousands of dimensions wide.

Each run's coarse noise (one value per 4x4-latent cell per channel, averaged over
the whole clip) waits on the taste key for a rating. The next run then either
reuses a liked shot -- its coarse noise blended into the fresh noise, the fine
noise always fresh so the details are new -- or starts fresh. Both choices learn
from ratings (Thompson draws): whether reusing beats fresh for this key, which
liked shot, and how strongly (one number, nudged toward what was liked). A liked
child is a shot in its own right, so good shots breed.

The noise stays unit Gaussian: swapping one block mean for another of the same
variance changes nothing the model can tell apart from ordinary noise. Zero model
calls. Sound noise is never touched. A latent that is not empty (a second pass, an
anchor) is left alone, and the run says so.

Unvalidated on a GPU. v4 lessons kept: reusing a shot's own seed stacks its layout
1.41x (the blend is capped per channel, only ever turned down), and a one-cell grid
has no spread (left alone).
"""

import math
import random

import torch
import torch.nn.functional as F
from comfy.patcher_extension import WrappersMP

from ..._core import log, registry, streams

ID = "shot_memory"
TITLE = "Shot memory"
MOUNT = "generation.sampling"
STAGE = "latent"
CATEGORY = "sampling"
STATUS = "experimental"
REQUIRES = ["temporal_latent"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Shot memory",
        "hint": "Starts from the layout of shots you liked, with new details.",
    },
    "mode": {
        "type": "enum", "default": "learned", "options": [
            {"value": "learned", "label": "Learn from ratings"},
            {"value": "manual", "label": "My value"},
        ],
        "label": "How much", "ui": "segmented",
        "hint": "Learn: how strongly to reuse is tuned by your ratings.",
        "when": {"enabled": True},
    },
    "amount": {
        "type": "float", "default": 0.7, "min": 0.0, "max": 0.95, "step": 0.05,
        "label": "Reuse amount", "ui": "slider",
        "hint": "How much of the liked layout carries over. 0 = none.",
        "when": {"enabled": True, "mode": "manual"},
    },
}

KIND = "shot_memory"
# Latent pixels per coarse cell: 64 image px on H3's 16x VAE.
# ponytail: fixed; learn it per key if one cell size stops fitting every resolution.
CELL = 4
MAX_ROWS = 64
START_AMOUNT = 0.7
EXPLORE = 0.15
LR = 0.5
AMOUNT_MIN, AMOUNT_MAX = 0.2, 0.95


def _say(message, level=log.ALERT):
    log.once(f"{ID}:{message}", level, "FunPack Shot memory", message)


def _sign(x):
    return (x > 0) - (x < 0)


def coarse_of(video):
    """[1, C, T, H, W] noise -> its coarse part [C, h, w], unit variance, or None."""
    if video.dim() != 5 or video.shape[0] != 1:
        return None
    _b, c, t, H, W = video.shape
    h, w = H // CELL, W // CELL
    if h < 2 or w < 2:          # a layout needs cells to compare; one cell has no spread
        return None
    blocks = video[0, :, :, :h * CELL, :w * CELL].float().reshape(c, t, h, CELL, w, CELL)
    return blocks.mean(dim=(1, 3, 5)) * math.sqrt(t * CELL * CELL)


def with_coarse(video, coarse):
    """`video` with its coarse part replaced by `coarse` [C, h, w]; the fine part kept."""
    _b, c, t, _H, _W = video.shape
    h, w = coarse.shape[-2:]
    core = video[0, :, :, :h * CELL, :w * CELL].float().reshape(c, t, h, CELL, w, CELL)
    delta = coarse.to(core) / math.sqrt(t * CELL * CELL) - core.mean(dim=(1, 3, 5))
    core = core + delta[:, None, :, None, :, None]
    out = video.clone()
    out[0, :, :, :h * CELL, :w * CELL] = core.reshape(c, t, h * CELL, w * CELL).to(video.dtype)
    return out


def fit(coarse, h, w, device=None):
    """A stored coarse map on this run's grid and device (stored maps live on the CPU; the
    noise is on the GPU). Another resolution is resampled and re-standardised: the layout
    survives, exact Gaussianity does not."""
    coarse = coarse.float().to(device) if device is not None else coarse.float()
    if tuple(coarse.shape[-2:]) == (h, w):
        return coarse
    x = F.interpolate(coarse[None], size=(h, w), mode="bilinear", align_corners=False)[0]
    return (x - x.mean()) / x.std().clamp(min=1e-6)


def pooled(cond):
    if not torch.is_tensor(cond):
        return None
    p = cond.detach().float()
    while p.dim() > 1:
        p = p.mean(dim=0)
    return p.cpu()


def shots(rows):
    """The key's rated shots, oldest first, as plain dicts."""
    out = []
    for r in rows:
        p = r["rows"]
        if "coarse" in p and "id" in p:
            out.append({"id": int(p["id"]), "parent": int(p["parent"]), "amount": float(p["amount"]),
                        "reward": float(r["reward"]), "coarse": p["coarse"], "cond": p.get("cond")})
    return out


def learned_amount(rows):
    """How strongly to reuse, replayed from the rated reuses in order: a liked one pulls
    the value toward the amount it used, a disliked one pushes it away."""
    m = START_AMOUNT
    for r in rows:
        if r["parent"] < 0:
            continue
        m += LR * _sign(r["reward"]) * (r["amount"] - m)
        m = min(max(m, AMOUNT_MIN), AMOUNT_MAX)
    return m


def choose_parent(rows, cond, channels, rng=random):
    """-> (liked shot to reuse | None, why). Two Thompson draws: reuse-vs-fresh from how
    reuses and fresh shots have been rated, then which liked shot, weighted by how alike
    its prompt was."""
    by_id = {r["id"] for r in rows}
    reuse_ab, fresh_ab, kids = [1.0, 1.0], [1.0, 1.0], {}
    for r in rows:
        s = _sign(r["reward"])
        if not s:
            continue
        (fresh_ab if r["parent"] < 0 else reuse_ab)[0 if s > 0 else 1] += 1.0
        if r["parent"] in by_id:
            kids.setdefault(r["parent"], [1.0, 1.0])[0 if s > 0 else 1] += 1.0
    liked = [r for r in rows if r["reward"] > 0 and int(r["coarse"].shape[0]) == int(channels)]
    if not liked:
        return None, "fresh: no liked shot yet"
    reuse, fresh_ = rng.betavariate(*reuse_ab), rng.betavariate(*fresh_ab)
    if fresh_ >= reuse:
        return None, f"fresh (reuse {reuse:.2f} < fresh {fresh_:.2f})"
    best, best_theta = None, -1.0
    for r in liked:
        a, b = kids.get(r["id"], [1.0, 1.0])
        theta = rng.betavariate(a + 1.0, b)              # +1: its own like
        stored = r["cond"]
        if cond is not None and torch.is_tensor(stored) and stored.numel() == cond.numel():
            # ponytail: raw pooled-prompt cosine; sits high for most prompts under one key.
            theta *= float(F.cosine_similarity(cond, stored.float(), dim=0).clamp(0.0, 1.0))
        if theta > best_theta:
            best, best_theta = r, theta
    return best, f"reusing a liked shot of {len(liked)} (reuse {reuse:.2f} > fresh {fresh_:.2f})"


def blend(own, parent_coarse, amount):
    """The layout to start from: the liked shot's, `amount` of the way."""
    coarse = amount * fit(parent_coarse, *own.shape[-2:], device=own.device) + math.sqrt(1.0 - amount ** 2) * own
    # Unit variance only holds when the two layouts are unrelated. Rerunning the liked
    # shot's own seed makes them the same pattern and the blend stacks it (1.41x at 0.7):
    # back to a normal-strength layout, per channel. Only ever down: a few-cell grid's
    # measured spread can read low by chance, and dividing by it would strengthen it.
    return coarse / coarse.std(dim=(1, 2), keepdim=True).clamp(min=1.0)


def _prompt_of(guider):
    try:
        return pooled(guider.conds["positive"][0]["cross_attn"])
    except (AttributeError, KeyError, IndexError, TypeError):
        return None


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    taste = registry.current().ask("taste_store", patcher)
    if taste is None:
        _say("off: set a Taste key first -- it is what your ratings teach")
        return None
    manual = values.get("mode") == "manual"
    manual_amount = min(max(float(values.get("amount", START_AMOUNT)), 0.0), AMOUNT_MAX)
    live = {"rows": [], "plan": None}

    def fresh():
        live["rows"], live["plan"] = shots(taste.rows(KIND)), None
        liked = sum(1 for r in live["rows"] if r["reward"] > 0)
        log.once(f"{ID}:state", log.INFO, "FunPack Shot memory",
                 f"key {taste.key!r}: {liked} liked shot(s) of {len(live['rows'])} rated")

    captured = taste.collect(patcher, key, KIND, keep=MAX_ROWS, fresh=fresh)

    def plan(cond, channels):
        if live["plan"] is None:
            parent, why = choose_parent(live["rows"], cond, channels)
            if parent is not None and manual and manual_amount <= 0.0:
                parent, why = None, "fresh (reuse amount is 0)"       # not a reuse: do not record one
            amount = 0.0
            if parent is not None:
                amount = manual_amount if manual else min(max(
                    learned_amount(live["rows"]) + random.gauss(0.0, EXPLORE), AMOUNT_MIN), AMOUNT_MAX)
                why += f", amount {amount:.2f}"
            log.once(f"{ID}:plan", log.INFO, "FunPack Shot memory", why)
            live["plan"] = (parent, amount)
        return live["plan"]

    def sampler_sample(executor, model_wrap, sigmas, extra_args, callback, noise,
                       latent_image=None, denoise_mask=None, disable_pbar=False):
        def run(n):
            return executor(model_wrap, sigmas, extra_args, callback, n,
                            latent_image, denoise_mask, disable_pbar)

        named = {"latent_shapes": getattr(getattr(model_wrap, "inner_model", None), "latent_shapes", None)}
        split = streams.video_of(noise, named)
        if split is None:
            _say("off this run: could not find the picture in this model's noise")
            return run(noise)
        video, rebuild = split
        base = streams.video_of(latent_image, named) if latent_image is not None else None
        if base is None or bool(torch.count_nonzero(base[0])):
            _say("Inactive | the starting latent is not empty (a second pass or an anchor), "
                 "so shot memory leaves this run's noise alone")
            return run(noise)
        own = coarse_of(video)
        if own is None:
            _say(f"Inactive | the picture is under {2 * CELL}x{2 * CELL} latent px (two layout "
                 "cells a side) or batched, so shot memory leaves it alone")
            return run(noise)
        cond = _prompt_of(model_wrap)
        parent, amount = plan(cond, int(own.shape[0]))
        coarse = own
        if parent is not None:
            coarse = blend(own, parent["coarse"], amount)
            video = with_coarse(video, coarse)
        captured.update({
            "id": torch.tensor(random.getrandbits(62)),
            "parent": torch.tensor(parent["id"] if parent else -1),
            "amount": torch.tensor(float(amount)), "coarse": coarse.half().cpu()})
        if cond is not None:
            captured["cond"] = cond.half()
        return run(rebuild(video))

    patcher.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, key, sampler_sample)
    return ("learned" if not manual else f"manual {manual_amount:g}") + ": starts from a liked layout when one is drawn"


PROVIDES = {"modifier": install}
