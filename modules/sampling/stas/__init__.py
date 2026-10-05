"""STAS: steer the massive activations of a video DiT (arXiv 2603.17825).

A few channels of a video transformer's hidden state carry huge values ("massive
activations"), strongest on the first frame and on the edges of each latent frame. They
act as the model's own rescalers. STAS sets them, at one block and over the first steps
only, to alpha x their current peak:

    g(i, d) = alpha * max_j |H[j, d]| * sign(H[i, d])     for i in S, d in M

S = every token of the first latent frame + the first and last p% of every frame's
tokens; M = channels whose peak is over 50x the mean magnitude (at most 8). Paper
settings: one block ~30% deep, first 20 of 50 steps, p = 8%, alpha 1.2-2.5 (tested on
Wan and CogVideoX, not H3). No extra model call. Alpha is learned from ratings
(core/rated_dial.py) or typed.

Whether H3 has massive channels at the chosen block is unknown: the run says which it
found, or that it steered nothing. v4's user saw motion get "much more fluid and
detailed" with this stack on; STAS is the prime suspect, not isolated.

Keep this block different from late-branch guidance's branch block: at the same block the
weak copy lacks this edit and guidance would amplify it.

Switches ComfyUI's model compiler off for the run (an in-place edit of one block's output).
"""

import math

import torch
from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, log, rated_dial, registry, streams

ID = "stas"
TITLE = "Massive-activation steering"
MOUNT = "generation.sampling"
STAGE = "guidance"
CATEGORY = "guidance"
STATUS = "experimental"
REQUIRES = ["temporal_latent", "dit_block_hooks"]
USES = ["taste_store"]
USES_WHEN = {"mode": "learned"}      # "My value" never asks the store

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Massive-activation steering",
        "hint": "Boosts the model's own rescaling channels on the first frame and frame edges, early on.",
    },
    "mode": {
        "type": "enum", "default": "learned", "options": [
            {"value": "learned", "label": "Learn from ratings"},
            {"value": "manual", "label": "My value"},
        ],
        "label": "Strength", "ui": "segmented",
        "hint": "Learn: each rating nudges it toward what you liked.",
        "when": {"enabled": True},
    },
    "alpha": {
        "type": "float", "default": 2.0, "min": 0.5, "max": 3.0, "step": 0.1,
        "label": "Alpha", "ui": "slider",
        "hint": "The paper used 1.2-2.5.",
        "when": {"enabled": True, "mode": "manual"},
    },
    "block": {
        "type": "int", "default": 15, "min": 0, "max": 49,
        "label": "Block",
        "hint": "About 30% deep, as in the paper (15 of H3's 50).",
        "when": {"enabled": True},
    },
}

KIND = "stas"
DIAL = rated_dial.Dial(start=2.0, lo=0.5, hi=3.0, explore=0.2)
EARLY_FRACTION = 0.4        # first 20 of 50 steps
EDGE = 0.08                 # p: head and tail share of each frame's tokens
MA_RATIO = 50.0
MAX_DIMS = 8


def _say(message, level=log.ALERT):
    log.once(f"{ID}:{message}", level, "FunPack STAS", message)


def first_steps(total):
    """How many of `total` steps are steered."""
    return max(1, math.ceil(EARLY_FRACTION * int(total)))


def target_rows(frames, per_frame, edge=EDGE):
    """Row indices (within the video rows) of S: all of frame 0, then the head and tail
    `edge` share of every frame. Rows are frame-major, as H3 packs them."""
    k = max(1, round(edge * per_frame))
    within = torch.cat([torch.arange(k), torch.arange(per_frame - k, per_frame)]).unique()
    rows = (torch.arange(frames)[:, None] * per_frame + within[None, :]).flatten()
    return torch.cat([torch.arange(per_frame), rows]).unique()


def ma_dims(video):
    """-> (channel indices, their peaks, their peak/mean ratios). `video` is [n, D]."""
    peak = torch.maximum(video.amax(dim=0).float().abs(), video.amin(dim=0).float().abs())
    mean = video[::max(1, video.shape[0] // 1024)].float().abs().mean().clamp(min=1e-12)
    ratio = peak / mean
    dims = torch.nonzero(ratio > MA_RATIO).flatten()
    if dims.numel() > MAX_DIMS:
        dims = dims[ratio[dims].topk(MAX_DIMS).indices]
    return dims, peak[dims], ratio[dims]


def steer(video, rows, dims, peaks, alpha):
    """Set video[rows, dims] to alpha * peak * sign, in place."""
    if not dims.numel() or not rows.numel():
        return video
    r, d = rows.to(video.device)[:, None], dims.to(video.device)[None, :]
    target = (alpha * peaks.to(video.device))[None, :] * torch.sign(video[r, d].float())
    video[r, d] = target.to(video.dtype)
    return video


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    n = dit_hooks.block_count(patcher)
    block = int(values.get("block", 15))
    if not 0 <= block < n:
        _say(f"off: block {block} is not a block of this model (0-{n - 1})")
        return None
    manual = values.get("mode") == "manual"
    taste = None if manual else registry.current().ask("taste_store", patcher)
    if not manual and taste is None:
        _say("off: learning needs a Taste key -- set one, or pick 'My value'")
        return None
    patch = tuple(getattr(getattr(getattr(patcher, "model", None), "diffusion_model", None),
                          "patch_size", (1, 2, 2)))
    live = {"alpha": float(values.get("alpha", 2.0)), "per_frame": None, "spans": {},
            "steered": 0, "empty": 0, "dims": None, "why": None}

    def fresh():
        live.update(spans={}, steered=0, empty=0, dims=None, why=None)
        if manual:
            return log.once(f"{ID}:state", log.INFO, "FunPack STAS",
                            f"manual alpha {live['alpha']:.2f} at block {block}")
        live["alpha"], centre, held = DIAL.pick(rated_dial.history(taste.rows(KIND, blind_to="image"), {"b": block}))
        log.once(f"{ID}:state", log.INFO, "FunPack STAS",
                 f"key {taste.key!r}: learned alpha {centre:.2f} from {held} rating(s), trying "
                 f"{live['alpha']:.2f} at block {block}")

    captured = taste.collect(patcher, key, KIND, fresh=fresh) if not manual else {}

    def finish(executor, *args, **kwargs):
        if manual:
            fresh()
        out = executor(*args, **kwargs)
        if live["steered"]:
            log.once(f"{ID}:result", log.INFO, "FunPack STAS",
                     f"Active | alpha {live['alpha']:.2f} at block {block}: steered "
                     f"{live['steered']} call(s), channels {live['dims']}"
                     + (f"; {live['empty']} call(s) had no massive channel" if live["empty"] else ""))
            if not manual:
                captured["v"] = torch.tensor(live["alpha"])
                captured["b"] = torch.tensor(block)       # an alpha means something per block
        else:
            _say("Inactive | " + (live["why"] or f"no channel at block {block} is over "
                                  f"{MA_RATIO:.0f}x the mean, nothing steered -- try another block"))
        return out

    patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, finish)

    def apply_model(executor, x, t, *args, **kwargs):
        shapes = streams.model_args(args, kwargs).get("latent_shapes") or ()
        video = [tuple(s) for s in shapes if len(tuple(s)) == 5]
        if video:
            _b, _c, _t, h, w = max(video, key=lambda s: math.prod(s))
            live["per_frame"] = -(-int(h) // patch[1]) * -(-int(w) // patch[2])
        return executor(x, t, *args, **kwargs)

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)

    def hook(args, extra):
        out = extra["original_block"](args)["img"]
        to = args.get("transformer_options")
        where = dit_hooks.current_step(to)
        if where is None:
            live["why"] = "no step schedule to read, nothing steered"
            return {"img": out}
        if where[0] >= first_steps(where[1]):
            return {"img": out}
        if not live["per_frame"]:
            live["why"] = "the picture's frame size was not seen (no latent shapes), nothing steered"
            return {"img": out}
        seq = int(out.shape[0])
        span = live["spans"].get(seq, "unset")
        if span == "unset":
            rows = dit_hooks.row_span(dit_hooks.target_rows(args.get("mod_segments"), seq, out.device, "video"))
            span = None
            if rows is not None and (rows[1] - rows[0]) % live["per_frame"] == 0:
                count = rows[1] - rows[0]
                span = (rows[0], count, target_rows(count // live["per_frame"], live["per_frame"]))
            live["spans"][seq] = span
        if span is None:
            live["why"] = "picture rows not found as whole frames in this sequence, nothing steered"
            return {"img": out}
        a, count, rows = span
        picture = out[a:a + count]                         # a view: steered in place
        dims, peaks, ratios = ma_dims(picture)
        if not dims.numel():
            live["empty"] += 1
            return {"img": out}
        steer(picture, rows, dims, peaks, live["alpha"])
        live["steered"] += 0 if dit_hooks.probing(to) else 1
        if live["dims"] is None:
            live["dims"] = ", ".join(f"{int(d)} ({float(r):.0f}x)" for d, r in zip(dims, ratios))
        return {"img": out}

    dit_hooks.add_block_hook(patcher, key, block, hook)
    dit_hooks.without_compiler(patcher, key)
    return (("manual" if manual else "learned") + f" alpha at block {block}, first "
            f"{EARLY_FRACTION:.0%} of steps, frame 0 + frame edges, picture rows")


PROVIDES = {"modifier": install}
