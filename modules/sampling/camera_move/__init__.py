"""Camera move: pan and zoom the running latent partway through sampling (H3).

Writing a move into the STARTING noise failed on a distilled 4-step model: correlating the
noise across frames is statistics the model never saw, and it cannot repair that in four
steps. Here the noise stays white. From one step on, every model call sees the picture's
running latent moved frame by frame (frame t is shifted, or scaled about a focus point, by a
share that grows over the clip); the sampler's own state stays in the original frame, so each
returned prediction is moved back; the LAST call returns the model's own prediction on the
moved input. Shifting white noise gives white noise, so only the picture in the latent moves,
and the model repairs the edges and re-draws detail on the remaining steps.

Honest limit: a camera move on a flat picture (no new parallax), the model re-drawing what
it uncovers. Unvalidated on a GPU.

Values: pan is the fraction of the frame the picture travels over the clip (x positive =
camera moves right, so the picture moves left; y positive = camera moves down), zoom is the
end scale (above 1 = camera moves in), focus is where the zoom aims (0..1), `from step` is the
1-based step from which the model sees the moved latent (earlier = a stronger move the model
builds on longer; later = closer to a plain pan of the finished video).

Keyframe pins after frame 0 do not move with the picture and may fight the move.

Stands down, and says so, for: context windows, a latent that is not empty (second pass,
anchor), batched CFG, one latent frame, a model call that is off the schedule or repeats a
step (midpoint and second-order samplers).
"""

import math
from dataclasses import dataclass

import torch
from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, log, streams

ID = "camera_move"
TITLE = "Camera move"
MOUNT = "generation.sampling"
STAGE = "latent"            # added first = outermost: every other wrapper sees the moved latent
CATEGORY = "sampling"
STATUS = "experimental"
REQUIRES = ["temporal_latent"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Camera move",
        "hint": "Pans or zooms the picture while it is being drawn.",
    },
    "pan_x": {
        "type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
        "label": "Pan left / right", "ui": "slider",
        "hint": "Share of the frame the camera travels over the clip. Positive = right.",
        "when": {"enabled": True},
    },
    "pan_y": {
        "type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
        "label": "Pan up / down", "ui": "slider",
        "hint": "Positive = down.",
        "when": {"enabled": True},
    },
    "zoom": {
        "type": "float", "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
        "label": "Zoom", "ui": "slider",
        "hint": "End scale. Above 1 moves in, below 1 moves out.",
        "when": {"enabled": True},
    },
    "focus_x": {
        "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
        "label": "Zoom aims at (x)", "ui": "slider", "when": {"enabled": True},
    },
    "focus_y": {
        "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
        "label": "Zoom aims at (y)", "ui": "slider", "when": {"enabled": True},
    },
    "step": {
        "type": "int", "default": 3, "min": 1, "max": 20,
        "label": "From step",
        "hint": "Earlier = stronger move; later = closer to a pan of the finished video.",
        "when": {"enabled": True},
    },
}


def _say(message, level=log.ALERT):
    log.once(f"{ID}:{message}", level, "FunPack Camera move", message)


def finite(v, default, lo, hi):
    """`v` clamped to [lo, hi]; anything unreadable (NaN) becomes `default`."""
    v = float(v)
    return default if v != v else min(max(v, lo), hi)


@dataclass(frozen=True)
class Move:
    pan_x: float = 0.0
    pan_y: float = 0.0
    zoom: float = 1.0
    focus_x: float = 0.5
    focus_y: float = 0.5
    step: int = 3

    def still(self):
        return self.pan_x == 0.0 and self.pan_y == 0.0 and abs(math.log(self.zoom)) < 1e-6

    def first_step(self, steps):
        """The step index (0-based) from which the model sees the moved latent."""
        return min(max(int(self.step) - 1, 0), max(int(steps) - 1, 0))

    def describe(self):
        bits = []
        for v, neg, pos in ((self.pan_x, "left", "right"), (self.pan_y, "up", "down")):
            if v:
                bits.append(f"pan {pos if v > 0 else neg} {abs(v):.2f}")
        if abs(math.log(self.zoom)) >= 1e-6:
            bits.append(f"zoom {'in' if self.zoom > 1 else 'out'} x{self.zoom:.2f} toward "
                        f"({self.focus_x:.2f}, {self.focus_y:.2f})")
        return ", ".join(bits)


def _axis(t_count, size, pan, zoom, focus):
    """Source cell of every output cell on one axis -> [T, size] (may leave [0, size))."""
    u = torch.linspace(0.0, 1.0, t_count) if t_count > 1 else torch.zeros(1)
    scale = zoom ** u                                     # constant zoom rate
    f = focus * size
    p = torch.arange(size, dtype=torch.float32) + 0.5
    q = f + (p[None, :] - f) / scale[:, None] + (u * pan * size)[:, None]
    return torch.floor(q).long()


def sources(move, t_count, h, w):
    """-> (iy [T, h], ix [T, w]): the source cell each output cell copies."""
    return (_axis(t_count, h, move.pan_y, move.zoom, move.focus_y),
            _axis(t_count, w, move.pan_x, move.zoom, move.focus_x))


def warp(video, move, sigma, generator):
    """[C, T, h, w] -> the same latent seen through the move. Cells the move uncovers get
    fresh noise at the current noise level `sigma` (x = (1 - sigma) * picture + sigma * noise
    with the picture unknown there), so they look like the rest of a noisy latent."""
    c, t, h, w = video.shape
    iy, ix = (v.to(video.device) for v in sources(move, t, h, w))
    ok_y, ok_x = (iy >= 0) & (iy < h), (ix >= 0) & (ix < w)
    fresh = torch.randn(video.shape, generator=generator).to(video) * float(sigma)
    out = torch.empty_like(video)
    for f in range(t):
        moved = video[:, f][:, iy[f].clamp(0, h - 1)[:, None], ix[f].clamp(0, w - 1)[None, :]]
        keep = (ok_y[f][:, None] & ok_x[f][None, :])[None]
        out[:, f] = torch.where(keep, moved, fresh[:, f])
    return out


def unwarp(moved, original, move):
    """Carry `moved` (a prediction made on the warped latent) back into the original frame.
    Cells the move never showed keep `original`'s value, so the sampler's state there is
    left as it was."""
    c, t, h, w = moved.shape
    iy, ix = (v.to(moved.device) for v in sources(move, t, h, w))
    ok_y, ok_x = (iy >= 0) & (iy < h), (ix >= 0) & (ix < w)
    out = original.clone()
    for f in range(t):
        ys, xs = iy[f][ok_y[f]], ix[f][ok_x[f]]
        out[:, f][:, ys[:, None], xs[None, :]] = moved[:, f][:, ok_y[f]][:, :, ok_x[f]]
    return out


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    move = Move(pan_x=finite(values.get("pan_x", 0.0), 0.0, -1.0, 1.0),
                pan_y=finite(values.get("pan_y", 0.0), 0.0, -1.0, 1.0),
                zoom=finite(values.get("zoom", 1.0), 1.0, 0.5, 2.0),
                focus_x=finite(values.get("focus_x", 0.5), 0.5, 0.0, 1.0),
                focus_y=finite(values.get("focus_y", 0.5), 0.5, 0.0, 1.0),
                step=int(values.get("step", 3)))
    if move.still():
        _say("Inactive | no pan and no zoom is set, so there is nothing to move")
        return None
    live = {"empty": True, "seed": 0, "seen": set(), "multi": False, "moved": 0}

    def sampler_sample(executor, model_wrap, sigmas, extra_args, callback, noise,
                       latent_image=None, denoise_mask=None, disable_pbar=False):
        named = {"latent_shapes": getattr(getattr(model_wrap, "inner_model", None), "latent_shapes", None)}
        base = streams.video_of(latent_image, named) if latent_image is not None else None
        live.update(empty=base is not None and not bool(torch.count_nonzero(base[0])),
                    seed=int((extra_args or {}).get("seed") or 0), seen=set(), multi=False, moved=0)
        if not live["empty"]:
            _say("Inactive | the starting latent is not empty (a second pass or an anchor), "
                 "so the picture is not moved")
        out = executor(model_wrap, sigmas, extra_args, callback, noise,
                       latent_image, denoise_mask, disable_pbar)
        if live["moved"]:
            log.once(f"{ID}:result", log.INFO, "FunPack Camera move",
                     f"Active | {move.describe()}, from step {move.step}; moved {live['moved']} call(s)")
        return out

    patcher.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, key, sampler_sample)

    def apply_model(executor, x, t, *args, **kwargs):
        named = streams.model_args(args, kwargs)
        to = named.get("transformer_options") or {}
        where = dit_hooks.current_step(to)
        if dit_hooks.probing(to) or live["multi"] or not live["empty"]:
            return executor(x, t, *args, **kwargs)
        if where is None:
            _say("Inactive | this call is not a step of the schedule (a midpoint sampler?), so "
                 "it is not moved")
            return executor(x, t, *args, **kwargs)
        i, n = where
        if to.get("context_window") is not None:
            _say("Inactive | context windows are not supported, so the picture is not moved")
            return executor(x, t, *args, **kwargs)
        if i in live["seen"]:
            live["multi"] = True
            _say("Inactive | the model is called more than once per step (a second-order "
                 "sampler), so the picture is not moved from here on; use euler-style sampling")
            return executor(x, t, *args, **kwargs)
        live["seen"].add(i)
        if i < move.first_step(n):
            return executor(x, t, *args, **kwargs)
        split = streams.video_of(x, named)
        if split is None:
            _say("Inactive | picture and sound could not be told apart in the latent")
            return executor(x, t, *args, **kwargs)
        video, rebuild_x = split
        if video.shape[0] != 1:
            _say("Inactive | batched calls (CFG above 1) are not supported, so the picture is not moved")
            return executor(x, t, *args, **kwargs)
        if video.shape[2] < 2:
            _say("Inactive | one latent frame has nothing to move across")
            return executor(x, t, *args, **kwargs)
        gen = torch.Generator().manual_seed((live["seed"] ^ 0x5CA3E12A) + int(i))
        vid = video[0]
        moved = rebuild_x(warp(vid, move, float(t.max()), gen)[None])
        if i == move.first_step(n):
            # An edit another modifier filed one step ago is in the unmoved frame.
            args, kwargs = streams.with_options(args, kwargs, {**to, dit_hooks.FRAME_CHANGE: True})
        d = executor(moved, t, *args, **kwargs)
        live["moved"] += 1
        if i >= n - 1:
            return d                                 # the model's own prediction, left moved
        split_d = streams.video_of(d, named)
        if split_d is None:
            return d
        dv, rebuild_d = split_d
        return rebuild_d(unwarp(dv[0], vid, move)[None].to(dv.dtype))

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)
    return f"{move.describe()}, from step {move.step}"


PROVIDES = {"modifier": install}
