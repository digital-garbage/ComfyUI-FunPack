"""Camera move: pan and zoom the running latent partway through sampling (H3).

Writing a move into the STARTING noise failed on a distilled 4-step model: correlating the
noise across frames is statistics the model never saw, and it cannot repair that in four
steps. Here the noise stays white. From one step on, every model call sees the picture's
running latent moved frame by frame (frame t is shifted, or scaled about a focus point, by
a share that grows over the clip); the sampler's own state stays in the original frame, so
each returned prediction is moved back; the LAST call returns the model's own prediction on
the moved input. Shifting white noise gives white noise, so only the picture in the latent
moves, and the model repairs the edges and re-draws detail on the remaining steps.

For a model that treats positions alike the result is the picture the model would have made,
panned or zoomed. Honest limit: that is a camera move on a flat picture (no new parallax),
with the model re-drawing what it uncovers.

Values: pan is the fraction of the frame the picture travels over the clip (x positive =
camera moves right, so the picture moves left; y positive = camera moves down), zoom is the
end scale (above 1 = camera moves in), focus is where the zoom aims (0..1), `step` is the
1-based sampling step from which the model sees the moved latent (a knob in the interface:
earlier = a stronger move that the model builds on for longer, later = closer to a plain pan
of the finished video; a schedule shorter than `step` uses its last step).
"""

import math
from dataclasses import dataclass

import torch

MODES = ("off", "manual")


def clamp_pan(v):
    """A pan share in [-1, 1]; anything unreadable (NaN) is no pan."""
    v = float(v)
    return 0.0 if v != v else min(max(v, -1.0), 1.0)


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

    def describe(self, steps=None):
        bits = []
        for v, neg, pos in ((self.pan_x, "left", "right"), (self.pan_y, "up", "down")):
            if v:
                bits.append(f"pan {pos if v > 0 else neg} {abs(v):.2f}")
        if abs(math.log(self.zoom)) >= 1e-6:
            bits.append(f"zoom {'in' if self.zoom > 1 else 'out'} x{self.zoom:.2f} toward "
                        f"({self.focus_x:.2f}, {self.focus_y:.2f})")
        text = ", ".join(bits)
        if steps:
            text += f", from step {self.first_step(steps) + 1} of {steps}"
        return text


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
    iy, ix = sources(move, t, h, w)
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
    iy, ix = sources(move, t, h, w)
    ok_y, ok_x = (iy >= 0) & (iy < h), (ix >= 0) & (ix < w)
    out = original.clone()
    for f in range(t):
        ys, xs = iy[f][ok_y[f]], ix[f][ok_x[f]]
        out[:, f][:, ys[:, None], xs[None, :]] = moved[:, f][:, ok_y[f]][:, :, ok_x[f]]
    return out
