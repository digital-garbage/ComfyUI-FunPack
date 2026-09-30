"""Camera noise: a camera move written into the starting noise, no prompt words.

Shot memory showed the starting noise decides where things sit in the frame. A camera move
is that same pattern travelling: frame t's noise is frame 0's noise pulled sideways (pan)
or scaled about a point (zoom in / out), the way a printed sheet is pulled past a window.
The model, which draws each place from the noise around it, is nudged to draw the picture
travelling too.

Every output value is copied from a canvas of independent unit Gaussians (nearest cell), so
each frame stays exactly unit Gaussian. `amount` blends in each frame's own fresh noise
(independent, so the blend is rescaled to keep unit variance): 1.0 = the travelling noise
alone, lower = looser. Cells a pan or zoom-out uncovers are fresh canvas. Zero model calls;
audio noise is never touched.

All values are unit-free: pan is the fraction of the frame the noise travels over the clip
(x positive = camera moves right, so the picture moves left; y positive = camera moves down),
zoom is the end scale (above 1 = camera moves in), focus is where the zoom aims, 0..1.

Known cost: zooming in copies each source cell into several neighbours, so neighbouring
values become alike (adjacent correlation ~0.34 at zoom 1.5, ~0.51 at 2.0 in the travelling
part, roughly 0.2 to 0.3 after the default blend). Every value is still N(0, 1); whether the
model minds that blockiness is a question for the GPU.
"""

import math
from dataclasses import dataclass

import torch

MODES = ("off", "manual")
AMOUNT_MAX = 0.95


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
    amount: float = 0.8

    def still(self):
        return self.pan_x == 0.0 and self.pan_y == 0.0 and abs(math.log(self.zoom)) < 1e-6

    def describe(self):
        bits = []
        for v, neg, pos in ((self.pan_x, "left", "right"), (self.pan_y, "up", "down")):
            if v:
                bits.append(f"pan {pos if v > 0 else neg} {abs(v):.2f}")
        if abs(math.log(self.zoom)) >= 1e-6:
            bits.append(f"zoom {'in' if self.zoom > 1 else 'out'} x{self.zoom:.2f} toward "
                        f"({self.focus_x:.2f}, {self.focus_y:.2f})")
        return ", ".join(bits) + f", {self.amount:.0%} carried"


def _axis(t_count, size, pan, zoom, focus):
    """Continuous source coordinate of every output cell on one axis -> [T, size], the cell
    index it copies (floor) and the smallest index reached."""
    u = torch.linspace(0.0, 1.0, t_count) if t_count > 1 else torch.zeros(1)
    scale = zoom ** u                                     # constant zoom rate
    f = focus * size
    p = torch.arange(size, dtype=torch.float32) + 0.5
    q = f + (p[None, :] - f) / scale[:, None] + (u * pan * size)[:, None]
    return torch.floor(q).long()


def sources(move, t_count, h, w):
    """-> (iy [T, h], ix [T, w]) source cell per output cell, in frame coordinates (they
    leave [0, h) x [0, w) where a pan or zoom-out uncovers new cells)."""
    return (_axis(t_count, h, move.pan_y, move.zoom, move.focus_y),
            _axis(t_count, w, move.pan_x, move.zoom, move.focus_x))


def travel(move, channels, t_count, h, w, generator):
    """[C, T, h, w] unit Gaussian whose frame t is a fresh canvas seen through the move."""
    iy, ix = sources(move, t_count, h, w)
    y0, x0 = int(iy.min()), int(ix.min())
    canvas = torch.randn(channels, int(iy.max()) - y0 + 1, int(ix.max()) - x0 + 1,
                         generator=generator)
    return torch.stack([canvas[:, (iy[t] - y0)[:, None], (ix[t] - x0)[None, :]]
                        for t in range(t_count)], dim=1)


def shape(noise, samples, move, seed):
    """-> (noise, note). `note` starts with Active or Inactive, ready for the run log.
    The noise is returned untouched when the latent is not empty (a second pass, a latent
    anchor, a carried overlap: the noise does not decide the shot there) or has one frame."""
    nested = getattr(noise, "is_nested", False)
    video = noise.unbind()[0] if nested else noise
    base = samples.unbind()[0] if getattr(samples, "is_nested", False) else samples
    if not isinstance(video, torch.Tensor) or video.dim() != 5 or video.shape[0] != 1:
        return noise, "Inactive | this latent is not a single video"
    if move.still():
        return noise, "Inactive | no pan and no zoom set, so there is no move to write"
    if bool(torch.count_nonzero(base)):
        return noise, ("Inactive | this pass starts from a picture (second pass, latent anchor "
                       "or carried overlap), so the noise does not decide the shot")
    _b, c, t, h, w = video.shape
    if t < 2:
        return noise, "Inactive | one latent frame has nothing to travel across"
    gen = torch.Generator().manual_seed((int(seed) ^ 0x5CA3E12A) & 0x7FFFFFFFFFFFFFFF)
    moved = travel(move, c, t, h, w, gen).to(video.device, video.dtype)[None]
    a = min(max(float(move.amount), 0.0), AMOUNT_MAX)
    out = a * moved + math.sqrt(1.0 - a * a) * video
    if not nested:
        return out, "Active | " + move.describe()
    from comfy.nested_tensor import NestedTensor
    return NestedTensor([out, *noise.unbind()[1:]]), "Active | " + move.describe()
