"""STAS: steer the massive activations of a video DiT (arXiv 2603.17825).

A few channels of a video transformer's hidden state carry huge values ("massive
activations"), strongest on the first frame and on the edges of each latent frame. They
act as the model's own rescalers. STAS sets them, at one block and over the first steps
only, to alpha x their current peak:

    g(i, d) = alpha * max_j |H[j, d]| * sign(H[i, d])     for i in S, d in M

S = every token of the first latent frame + the first and last p% of every frame's
tokens; M = channels whose peak is over 50x the mean magnitude. Paper settings: one
block ~30% deep, first 20 of 50 steps, p = 8%, alpha 1.2-2.5 (tested on Wan, CogVideoX).
No extra model call. Alpha is learned from ratings (rated_dial.py) or set by hand.
"""

import math

import torch

try:
    from .rated_dial import Dial, MODES  # noqa: F401
except ImportError:
    from rated_dial import Dial, MODES  # noqa: F401

DIAL = Dial("stas", start=2.0, lo=0.5, hi=3.0, explore=0.2)
DEFAULT_BLOCK = 15          # ~30% of H3's 50, the paper's depth (block 9 of Wan's 30)
EARLY_FRACTION = 0.4        # first 20 of 50 steps
EDGE = 0.08                 # p: head and tail share of each frame's tokens
MA_RATIO = 50.0
MAX_DIMS = 8


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
    stride = max(1, video.shape[0] // 1024)
    mean = video[::stride].float().abs().mean().clamp(min=1e-12)
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
