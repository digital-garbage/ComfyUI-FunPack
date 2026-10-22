"""Hop times and audio clocks of the PTD releases.

The EPFL grids and the shared video/audio clock follow ComfyUI-PTD (EPFL VITA Lab, Apache-2.0); the
32-interval PDD grid, head plans and chord correction follow ComfyUI-MiniMaxH3-PTD (taobao-content-ai,
Apache-2.0), both via ComfyUI-PTD-Compact (Nynxz). See the README Thank you section.
"""

from itertools import pairwise

import torch

T_MAX = 0.999                    # upper end of the shared clock
AUDIO_RELATIVE_SHIFT = 0.25

#: EPFL releases: hop times t_0 > ... > t_N = 0.
RECIPES = {
    "h3_4step": (0.999, 0.947, 0.857, 0.667, 0.0),
    "h3_6step": (0.999, 0.968, 0.923, 0.857, 0.750, 0.545, 0.0),
}

PDD_STEPS = 32                   # intervals of the Ref2VA releases' own grid
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


def _time(value):
    value = float(value)
    if not 0.0 <= value <= T_MAX:
        raise ValueError(f"PTD time must lie in [0, {T_MAX}], got {value}")
    return value


def audio_sigma(time):
    """Audio's noise level on the shared clock."""
    time = _time(time)
    if time == T_MAX:
        return T_MAX
    return AUDIO_RELATIVE_SHIFT * time * T_MAX / (T_MAX + (AUDIO_RELATIVE_SHIFT - 1.0) * time)


def audio_sigma_prime(time):
    """d(audio_sigma)/dt."""
    time = _time(time)
    return AUDIO_RELATIVE_SHIFT * T_MAX ** 2 / (T_MAX + (AUDIO_RELATIVE_SHIFT - 1.0) * time) ** 2


def native_audio_shift(time):
    """The T=1 shift that gives native H3 PTD's audio sigma at this time (video shift 1)."""
    time = _time(time)
    if time == 0.0:
        return AUDIO_RELATIVE_SHIFT
    if time == T_MAX:
        return 1.0
    sigma = audio_sigma(time)
    return sigma * (1.0 - time) / (time * (1.0 - sigma))


def _shifted(shift, sigma):
    return shift * sigma / (1 + (shift - 1) * sigma)


def pdd_grid(hops, steps=PDD_STEPS, shift=VIDEO_SHIFT):
    """Hop times of a Ref2VA file: the video block boundaries of its 32-interval grid, then 0."""
    size = steps // hops
    noise = _shifted(float(shift), torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64))
    return tuple(T_MAX * float(noise[h * size]) for h in range(hops)) + (0.0,)


def pdd_plans(hops, steps=PDD_STEPS, video_shift=VIDEO_SHIFT, audio_shift=AUDIO_SHIFT):
    """Per hop (video, audio) weights over the per-interval heads: the block's normalised step sizes."""
    size = steps // hops
    plans = []
    for shift in (video_shift, audio_shift):
        sizes = (1.0 - _shifted(float(shift), torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64))).diff()
        rows = []
        for h in range(hops):
            plan = torch.zeros(steps, dtype=torch.float64)
            block = sizes[h * size:(h + 1) * size]
            plan[h * size:(h + 1) * size] = block / block.sum()
            rows.append(plan.float())
        plans.append(rows)
    return list(zip(*plans))


def audio_chords(grid):
    """Per hop, the audio clock's mean slope. The Ref2VA audio head is the block-mean velocity."""
    return tuple((audio_sigma(a) - audio_sigma(b)) / (a - b) for a, b in pairwise(grid))


def audio_tangents(grid):
    """Per hop, the audio clock's slope at the hop start. The EPFL audio head is an instant velocity."""
    return tuple(audio_sigma_prime(a) for a in grid[:-1])
