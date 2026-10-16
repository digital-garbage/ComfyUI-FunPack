"""The low-pass filter ALG swaps in, and the per-step decision to use it.

Both are pure: given a latent and a factor you get a tensor, and given a sigma
you get a choice. That is deliberate -- the parts that decide what happens are
testable on a laptop with 8x8 tensors, and only the parts that cannot fail
quietly are left to the GPU.
"""

import torch


def blur_frames(latent, kappa, frame_indices=(0,)):
    """Selected frames of a [B, C, T, H, W] latent, low-passed.

    Bilinear down then up at `kappa`, which removes the high-frequency content an
    i2v model uses to copy its anchor frame instead of animating it (arXiv
    2506.08456). Frames outside the clip are skipped rather than guessed at.

    Returns None when the latent is not a shape this understands. None means
    "not applicable", and the caller must then leave the sampling untouched --
    a half-applied blur is worse than none.
    """
    if latent is None or getattr(latent, "ndim", 0) != 5:
        return None

    _b, _c, frames, height, width = latent.shape
    wanted = sorted({int(i) for i in frame_indices if 0 <= int(i) < frames})
    if not wanted:
        return None

    factor = max(1.0, float(kappa))
    small_h, small_w = max(1, round(height / factor)), max(1, round(width / factor))
    if (small_h, small_w) == (height, width):
        return None                              # nothing would change

    out = latent.clone()
    for index in wanted:
        frame = latent[:, :, index]              # [B, C, H, W]
        down = torch.nn.functional.interpolate(
            frame, size=(small_h, small_w), mode="bilinear", align_corners=False)
        up = torch.nn.functional.interpolate(
            down, size=(height, width), mode="bilinear", align_corners=False)
        out[:, :, index] = up.to(latent.dtype)
    return out


def use_blurred(sigma, threshold):
    """Whether this step gets the blurred anchor.

    High sigma is early, where the shortcut gets taken; the sharp anchor comes
    back once sigma drops past the threshold, so the detail is still there for
    the steps that resolve it.
    """
    try:
        value = float(sigma.flatten()[0]) if hasattr(sigma, "flatten") else float(sigma)
    except Exception:                            # noqa: BLE001
        return False                             # never cost a step over the guidance
    return value > float(threshold)


class AnchorSwap:
    """A denoiser proxy that re-pins the anchor from each call's own sigma.

    ALG's only per-step decision is blurred-or-sharp and it keys on sigma, which
    is an argument of every model call -- so it does not need to be inside a
    sampler's loop. That is what lets one implementation cover a stock KSampler,
    a distilled flow sampler, and multi-evaluation samplers like heun, where each
    evaluation gets the anchor its own sigma calls for.

    Everything else forwards, so the object it wraps cannot tell the difference.
    """

    _OWN = ("_inner", "_sharp", "_blurred", "_threshold")

    def __init__(self, inner, sharp, blurred, threshold):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_sharp", sharp)
        object.__setattr__(self, "_blurred", blurred)
        object.__setattr__(self, "_threshold", threshold)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        if name in AnchorSwap._OWN:
            object.__setattr__(self, name, value)
        else:
            # A sampler assigning to the denoiser must reach the real one.
            setattr(self._inner, name, value)

    def __call__(self, x, sigma, **kwargs):
        self._inner.latent_image = (
            self._blurred if use_blurred(sigma, self._threshold) else self._sharp)
        return self._inner(x, sigma, **kwargs)
