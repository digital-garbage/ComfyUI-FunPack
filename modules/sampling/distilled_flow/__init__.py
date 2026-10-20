"""Distilled Flow: samplers for few-step distilled flow models, listed in ComfyUI like any other.

Names read as parts: `dflow_ord1` / `dflow_ord2` (first order, or Adams-Bashforth 2 which reuses the
last prediction: no extra model calls), `_ramp` (the second order fades in across the run instead of
acting from step 2), `_heun1` / `_heun2` (the last one or two steps are corrected with a second model
call, so they cost one more call each). Plain `dflow_ord1` is Euler: same speed as `euler`.

Deterministic, no noise injection. Flow models only (the model predicts a straight path to the picture).
A sound stream riding in the latent is stepped first order, because second order corrupts it.
"""

import comfy.k_diffusion.sampling as ksampling
import comfy.samplers
import torch

ID = "distilled_flow"
TITLE = "Distilled Flow samplers"
STAGE = "sampling"
CATEGORY = "sampling"
STATUS = "experimental"


def _picture_only(new, old):
    """`new` for the picture, `old` for the sound, when the latent carries both."""
    if getattr(new, "is_nested", False):
        from comfy.nested_tensor import NestedTensor
        return NestedTensor((new.unbind()[0], *old.unbind()[1:]))
    return new


def make(order, ramp, heun):
    @torch.no_grad()
    def sample(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        steps = len(sigmas) - 1
        # The final step (sigma 0) cannot be corrected, so `heun` counts the steps that can.
        first_heun = sum(1 for s in sigmas[1:] if float(s) > 0) - heun
        ones = x.new_ones([x.shape[0]])
        before = before_h = None
        for i in range(steps):
            sigma, to = sigmas[i], sigmas[i + 1]
            denoised = model(x, sigma * ones, **extra_args)
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
            h = float((sigma - to).abs())
            used = denoised
            if order >= 2 and before is not None and before_h > 1e-7 and h > 1e-7:
                r = min(5.0, max(0.1, h / before_h))
                used = _picture_only(denoised * (1 + r / 2) - before * (r / 2), denoised)
                if ramp and steps > 1:
                    used = denoised + (used - denoised) * (i / (steps - 1))
            before, before_h = denoised, h
            if float(to) == 0:
                return used
            d = ksampling.to_d(x, sigma, used)
            if i >= first_heun:
                guess = x + d * (to - sigma)
                d = _picture_only((d + ksampling.to_d(guess, to, model(guess, to * ones, **extra_args))) / 2, d)
                before = before_h = None             # x moved by a corrected direction: no history to extrapolate from
            x = x + d * (to - sigma)
        return x
    return sample


def register():
    """Add every preset to ComfyUI's own sampler list (in place: its Sampler nodes share the list)."""
    for order, ramp, tag in ((1, False, "ord1"), (2, False, "ord2"), (2, True, "ord2_ramp")):
        for heun, suffix in ((0, ""), (1, "_heun1"), (2, "_heun2")):
            name = f"dflow_{tag}{suffix}"
            if name not in comfy.samplers.SAMPLER_NAMES:
                setattr(ksampling, f"sample_{name}", make(order, ramp, heun))
                comfy.samplers.SAMPLER_NAMES.append(name)


register()
