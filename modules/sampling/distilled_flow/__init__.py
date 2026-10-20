"""Distilled Flow: samplers for few-step distilled flow models, listed in ComfyUI like any other.

`dflow_ord1` is Euler (same speed as `euler`). `dflow_ord2` is Adams-Bashforth 2: it reuses the last
prediction for a second-order step, with no extra model calls. `dflow_ord2_ab2` fades that second order
in across the run (early noisy steps stay first order, late detail steps get the full second order),
which is gentler on short schedules. A first-order version of the fade would be plain `dflow_ord1`, so it
has no name of its own.

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


def make(order, ramp):
    @torch.no_grad()
    def sample(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        steps = len(sigmas) - 1
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
            x = x + d * (to - sigma)
        return x
    return sample


def register():
    """Add every preset to ComfyUI's own sampler list (in place: its Sampler nodes share the list)."""
    for name, order, ramp in (("dflow_ord1", 1, False), ("dflow_ord2", 2, False), ("dflow_ord2_ab2", 2, True)):
        if name not in comfy.samplers.SAMPLER_NAMES:
            setattr(ksampling, f"sample_{name}", make(order, ramp))
            comfy.samplers.SAMPLER_NAMES.append(name)


register()
