import math

import torch

import comfy.k_diffusion.sampling as k_diffusion_sampling
import comfy.model_sampling
import comfy.samplers
import comfy.utils

def sample_funpack_hybrid_euler_2s(model, x, sigmas, extra_args=None, callback=None,
                                   disable=None, eta=1.0, s_noise=1.0,
                                   high_quality_pct=0.35, correction_blend=1.0):
    """
    Hybrid sampler:
    - Early schedule: classic Euler ancestral (fast).
    - Late schedule: DPM-Solver++(2S)-style correction (higher quality) blended onto
      the Euler ancestral proposal only where late-step detail matters most.

    This keeps the cost close to Euler ancestral while paying the extra model
    evaluation only on the tail of the schedule.
    """
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        return k_diffusion_sampling.sample_euler_ancestral(
            model, x, sigmas, extra_args=extra_args, callback=callback,
            disable=disable, eta=eta, s_noise=s_noise
        )

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = k_diffusion_sampling.default_noise_sampler(x, seed=seed)

    total_steps = max(0, len(sigmas) - 1)
    if total_steps <= 0:
        return x

    high_quality_pct = max(0.0, min(1.0, float(high_quality_pct)))
    correction_blend = max(0.0, min(1.0, float(correction_blend)))
    late_steps = max(1, int(math.ceil(total_steps * high_quality_pct))) if high_quality_pct > 0.0 else 0
    late_start = max(0, total_steps - late_steps)

    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    s_in = x.new_ones([x.shape[0]])

    for i in comfy.utils.model_trange(total_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised = model(x, sigma * s_in, **extra_args)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})

        sigma_down, sigma_up = k_diffusion_sampling.get_ancestral_step(sigma, sigma_next, eta=eta)

        if sigma_down == 0:
            x = denoised
        else:
            d = k_diffusion_sampling.to_d(x, sigma, denoised)
            dt = sigma_down - sigma
            x_euler = x + d * dt

            if i >= late_start and correction_blend > 0.0:
                t = t_fn(sigma)
                t_next = t_fn(sigma_down)
                h = t_next - t
                r = 0.5
                s = t + r * h

                x_mid = (sigma_fn(s) / sigma_fn(t)) * x - torch.expm1(-h * r) * denoised
                denoised_mid = model(x_mid, sigma_fn(s) * s_in, **extra_args)
                x_2s = (sigma_fn(t_next) / sigma_fn(t)) * x - torch.expm1(-h) * denoised_mid

                x = x_euler.lerp(x_2s, correction_blend)
            else:
                x = x_euler

            if sigma_next > 0 and eta > 0 and s_noise > 0:
                x = x + noise_sampler(sigma, sigma_next) * s_noise * sigma_up

    return x


class FunPackHybridEuler2SSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "eta": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Ancestral stochasticity. Keep at 1.0 for classic ancestral behaviour."
                }),
                "s_noise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Noise scale for ancestral noise injection."
                }),
                "high_quality_pct": ("FLOAT", {
                    "default": 0.35,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Fraction of late denoise steps that receive the slower 2S quality correction."
                }),
                "correction_blend": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Blend between Euler ancestral (0.0) and late-step 2S correction (1.0)."
                }),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "get_sampler"
    CATEGORY = "FunPack/Sampling"
    DESCRIPTION = "Hybrid sampler: Euler ancestral for most steps, DPM-Solver++(2S)-style correction only on late steps for a better quality/speed tradeoff."

    def get_sampler(self, eta, s_noise, high_quality_pct, correction_blend):
        sampler = comfy.samplers.KSAMPLER(
            sample_funpack_hybrid_euler_2s,
            extra_options={
                "eta": eta,
                "s_noise": s_noise,
                "high_quality_pct": high_quality_pct,
                "correction_blend": correction_blend,
            }
        )
        return (sampler,)
