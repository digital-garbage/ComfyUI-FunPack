import math

import torch

import comfy.k_diffusion.sampling as k_diffusion_sampling
import comfy.model_sampling
import comfy.samplers
import comfy.utils


def _sigma_fn(t):
    return t.neg().exp()


def _t_fn(sigma):
    return sigma.log().neg()


def _hybrid_ode_step(model, x, sigma, sigma_next, s_in, extra_args, correction_blend, denoised=None):
    if denoised is None:
        denoised = model(x, sigma * s_in, **extra_args)

    if sigma_next == 0:
        return denoised, denoised

    d = k_diffusion_sampling.to_d(x, sigma, denoised)
    dt = sigma_next - sigma
    x_euler = x + d * dt

    if correction_blend <= 0.0:
        return x_euler, denoised

    t = _t_fn(sigma)
    t_next = _t_fn(sigma_next)
    h = t_next - t
    r = 0.5
    s = t + r * h

    x_mid = (_sigma_fn(s) / _sigma_fn(t)) * x - torch.expm1(-h * r) * denoised
    denoised_mid = model(x_mid, _sigma_fn(s) * s_in, **extra_args)
    x_2s = (_sigma_fn(t_next) / _sigma_fn(t)) * x - torch.expm1(-h) * denoised_mid

    if correction_blend >= 1.0:
        return x_2s, denoised
    return x_euler.lerp(x_2s, correction_blend), denoised


def _renoise_to_sigma(x, current_sigma, target_sigma, restart_noise, noise_sampler):
    if target_sigma <= current_sigma or restart_noise <= 0.0:
        return x

    sigma_delta_sq = max(0.0, float(target_sigma * target_sigma - current_sigma * current_sigma))
    if sigma_delta_sq <= 0.0:
        return x

    sigma_delta = math.sqrt(sigma_delta_sq)
    return x + noise_sampler(current_sigma, target_sigma) * (restart_noise * sigma_delta)


def _find_restart_anchor_index(sigmas, late_start, total_steps, restart_trigger_pct):
    if late_start >= total_steps:
        return late_start

    quality_sigmas = sigmas[late_start:total_steps]
    if quality_sigmas.numel() <= 1:
        return late_start

    high_sigma = float(quality_sigmas[0].item())
    low_sigma = None
    for idx in range(int(quality_sigmas.shape[0]) - 1, -1, -1):
        value = float(quality_sigmas[idx].item())
        if value > 0.0:
            low_sigma = value
            break
    if low_sigma is None:
        return total_steps - 1

    quality_progress = 0.0
    if total_steps > late_start:
        quality_progress = (restart_trigger_pct * total_steps - late_start) / max(1e-6, float(total_steps - late_start))
    quality_progress = max(0.0, min(1.0, quality_progress))

    if high_sigma <= 0.0 or low_sigma <= 0.0 or abs(high_sigma - low_sigma) <= 1e-12:
        return min(total_steps - 1, max(late_start, int(round(late_start + quality_progress * max(0, total_steps - late_start - 1)))))

    log_high = math.log(high_sigma)
    log_low = math.log(low_sigma)
    target_log_sigma = log_high + (log_low - log_high) * quality_progress

    best_index = late_start
    best_distance = None
    for idx in range(late_start, total_steps):
        sigma_value = float(sigmas[idx].item())
        if sigma_value <= 0.0:
            continue
        distance = abs(math.log(sigma_value) - target_log_sigma)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = idx
    return best_index


def _build_restart_sigmas(sigmas, late_start, total_steps, restart_trigger_pct, restart_steps):
    if late_start >= total_steps:
        return sigmas[late_start:late_start], late_start

    anchor_index = _find_restart_anchor_index(sigmas, late_start, total_steps, restart_trigger_pct)
    anchor_index = min(total_steps - 1, max(late_start, anchor_index))

    if restart_steps <= 1:
        return sigmas[anchor_index:anchor_index + 1], anchor_index

    quality_sigmas = sigmas[late_start:total_steps]
    if quality_sigmas.numel() <= 1:
        return sigmas[anchor_index:anchor_index + 1], anchor_index

    positive_logs = []
    for idx in range(int(quality_sigmas.shape[0])):
        sigma_value = float(quality_sigmas[idx].item())
        if sigma_value > 0.0:
            positive_logs.append(math.log(sigma_value))

    if len(positive_logs) >= 2:
        total_log_span = abs(positive_logs[0] - positive_logs[-1])
        avg_log_step = total_log_span / max(1, len(positive_logs) - 1)
        target_log_span = avg_log_step * max(1, restart_steps - 1)
    else:
        target_log_span = 0.0

    restart_from = anchor_index
    accumulated_span = 0.0
    for idx in range(anchor_index, late_start, -1):
        sigma_hi = float(sigmas[idx - 1].item())
        sigma_lo = float(sigmas[idx].item())
        restart_from = idx - 1
        if sigma_hi > 0.0 and sigma_lo > 0.0:
            accumulated_span += abs(math.log(sigma_hi) - math.log(sigma_lo))
        if (anchor_index - restart_from) >= 1 and accumulated_span >= target_log_span:
            break

    restart_from = max(late_start, restart_from)
    return sigmas[restart_from:anchor_index + 1], anchor_index


def _expand_restart_sigmas(sigmas, high_quality_pct, restart_steps, restart_repeats, restart_trigger_pct):
    if sigmas is None or not isinstance(sigmas, torch.Tensor):
        return None, None

    expanded_sigmas = sigmas.detach().clone()
    total_steps = max(0, int(expanded_sigmas.shape[0]) - 1)
    if total_steps <= 0:
        return expanded_sigmas, None

    high_quality_pct = max(0.0, min(1.0, float(high_quality_pct)))
    restart_steps = max(2, int(restart_steps))
    restart_repeats = max(0, int(restart_repeats))
    restart_trigger_pct = max(0.0, min(1.0, float(restart_trigger_pct)))

    late_steps = max(1, int(math.ceil(total_steps * high_quality_pct))) if high_quality_pct > 0.0 else 0
    late_start = max(0, total_steps - late_steps)

    quality_sigma_start = None
    if late_start < expanded_sigmas.shape[0]:
        quality_sigma_start = float(expanded_sigmas[late_start].item())

    if restart_repeats <= 0:
        return expanded_sigmas, quality_sigma_start

    restart_sigmas, anchor_index = _build_restart_sigmas(
        expanded_sigmas,
        late_start,
        total_steps,
        restart_trigger_pct,
        restart_steps,
    )
    if restart_sigmas.numel() <= 1:
        return expanded_sigmas, quality_sigma_start

    prefix = expanded_sigmas[:anchor_index + 1]
    suffix = expanded_sigmas[anchor_index + 1:]
    repeated = [restart_sigmas.clone() for _ in range(restart_repeats)]
    expanded_sigmas = torch.cat([prefix] + repeated + [suffix], dim=0)
    return expanded_sigmas, quality_sigma_start


def sample_funpack_hybrid_euler_2s(model, x, sigmas, extra_args=None, callback=None,
                                   disable=None, eta=1.0, s_noise=1.0,
                                   high_quality_pct=0.35, correction_blend=1.0,
                                   restart_steps=3, restart_repeats=0,
                                   restart_trigger_pct=0.85, restart_noise=1.0,
                                   quality_sigma_start=None):
    """
    Hybrid sampler:
    - Early schedule: Euler ancestral for motion/anatomy buildup.
    - Late schedule: deterministic Euler / DPM-Solver++(2S) ODE refinement for detail.
    - Restart: paper-style re-noise and short ODE replay inside the late quality window.
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
    restart_steps = max(2, int(restart_steps))
    restart_repeats = max(0, int(restart_repeats))
    restart_trigger_pct = max(0.0, min(1.0, float(restart_trigger_pct)))
    restart_noise = max(0.0, float(restart_noise))
    s_in = x.new_ones([x.shape[0]])
    callback_step = 0

    if quality_sigma_start is None:
        late_steps = max(1, int(math.ceil(total_steps * high_quality_pct))) if high_quality_pct > 0.0 else 0
        late_start = max(0, total_steps - late_steps)
        if late_start < sigmas.shape[0]:
            quality_sigma_start = float(sigmas[late_start].item())
    else:
        quality_sigma_start = float(quality_sigma_start)

    for i in comfy.utils.model_trange(total_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        is_restart_jump = sigma_next > sigma
        in_quality_phase = quality_sigma_start is not None and float(sigma.item()) <= quality_sigma_start

        if is_restart_jump:
            denoised = None
            if callback is not None:
                denoised = model(x, sigma * s_in, **extra_args)
                callback({
                    "x": x,
                    "i": callback_step,
                    "sigma": sigma,
                    "sigma_hat": sigma,
                    "denoised": denoised,
                })
                callback_step += 1
            x = _renoise_to_sigma(x, sigma, sigma_next, restart_noise, noise_sampler)
            continue

        if not in_quality_phase:
            denoised = model(x, sigma * s_in, **extra_args)

            if callback is not None:
                callback({
                    "x": x,
                    "i": callback_step,
                    "sigma": sigma,
                    "sigma_hat": sigma,
                    "denoised": denoised,
                })
                callback_step += 1

            sigma_down, sigma_up = k_diffusion_sampling.get_ancestral_step(sigma, sigma_next, eta=eta)

            if sigma_down == 0:
                x = denoised
            else:
                d = k_diffusion_sampling.to_d(x, sigma, denoised)
                dt = sigma_down - sigma
                x = x + d * dt

                if sigma_next > 0 and eta > 0 and s_noise > 0:
                    x = x + noise_sampler(sigma, sigma_next) * s_noise * sigma_up
        else:
            denoised = model(x, sigma * s_in, **extra_args)

            if callback is not None:
                callback({
                    "x": x,
                    "i": callback_step,
                    "sigma": sigma,
                    "sigma_hat": sigma,
                    "denoised": denoised,
                })
                callback_step += 1

            x, _ = _hybrid_ode_step(
                model,
                x,
                sigma,
                sigma_next,
                s_in,
                extra_args,
                correction_blend,
                denoised=denoised,
            )

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
                "restart_steps": ("INT", {
                    "default": 3,
                    "min": 2,
                    "max": 32,
                    "step": 1,
                    "tooltip": "How many previous late-phase sigma steps are revisited during Restart replay."
                }),
                "restart_repeats": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8,
                    "step": 1,
                    "tooltip": "How many paper-style re-noise/replay loops to run in the late quality phase. Set to 0 to disable Restart."
                }),
                "restart_trigger_pct": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Sampling progress point where Restart is triggered. Clamped into the late quality phase."
                }),
                "restart_noise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Noise strength used when re-noising up to the Restart interval's higher sigma."
                }),
            },
            "optional": {
                "sigmas": ("SIGMAS",),
            }
        }

    RETURN_TYPES = ("SAMPLER", "SIGMAS")
    RETURN_NAMES = ("sampler", "sigmas")
    FUNCTION = "get_sampler"
    CATEGORY = "FunPack/Sampling"
    DESCRIPTION = "Hybrid sampler: early Euler ancestral for motion, late DPM-Solver++(2S) ODE refinement for quality, with optional sigma-driven Restart replay."

    def get_sampler(self, eta, s_noise, high_quality_pct, correction_blend,
                    restart_steps, restart_repeats, restart_trigger_pct, restart_noise, sigmas=None):
        expanded_sigmas, quality_sigma_start = _expand_restart_sigmas(
            sigmas,
            high_quality_pct,
            restart_steps,
            restart_repeats,
            restart_trigger_pct,
        )
        sampler = comfy.samplers.KSAMPLER(
            sample_funpack_hybrid_euler_2s,
            extra_options={
                "eta": eta,
                "s_noise": s_noise,
                "high_quality_pct": high_quality_pct,
                "correction_blend": correction_blend,
                "restart_steps": restart_steps,
                "restart_repeats": restart_repeats,
                "restart_trigger_pct": restart_trigger_pct,
                "restart_noise": restart_noise,
                "quality_sigma_start": quality_sigma_start,
            }
        )
        return (sampler, expanded_sigmas)
