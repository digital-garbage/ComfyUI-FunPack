import math

import torch

import comfy.k_diffusion.sampling as k_diffusion_sampling
import comfy.model_sampling
import comfy.samplers
import comfy.utils


TRANSITION_MODES = ["off", "balanced", "aggressive", "custom"]


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


def _apply_motion_pulse(x, sigma, sigma_next, pulse_noise, noise_sampler):
    if pulse_noise <= 0.0 or sigma_next >= sigma:
        return x

    sigma_delta_sq = max(0.0, float(sigma * sigma - sigma_next * sigma_next))
    if sigma_delta_sq <= 0.0:
        return x

    sigma_delta = math.sqrt(sigma_delta_sq)
    return x + noise_sampler(sigma, sigma_next) * (pulse_noise * sigma_delta)


def _find_schedule_anchor_index(sigmas, total_steps, schedule_progress):
    if sigmas is None or total_steps <= 1:
        return 0

    schedule_progress = max(0.0, min(1.0, schedule_progress))
    return min(total_steps - 1, max(0, int(round(schedule_progress * max(0, total_steps - 1)))))


def _resolve_transition_options(transition_mode, transition_start_pct,
                                transition_count, transition_spacing_pct,
                                transition_strength):
    mode = (transition_mode or "off").lower()
    if mode not in TRANSITION_MODES:
        mode = "off"

    if mode == "off":
        return {
            "enabled": False,
            "start_pct": 0.30,
            "count": 0,
            "spacing_pct": 0.22,
            "strength": 0.0,
            "noise": 0.0,
        }

    start_pct = 0.30 if transition_start_pct is None else float(transition_start_pct)
    spacing_pct = 0.22 if transition_spacing_pct is None else float(transition_spacing_pct)
    strength = 0.85 if transition_strength is None else float(transition_strength)
    count = 2 if transition_count is None else int(transition_count)

    start_pct = max(0.02, min(0.90, start_pct))
    spacing_pct = max(0.04, min(0.45, spacing_pct))
    strength = max(0.0, min(1.0, strength))
    count = max(1, min(6, count))

    if mode == "balanced":
        count = min(count, 1)
        strength = 0.55 if transition_strength is None else min(strength, 0.70)
    elif mode == "aggressive":
        count = max(2, count)
        strength = max(strength, 0.85)

    noise = 0.10 + strength * 0.55

    return {
        "enabled": True,
        "start_pct": start_pct,
        "count": count,
        "spacing_pct": spacing_pct,
        "strength": strength,
        "noise": max(0.0, min(0.80, noise)),
    }


def _get_late_start_index(total_steps, high_quality_pct):
    high_quality_pct = max(0.0, min(1.0, float(high_quality_pct)))
    late_steps = max(1, int(math.ceil(total_steps * high_quality_pct))) if high_quality_pct > 0.0 else 0
    return max(0, total_steps - late_steps)


def _build_transition_pulse_steps(sigmas, total_steps, high_quality_pct,
                                  transition_mode, transition_start_pct,
                                  transition_count, transition_spacing_pct,
                                  transition_strength):
    options = _resolve_transition_options(
        transition_mode,
        transition_start_pct,
        transition_count,
        transition_spacing_pct,
        transition_strength,
    )
    if not options["enabled"] or total_steps <= 2:
        return [], options

    late_start = _get_late_start_index(total_steps, high_quality_pct)
    latest_anchor_pct = max(0.04, min(0.92, (late_start - 1) / max(1, total_steps - 1)))
    pulse_steps = []
    used_anchors = set()

    for pulse_index in range(options["count"]):
        trigger_pct = options["start_pct"] + options["spacing_pct"] * pulse_index
        if trigger_pct >= latest_anchor_pct:
            break

        anchor_index = _find_schedule_anchor_index(sigmas, total_steps, trigger_pct)
        anchor_index = min(total_steps - 1, max(0, anchor_index))
        if anchor_index in used_anchors or anchor_index >= late_start:
            continue

        used_anchors.add(anchor_index)
        pulse_steps.append({
            "step_index": anchor_index,
            "noise": options["noise"],
        })

    return pulse_steps, options


def _prepare_dynamic_sigmas(sigmas, high_quality_pct, transition_mode="off",
                           transition_start_pct=0.30, transition_count=2,
                           transition_spacing_pct=0.22, transition_strength=0.85):
    if sigmas is None or not isinstance(sigmas, torch.Tensor):
        return None, None, [], 0.0

    base_sigmas = sigmas.detach().clone()
    total_steps = max(0, int(base_sigmas.shape[0]) - 1)
    if total_steps <= 0:
        return base_sigmas, None, [], 0.0

    late_start = _get_late_start_index(total_steps, high_quality_pct)
    quality_sigma_start = None
    if late_start < base_sigmas.shape[0]:
        quality_sigma_start = float(base_sigmas[late_start].item())

    pulse_steps, transition_options = _build_transition_pulse_steps(
        base_sigmas,
        total_steps,
        high_quality_pct,
        transition_mode,
        transition_start_pct,
        transition_count,
        transition_spacing_pct,
        transition_strength,
    )
    return base_sigmas, quality_sigma_start, pulse_steps, transition_options["noise"]


def sample_funpack_hybrid_euler_2s(model, x, sigmas, extra_args=None, callback=None,
                                   disable=None, eta=1.0, s_noise=1.0,
                                   high_quality_pct=0.35, correction_blend=1.0,
                                   quality_sigma_start=None,
                                   transition_mode="off",
                                   transition_start_pct=0.30,
                                   transition_count=2,
                                   transition_spacing_pct=0.22,
                                   transition_strength=0.85,
                                   transition_noise=0.0,
                                   transition_pulse_steps=None):
    """
    Hybrid sampler:
    - Early schedule: Euler ancestral for motion/anatomy buildup.
    - Late schedule: deterministic Euler / DPM-Solver++(2S) ODE refinement for detail.
    - Motion pulses: optional monotonic noise kicks before the late quality phase.
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
    if not transition_pulse_steps:
        _, _, transition_pulse_steps, computed_transition_noise = _prepare_dynamic_sigmas(
            sigmas,
            high_quality_pct,
            transition_mode,
            transition_start_pct,
            transition_count,
            transition_spacing_pct,
            transition_strength,
        )
        transition_noise = computed_transition_noise
    transition_noise = max(0.0, float(transition_noise))
    transition_step_noise = {
        int(item.get("step_index", -1)): max(0.0, float(item.get("noise", transition_noise)))
        for item in (transition_pulse_steps or [])
        if isinstance(item, dict)
    }
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
        in_quality_phase = quality_sigma_start is not None and float(sigma.item()) <= quality_sigma_start

        if not in_quality_phase:
            pulse_noise = transition_step_noise.get(int(i), 0.0)
            if pulse_noise > 0.0:
                x = _apply_motion_pulse(x, sigma, sigma_next, pulse_noise, noise_sampler)

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
                    "tooltip": "Legacy Restart control. Ignored because Restart replay is disabled for LTX audio safety."
                }),
                "restart_repeats": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Legacy Restart control. Ignored because Restart replay is disabled for LTX audio safety."
                }),
                "restart_trigger_pct": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Legacy Restart control. Ignored because Restart replay is disabled for LTX audio safety."
                }),
                "restart_noise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Legacy Restart control. Ignored because Restart replay is disabled for LTX audio safety."
                }),
                "transition_mode": (TRANSITION_MODES, {
                    "default": "off",
                    "tooltip": "Adds early/mid motion pulses for single-clip image-to-video. Off preserves legacy sampler behavior."
                }),
                "transition_start_pct": ("FLOAT", {
                    "default": 0.30,
                    "min": 0.0,
                    "max": 0.95,
                    "step": 0.01,
                    "tooltip": "Sampling progress where the first motion pulse starts."
                }),
                "transition_count": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 6,
                    "step": 1,
                    "tooltip": "How many early/mid motion pulses to request before the late quality phase."
                }),
                "transition_spacing_pct": ("FLOAT", {
                    "default": 0.22,
                    "min": 0.04,
                    "max": 0.45,
                    "step": 0.01,
                    "tooltip": "Progress spacing between motion pulses."
                }),
                "transition_strength": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "How strongly motion pulses add monotonic noise kicks. Higher values push harder against stale image references."
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
    DESCRIPTION = "Hybrid sampler: early Euler ancestral for motion, late DPM-Solver++(2S) ODE refinement for quality, with optional audio-safe motion pulses."

    def get_sampler(self, eta, s_noise, high_quality_pct, correction_blend,
                    restart_steps, restart_repeats, restart_trigger_pct, restart_noise,
                    transition_mode="off", transition_start_pct=0.30,
                    transition_count=2, transition_spacing_pct=0.22,
                    transition_strength=0.85, sigmas=None):
        prepared_sigmas, quality_sigma_start, transition_pulse_steps, transition_noise = _prepare_dynamic_sigmas(
            sigmas,
            high_quality_pct,
            transition_mode,
            transition_start_pct,
            transition_count,
            transition_spacing_pct,
            transition_strength,
        )
        sampler = comfy.samplers.KSAMPLER(
            sample_funpack_hybrid_euler_2s,
            extra_options={
                "eta": eta,
                "s_noise": s_noise,
                "high_quality_pct": high_quality_pct,
                "correction_blend": correction_blend,
                "quality_sigma_start": quality_sigma_start,
                "transition_mode": transition_mode,
                "transition_start_pct": transition_start_pct,
                "transition_count": transition_count,
                "transition_spacing_pct": transition_spacing_pct,
                "transition_strength": transition_strength,
                "transition_noise": transition_noise,
                "transition_pulse_steps": transition_pulse_steps,
            }
        )
        return (sampler, prepared_sigmas)
