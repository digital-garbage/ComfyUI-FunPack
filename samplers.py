import math

import torch

import comfy.k_diffusion.sampling as k_diffusion_sampling
import comfy.model_sampling
import comfy.samplers
import comfy.utils


MOTION_PULSE_MODES = ["off", "balanced", "aggressive", "custom"]
VELOCITY_BIAS_MODES = ["off", "capture", "apply", "capture_and_apply"]
VELOCITY_BIAS_TARGETS = (0.90, 0.80)
VELOCITY_BIAS_MEMORY = {}


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


def _velocity_bias_enabled(mode, action):
    mode = (mode or "off").lower()
    if mode == "capture_and_apply":
        return action in {"capture", "apply"}
    return mode == action


def _velocity_bias_key(refinement_key, aspect_bucket, target, x):
    shape = tuple(int(item) for item in getattr(x, "shape", ()))
    key = str(refinement_key or "default").strip() or "default"
    aspect = str(aspect_bucket or "any").strip() or "any"
    return (key, aspect, f"{float(target):.2f}", shape)


def _sigma_ratio(sigmas, sigma):
    try:
        start = float(sigmas[0].item())
        current = float(sigma.item())
    except Exception:
        return None
    if start <= 0.0:
        return None
    return current / start


def _velocity_bias_target(sigmas, sigma):
    ratio = _sigma_ratio(sigmas, sigma)
    if ratio is None:
        return None
    target = min(VELOCITY_BIAS_TARGETS, key=lambda item: abs(float(item) - ratio))
    return target if abs(float(target) - ratio) <= 0.065 else None


def _capture_velocity_bias(refinement_key, aspect_bucket, target, x, sigma, denoised):
    if target is None:
        return
    try:
        direction = k_diffusion_sampling.to_d(x, sigma, denoised).detach().float().cpu()
    except Exception:
        return
    key = _velocity_bias_key(refinement_key, aspect_bucket, target, x)
    slot = VELOCITY_BIAS_MEMORY.setdefault(key, {"count": 0, "direction": None})
    count = int(slot.get("count", 0))
    previous = slot.get("direction")
    if count <= 0 or not isinstance(previous, torch.Tensor) or tuple(previous.shape) != tuple(direction.shape):
        slot["direction"] = direction
        slot["count"] = 1
        return
    slot["direction"] = (previous * count + direction) / float(count + 1)
    slot["count"] = min(count + 1, 256)


def _apply_velocity_bias(x, refinement_key, aspect_bucket, target, strength):
    if target is None or strength <= 0.0:
        return x
    key = _velocity_bias_key(refinement_key, aspect_bucket, target, x)
    slot = VELOCITY_BIAS_MEMORY.get(key)
    if not isinstance(slot, dict) or not isinstance(slot.get("direction"), torch.Tensor):
        return x
    direction = slot["direction"]
    if tuple(direction.shape) != tuple(x.shape):
        return x
    try:
        direction = direction.to(device=x.device, dtype=x.dtype)
        delta = direction * max(0.0, min(0.35, float(strength)))
        max_delta = x.detach().float().norm().clamp_min(1e-8) * 0.045
        delta_norm = delta.detach().float().norm().clamp_min(1e-8)
        if delta_norm > max_delta:
            delta = delta * (max_delta / delta_norm).to(device=x.device, dtype=x.dtype)
        return x + delta
    except Exception:
        return x


def _find_schedule_anchor_index(sigmas, total_steps, schedule_progress):
    if sigmas is None or total_steps <= 1:
        return 0

    schedule_progress = max(0.0, min(1.0, schedule_progress))
    return min(total_steps - 1, max(0, int(round(schedule_progress * max(0, total_steps - 1)))))


def _resolve_motion_pulse_options(motion_pulse_mode, motion_pulse_start_pct,
                                  motion_pulse_count, motion_pulse_spacing_pct,
                                  motion_pulse_strength):
    mode = (motion_pulse_mode or "off").lower()
    if mode not in MOTION_PULSE_MODES:
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

    start_pct = 0.30 if motion_pulse_start_pct is None else float(motion_pulse_start_pct)
    spacing_pct = 0.22 if motion_pulse_spacing_pct is None else float(motion_pulse_spacing_pct)
    strength = 0.85 if motion_pulse_strength is None else float(motion_pulse_strength)
    count = 2 if motion_pulse_count is None else int(motion_pulse_count)

    start_pct = max(0.02, min(0.90, start_pct))
    spacing_pct = max(0.04, min(0.45, spacing_pct))
    strength = max(0.0, min(1.0, strength))
    count = max(1, min(6, count))

    if mode == "balanced":
        count = min(count, 1)
        strength = 0.55 if motion_pulse_strength is None else min(strength, 0.70)
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


def _build_motion_pulse_steps(sigmas, total_steps, high_quality_pct,
                              motion_pulse_mode, motion_pulse_start_pct,
                              motion_pulse_count, motion_pulse_spacing_pct,
                              motion_pulse_strength):
    options = _resolve_motion_pulse_options(
        motion_pulse_mode,
        motion_pulse_start_pct,
        motion_pulse_count,
        motion_pulse_spacing_pct,
        motion_pulse_strength,
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


def _prepare_dynamic_sigmas(sigmas, high_quality_pct, motion_pulse_mode="off",
                           motion_pulse_start_pct=0.30, motion_pulse_count=2,
                           motion_pulse_spacing_pct=0.22, motion_pulse_strength=0.85):
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

    pulse_steps, motion_pulse_options = _build_motion_pulse_steps(
        base_sigmas,
        total_steps,
        high_quality_pct,
        motion_pulse_mode,
        motion_pulse_start_pct,
        motion_pulse_count,
        motion_pulse_spacing_pct,
        motion_pulse_strength,
    )
    return base_sigmas, quality_sigma_start, pulse_steps, motion_pulse_options["noise"]


def _order2_ancestral_denoised(denoised, prev_denoised, h, prev_h):
    """
    Linear extrapolation of the denoised estimate using the previous step's value.
    Equivalent to the DPM-Solver++ 2M approach applied to the ancestral phase.
    Gives second-order accuracy at zero extra model-call cost.
    """
    if prev_denoised is None or prev_h is None or prev_h < 1e-7 or h < 1e-7:
        return denoised
    r = max(0.25, min(4.0, prev_h / h))
    c1 = 1.0 + 0.5 / r
    c2 = 0.5 / r
    try:
        extrap = c1 * denoised - c2 * prev_denoised.to(device=denoised.device, dtype=denoised.dtype)
        return extrap
    except Exception:
        return denoised


def sample_funpack_hybrid_euler_2s(model, x, sigmas, extra_args=None, callback=None,
                                   disable=None, eta=1.0, s_noise=1.0,
                                   high_quality_pct=0.35, correction_blend=1.0,
                                   quality_sigma_start=None,
                                   motion_pulse_mode="off",
                                   motion_pulse_start_pct=0.30,
                                   motion_pulse_count=2,
                                   motion_pulse_spacing_pct=0.22,
                                   motion_pulse_strength=0.85,
                                   motion_pulse_noise=0.0,
                                   motion_pulse_steps=None,
                                   velocity_bias_mode="off",
                                   velocity_bias_strength=0.0,
                                   velocity_refinement_key="default",
                                   velocity_aspect_bucket="any",
                                   eta_final=1.0):
    """
    Hybrid sampler:
    - Early schedule: Euler ancestral with order-2 denoised extrapolation for
      motion/anatomy buildup. Order-2 reuses the previous step's denoised to
      extrapolate the score direction, giving DPM-Solver++ 2M accuracy at zero
      extra model-call cost.
    - Late schedule: deterministic DPM-Solver++(2S) ODE refinement for detail,
      with progressive correction_blend — first half of quality steps use single-
      eval Euler ODE, second half use the full configured 2S correction. This
      cuts quality-phase model calls by roughly half while preserving the 2S
      benefit where sigma is lowest and it matters most.
    - Eta decay: when eta_final < eta, ancestral noise strength decays toward
      eta_final as sigma approaches the quality boundary, giving a cleaner
      transition into deterministic refinement.
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
    eta_final = max(0.0, min(float(eta), float(eta_final)))
    velocity_bias_mode = (velocity_bias_mode or "off").lower()
    if velocity_bias_mode not in VELOCITY_BIAS_MODES:
        velocity_bias_mode = "off"
    velocity_bias_strength = max(0.0, min(0.35, float(velocity_bias_strength or 0.0)))

    if not motion_pulse_steps:
        _, _, motion_pulse_steps, computed_motion_pulse_noise = _prepare_dynamic_sigmas(
            sigmas,
            high_quality_pct,
            motion_pulse_mode,
            motion_pulse_start_pct,
            motion_pulse_count,
            motion_pulse_spacing_pct,
            motion_pulse_strength,
        )
        motion_pulse_noise = computed_motion_pulse_noise
    motion_pulse_noise = max(0.0, float(motion_pulse_noise))
    motion_step_noise = {
        int(item.get("step_index", -1)): max(0.0, float(item.get("noise", motion_pulse_noise)))
        for item in (motion_pulse_steps or [])
        if isinstance(item, dict)
    }

    s_in = x.new_ones([x.shape[0]])
    callback_step = 0

    # Resolve quality phase boundary
    late_start = _get_late_start_index(total_steps, high_quality_pct)
    if quality_sigma_start is None:
        if late_start < sigmas.shape[0]:
            quality_sigma_start = float(sigmas[late_start].item())
    else:
        quality_sigma_start = float(quality_sigma_start)

    num_quality_steps = total_steps - late_start

    # Order-2 ancestral state
    prev_denoised = None
    prev_h = None
    quality_step_index = 0

    for i in comfy.utils.model_trange(total_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        in_quality_phase = quality_sigma_start is not None and float(sigma.item()) <= quality_sigma_start

        if not in_quality_phase:
            # Adaptive eta: decay from eta toward eta_final as sigma
            # approaches the quality boundary.
            if quality_sigma_start is not None and quality_sigma_start > 0.0 and eta_final < eta:
                sigma_val = float(sigma.item())
                proximity = min(1.0, max(0.0, quality_sigma_start / max(sigma_val, 1e-8)))
                effective_eta = eta_final + (eta - eta_final) * (1.0 - proximity)
            else:
                effective_eta = eta

            pulse_noise = motion_step_noise.get(int(i), 0.0)
            if pulse_noise > 0.0:
                x = _apply_motion_pulse(x, sigma, sigma_next, pulse_noise, noise_sampler)
                # Pulse modifies x; previous denoised is no longer a valid
                # second-order estimate for the next step.
                prev_denoised = None
                prev_h = None

            velocity_target = _velocity_bias_target(sigmas, sigma)
            if _velocity_bias_enabled(velocity_bias_mode, "apply"):
                x = _apply_velocity_bias(x, velocity_refinement_key, velocity_aspect_bucket, velocity_target, velocity_bias_strength)
            denoised = model(x, sigma * s_in, **extra_args)
            if _velocity_bias_enabled(velocity_bias_mode, "capture"):
                _capture_velocity_bias(velocity_refinement_key, velocity_aspect_bucket, velocity_target, x, sigma, denoised)

            if callback is not None:
                callback({
                    "x": x,
                    "i": callback_step,
                    "sigma": sigma,
                    "sigma_hat": sigma,
                    "denoised": denoised,
                })
                callback_step += 1

            h = float((sigma - sigma_next).abs().item())
            denoised_eff = _order2_ancestral_denoised(denoised, prev_denoised, h, prev_h)

            sigma_down, sigma_up = k_diffusion_sampling.get_ancestral_step(sigma, sigma_next, eta=effective_eta)
            if sigma_down == 0:
                x = denoised_eff
            else:
                d = k_diffusion_sampling.to_d(x, sigma, denoised_eff)
                dt = sigma_down - sigma
                x = x + d * dt
                if sigma_next > 0 and effective_eta > 0 and s_noise > 0:
                    x = x + noise_sampler(sigma, sigma_next) * s_noise * sigma_up

            prev_denoised = denoised.detach()
            prev_h = h

        else:
            # Quality phase: progressive correction_blend.
            # First half of quality steps use blend=0 (single-eval Euler ODE),
            # second half use the configured blend (full 2S correction).
            # 2S matters most at the lowest sigmas, so this concentrates the
            # expensive second model call where it has the most impact.
            if num_quality_steps <= 1:
                effective_blend = correction_blend
            else:
                mid_quality = num_quality_steps // 2
                effective_blend = 0.0 if quality_step_index < mid_quality else correction_blend

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
                effective_blend,
                denoised=denoised,
            )
            quality_step_index += 1

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
                    "tooltip": "Ancestral stochasticity at the start of sampling. Keep at 1.0 for classic ancestral behaviour."
                }),
                "eta_final": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Eta value at the quality phase boundary. When below eta, ancestral noise decays linearly toward this value as sigma approaches the quality phase. Lower values give a cleaner hand-off into deterministic refinement. Set equal to eta to disable decay."
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
                    "tooltip": "Fraction of late denoise steps that enter the quality phase. The first half of quality steps use single-eval Euler ODE; the second half use the full 2S correction."
                }),
                "correction_blend": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Blend between Euler ODE (0.0) and 2S correction (1.0) for the second half of quality-phase steps."
                }),
                "motion_pulse_mode": (MOTION_PULSE_MODES, {
                    "default": "off",
                    "tooltip": "Adds early/mid anti-stiffness motion pulses. Off preserves legacy sampler behavior."
                }),
                "motion_pulse_start_pct": ("FLOAT", {
                    "default": 0.30,
                    "min": 0.0,
                    "max": 0.95,
                    "step": 0.01,
                    "tooltip": "Sampling progress where the first anti-stiffness pulse is applied."
                }),
                "motion_pulse_count": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 6,
                    "step": 1,
                    "tooltip": "How many early/mid motion pulses to request before the late quality phase."
                }),
                "motion_pulse_spacing_pct": ("FLOAT", {
                    "default": 0.22,
                    "min": 0.04,
                    "max": 0.45,
                    "step": 0.01,
                    "tooltip": "Progress spacing between motion pulses."
                }),
                "motion_pulse_strength": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "How strongly motion pulses add monotonic noise kicks. Higher values push harder against stale image references."
                }),
                "velocity_bias_mode": (VELOCITY_BIAS_MODES, {
                    "default": "off",
                    "tooltip": "Experimental: capture/apply averaged early model velocity around normalized sigma 0.9 and 0.8. Off preserves legacy sampler behavior."
                }),
                "velocity_bias_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.35,
                    "step": 0.01,
                    "tooltip": "Experimental strength for applying captured early velocity bias. Keep low; 0 disables the applied delta."
                }),
                "velocity_refinement_key": ("STRING", {
                    "default": "default",
                    "multiline": False,
                    "tooltip": "Memory key used to capture/apply early velocity bias."
                }),
                "velocity_aspect_bucket": ("STRING", {
                    "default": "any",
                    "multiline": False,
                    "tooltip": "Optional aspect bucket such as landscape, portrait, square, ultrawide, or vertical."
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
    DESCRIPTION = (
        "Hybrid sampler: early Euler ancestral with order-2 denoised extrapolation for motion, "
        "late DPM-Solver++(2S) ODE for quality with progressive correction blending. "
        "Optional eta decay, anti-stiffness motion pulses, and experimental velocity bias."
    )

    def get_sampler(self, eta, s_noise, high_quality_pct, correction_blend,
                    motion_pulse_mode="off", motion_pulse_start_pct=0.30,
                    motion_pulse_count=2, motion_pulse_spacing_pct=0.22,
                    motion_pulse_strength=0.85, velocity_bias_mode="off",
                    velocity_bias_strength=0.0, velocity_refinement_key="default",
                    velocity_aspect_bucket="any", sigmas=None, eta_final=1.0):
        prepared_sigmas, quality_sigma_start, motion_pulse_steps, motion_pulse_noise = _prepare_dynamic_sigmas(
            sigmas,
            high_quality_pct,
            motion_pulse_mode,
            motion_pulse_start_pct,
            motion_pulse_count,
            motion_pulse_spacing_pct,
            motion_pulse_strength,
        )
        sampler = comfy.samplers.KSAMPLER(
            sample_funpack_hybrid_euler_2s,
            extra_options={
                "eta": eta,
                "s_noise": s_noise,
                "high_quality_pct": high_quality_pct,
                "correction_blend": correction_blend,
                "quality_sigma_start": quality_sigma_start,
                "motion_pulse_mode": motion_pulse_mode,
                "motion_pulse_start_pct": motion_pulse_start_pct,
                "motion_pulse_count": motion_pulse_count,
                "motion_pulse_spacing_pct": motion_pulse_spacing_pct,
                "motion_pulse_strength": motion_pulse_strength,
                "motion_pulse_noise": motion_pulse_noise,
                "motion_pulse_steps": motion_pulse_steps,
                "velocity_bias_mode": velocity_bias_mode,
                "velocity_bias_strength": velocity_bias_strength,
                "velocity_refinement_key": velocity_refinement_key,
                "velocity_aspect_bucket": velocity_aspect_bucket,
                "eta_final": eta_final,
            }
        )
        return (sampler, prepared_sigmas)
