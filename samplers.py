import math

import torch

import comfy.k_diffusion.sampling as k_diffusion_sampling
import comfy.model_sampling
import comfy.nested_tensor
import comfy.sample
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


def sample_funpack_distilled_flow(model, x, sigmas, extra_args=None, callback=None,
                                   disable=None, order=2, s_noise=0.0,
                                   final_correction_steps=1):
    """
    ODE sampler for distilled few-step video models (e.g. LTX2.3 distilled LoRA).

    - Adams-Bashforth 2-step multistep (order=2): extrapolates the denoised
      direction from two consecutive steps for second-order accuracy at zero
      extra model-call cost. Reduces discretisation error across the large
      sigma jumps typical of 4–8 step distilled schedules.
    - Heun predictor-corrector on final steps: calls the model a second time
      at sigma_next to correct the update direction. Significantly improves
      sharpness and detail in the steps that define the final output.
    - Optional s_noise: tiny ancestral-style noise injection for diversity.
      Default 0 = fully deterministic ODE (recommended for distilled models).
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = k_diffusion_sampling.default_noise_sampler(x, seed=seed)

    total_steps = max(0, len(sigmas) - 1)
    if total_steps <= 0:
        return x

    order = max(1, min(2, int(order)))
    s_noise = max(0.0, min(0.5, float(s_noise)))
    final_correction_steps = max(0, min(total_steps // 2, int(final_correction_steps)))
    correction_start_idx = total_steps - final_correction_steps

    s_in = x.new_ones([x.shape[0]])
    prev_denoised = None
    prev_h = None

    for i in comfy.utils.model_trange(total_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = model(x, sigma * s_in, **extra_args)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})

        h = float((sigma - sigma_next).abs().item())

        # Adams-Bashforth 2-step multistep correction.
        # Coefficients for variable step sizes: r = h_current / h_previous.
        # denoised_eff = (1 + r/2) * denoised - (r/2) * prev_denoised
        if order >= 2 and prev_denoised is not None and prev_h is not None and prev_h > 1e-7 and h > 1e-7:
            r = max(0.1, min(5.0, h / prev_h))
            try:
                denoised_eff = (1.0 + r / 2.0) * denoised - (r / 2.0) * prev_denoised.to(device=denoised.device, dtype=denoised.dtype)
            except Exception:
                denoised_eff = denoised
        else:
            denoised_eff = denoised

        # Store current denoised for the next step's multistep correction.
        # Reset after a Heun step since x was updated with a corrected direction.
        prev_denoised = denoised.detach()
        prev_h = h

        if sigma_next == 0:
            x = denoised_eff
            continue

        dt = sigma_next - sigma  # negative: sigmas decrease

        if i >= correction_start_idx:
            # Heun predictor-corrector.
            # Predictor: Euler step using the (multistep-corrected) denoised.
            d1 = k_diffusion_sampling.to_d(x, sigma, denoised_eff)
            x_pred = x + d1 * dt
            # Corrector: evaluate model at the predicted x and sigma_next.
            denoised_pred = model(x_pred, sigma_next * s_in, **extra_args)
            d2 = k_diffusion_sampling.to_d(x_pred, sigma_next, denoised_pred)
            x = x + (d1 + d2) / 2.0 * dt
            # Heun updates x differently; invalidate multistep history.
            prev_denoised = None
            prev_h = None
        else:
            d = k_diffusion_sampling.to_d(x, sigma, denoised_eff)
            x = x + d * dt
            if s_noise > 0.0:
                sigma_up = math.sqrt(max(0.0, float(sigma.item()) ** 2 - float(sigma_next.item()) ** 2))
                if sigma_up > 0.0:
                    x = x + noise_sampler(sigma, sigma_next) * s_noise * sigma_up

    return x


class FunPackDistilledFlowSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "order": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 2,
                    "step": 1,
                    "tooltip": "Multistep order. 1 = standard Euler ODE. 2 = Adams-Bashforth 2-step: extrapolates the denoised direction from two consecutive steps for better accuracy at no extra model-call cost.",
                }),
                "final_correction_steps": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 3,
                    "step": 1,
                    "tooltip": "Number of final steps that use a Heun predictor-corrector pass. Each costs one extra model call but significantly improves final-step detail. 1 is usually enough for 8-step runs.",
                }),
                "s_noise": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.50,
                    "step": 0.01,
                    "tooltip": "Optional stochastic noise for diversity. 0 = fully deterministic ODE (recommended). Small values (0.05–0.15) add variation without strongly disrupting the distilled trajectory.",
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
        "ODE sampler for distilled few-step video models (e.g. LTX2.3 distilled LoRA). "
        "Adams-Bashforth 2-step multistep for better trajectory accuracy across large sigma jumps, "
        "Heun predictor-corrector on final steps for quality, and optional controlled noise for diversity."
    )

    def get_sampler(self, order=2, final_correction_steps=1, s_noise=0.0, sigmas=None):
        prepared_sigmas = sigmas.detach().clone() if isinstance(sigmas, torch.Tensor) else sigmas
        sampler = comfy.samplers.KSAMPLER(
            sample_funpack_distilled_flow,
            extra_options={
                "order": order,
                "final_correction_steps": final_correction_steps,
                "s_noise": s_noise,
            }
        )
        return (sampler, prepared_sigmas)


class FunPackLTXAVSceneChainSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "latent_template": ("LATENT",),
                "num_frames_per_scene": ("INT", {"default": 97, "min": 1, "max": 4096, "step": 8}),
                "frame_overlap": ("INT", {"default": 16, "min": 0, "max": 512, "step": 8}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "max_scenes": ("INT", {"default": 8, "min": 1, "step": 1}),
                "carry_i2v_guides": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Experimental: carry protected frames from latent_template noise_mask into each continuation chunk after the overlap.",
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING", "INT", "STRING")
    RETURN_NAMES = ("latent", "status", "scene_count", "scene_report")
    FUNCTION = "sample"
    CATEGORY = "FunPack/Sampling"
    DESCRIPTION = (
        "Samples multi-entry scene conditioning as a smooth LTXV/LTXAV continuation chain. "
        "Use with FunPack Studio split-by-transitions output."
    )

    def _is_nested(self, samples):
        return bool(getattr(samples, "is_nested", False))

    def _clone_value(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if self._is_nested(value):
            return comfy.nested_tensor.NestedTensor([t.detach().clone() for t in value.unbind()])
        return value

    def _clone_latent(self, latent):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("latent_template must be a LATENT dict with samples.")
        return {key: self._clone_value(value) for key, value in latent.items()}

    def _tensor_frames(self, tensor):
        if not isinstance(tensor, torch.Tensor) or tensor.dim() < 3:
            raise ValueError("Scene chain latents must have a time dimension at index 2.")
        return int(tensor.shape[2])

    def _latent_tensors(self, latent):
        samples = latent.get("samples")
        if self._is_nested(samples):
            tensors = list(samples.unbind())
            if not tensors:
                raise ValueError("Nested latent has no tensors.")
            return tensors
        if isinstance(samples, torch.Tensor):
            return [samples]
        raise ValueError("Scene chain sampler requires tensor or nested tensor latent samples.")

    def _latent_masks(self, latent, count):
        masks = latent.get("noise_mask")
        if masks is None:
            return [None] * count
        if self._is_nested(masks):
            out = list(masks.unbind())
        else:
            out = [masks]
        while len(out) < count:
            out.append(None)
        return out[:count]

    def _time_scale(self, vae):
        scale = getattr(vae, "downscale_index_formula", None)
        if isinstance(scale, (list, tuple)) and scale:
            try:
                return max(1, int(scale[0]))
            except Exception:
                return 1
        return 1

    def _expected_latent_frames(self, pixel_frames, time_scale):
        return ((max(1, int(pixel_frames)) - 1) // max(1, int(time_scale))) + 1

    def _validate_template_length(self, latent_template, num_frames_per_scene, time_scale):
        video_frames = self._tensor_frames(self._latent_tensors(latent_template)[0])
        expected = self._expected_latent_frames(num_frames_per_scene, time_scale)
        if video_frames != expected:
            raise ValueError(
                f"latent_template has {video_frames} video latent frames, expected {expected} "
                f"from num_frames_per_scene={num_frames_per_scene} and time scale={time_scale}."
            )
        return video_frames

    def _overlap_frames(self, latent_template, frame_overlap, time_scale):
        video_frames = self._tensor_frames(self._latent_tensors(latent_template)[0])
        overlap = self._expected_latent_frames(frame_overlap + 1, time_scale) - 1
        if frame_overlap <= 0:
            overlap = 0
        if overlap >= video_frames:
            raise ValueError("frame_overlap must be smaller than the latent scene length.")
        return max(0, int(overlap))

    def _derived_overlap(self, video_overlap, video_frames, tensor_frames):
        if video_overlap <= 0:
            return 0
        ratio = tensor_frames / max(1, video_frames)
        overlap = int(round(video_overlap * ratio))
        return max(1, min(tensor_frames - 1, overlap))

    def _replace_start(self, target, source_tail, overlap):
        if overlap <= 0:
            return target
        target = target.clone()
        source_tail = source_tail.to(device=target.device, dtype=target.dtype)
        target[:, :, :overlap] = source_tail
        return target

    def _tail(self, tensor, overlap):
        if overlap <= 0:
            return tensor[:, :, :0]
        return tensor[:, :, -overlap:]

    def _time_slice(self, tensor, start, end):
        slices = [slice(None)] * tensor.dim()
        slices[2] = slice(start, end)
        return tensor[tuple(slices)]

    def _set_time_slice(self, tensor, start, end, value):
        slices = [slice(None)] * tensor.dim()
        slices[2] = slice(start, end)
        tensor[tuple(slices)] = value
        return tensor

    def _expand_mask_like(self, mask, target):
        if mask.shape[0] != target.shape[0] or mask.shape[2] != target.shape[2]:
            raise ValueError("Guide mask batch/time dimensions must match target mask.")
        shape = list(mask.shape)
        while len(shape) < target.dim():
            shape.append(1)
            mask = mask.reshape(shape)
        expand_shape = list(target.shape)
        for dim in range(target.dim()):
            if dim == 2:
                continue
            if mask.shape[dim] not in (1, target.shape[dim]):
                raise ValueError("Guide mask dimensions are not broadcastable to target mask.")
            expand_shape[dim] = target.shape[dim]
        return mask.expand(expand_shape)

    def _make_mask_tensor(self, tensor, overlap):
        mask = torch.ones_like(tensor)
        if overlap > 0:
            mask[:, :, :overlap] = 0
        return mask

    def _protected_prefix_frames(self, template_mask, tensor_frames):
        if template_mask is None or not isinstance(template_mask, torch.Tensor) or template_mask.dim() < 3:
            return 0
        dims = [dim for dim in range(template_mask.dim()) if dim != 2]
        per_frame = template_mask.float().mean(dim=dims).flatten()
        limit = min(int(tensor_frames), int(per_frame.numel()))
        count = 0
        for value in per_frame[:limit]:
            if float(value) >= 0.999:
                break
            count += 1
        return count

    def _build_continuation_chunk(self, template, previous, video_overlap):
        chunk = self._clone_latent(template)
        chunk_tensors = self._latent_tensors(chunk)
        previous_tensors = self._latent_tensors(previous)
        if len(chunk_tensors) != len(previous_tensors):
            raise ValueError("Previous output and latent_template must have the same latent structure.")

        video_frames = self._tensor_frames(chunk_tensors[0])
        out_tensors = []
        mask_tensors = []
        for index, tensor in enumerate(chunk_tensors):
            tensor_frames = self._tensor_frames(tensor)
            overlap = video_overlap if index == 0 else self._derived_overlap(video_overlap, video_frames, tensor_frames)
            prev_tail = self._tail(previous_tensors[index], overlap)
            out_tensor = self._replace_start(tensor, prev_tail, overlap)
            mask_tensor = self._make_mask_tensor(tensor, overlap)
            out_tensors.append(out_tensor)
            mask_tensors.append(mask_tensor)

        if self._is_nested(chunk.get("samples")):
            chunk["samples"] = comfy.nested_tensor.NestedTensor(out_tensors)
            chunk["noise_mask"] = comfy.nested_tensor.NestedTensor(mask_tensors)
        else:
            chunk["samples"] = out_tensors[0]
            chunk["noise_mask"] = mask_tensors[0]
        return chunk

    def _condition_with_values(self, conditioning, values):
        out = []
        for cond, meta in conditioning:
            new_meta = dict(meta) if isinstance(meta, dict) else {}
            for key, value in values.items():
                if value is None:
                    new_meta.pop(key, None)
                else:
                    new_meta[key] = value
            out.append((cond, new_meta))
        return out

    def _conditioning_value(self, conditioning, key):
        for item in conditioning or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], dict) and key in item[1]:
                return item[1][key]
        return None

    def _guide_keyframe_idxs(self, guiding_latent, scale_factors):
        try:
            from comfy.ldm.lightricks.symmetric_patchifier import SymmetricPatchifier, latent_to_pixel_coords
            patchifier = SymmetricPatchifier(1, start_end=True)
            _, latent_coords = patchifier.patchify(guiding_latent)
            return latent_to_pixel_coords(latent_coords, scale_factors, causal_fix=True)
        except Exception:
            b, _, f, h, w = guiding_latent.shape
            return torch.zeros((b, 3, f * h * w, 2), dtype=torch.float32, device=guiding_latent.device)

    def _append_guide_conditioning(self, conditioning, keyframe_idxs, guide_entry):
        existing_idxs = self._conditioning_value(conditioning, "keyframe_idxs")
        if isinstance(existing_idxs, torch.Tensor):
            keyframe_idxs = torch.cat([existing_idxs.to(keyframe_idxs.device, keyframe_idxs.dtype), keyframe_idxs], dim=2)
        existing_entries = self._conditioning_value(conditioning, "guide_attention_entries")
        entries = list(existing_entries or [])
        entries.append(guide_entry)
        return self._condition_with_values(conditioning, {
            "keyframe_idxs": keyframe_idxs,
            "guide_attention_entries": entries,
        })

    def _append_i2v_guides(self, chunk, template, positive, negative, vae):
        chunk_tensors = self._latent_tensors(chunk)
        template_tensors = self._latent_tensors(template)
        template_masks = self._latent_masks(template, len(template_tensors))
        if not chunk_tensors or not template_tensors:
            return chunk, positive, negative, 0

        video_mask = template_masks[0]
        protected = self._protected_prefix_frames(video_mask, self._tensor_frames(template_tensors[0]))
        if protected <= 0:
            return chunk, positive, negative, 0

        guide = self._time_slice(template_tensors[0], 0, protected).to(
            device=chunk_tensors[0].device, dtype=chunk_tensors[0].dtype,
        )
        guide_mask = self._time_slice(video_mask, 0, protected).to(
            device=chunk_tensors[0].device, dtype=chunk_tensors[0].dtype,
        )

        out_tensors = list(chunk_tensors)
        out_masks = self._latent_masks(chunk, len(out_tensors))
        if out_masks[0] is None:
            out_masks[0] = torch.ones_like(out_tensors[0])
        out_tensors[0] = torch.cat([out_tensors[0], guide], dim=2)
        target_mask = self._time_slice(out_masks[0], 0, protected).to(guide_mask.device, guide_mask.dtype)
        guide_mask = self._expand_mask_like(guide_mask, target_mask)
        out_masks[0] = torch.cat([out_masks[0].to(guide_mask.device, guide_mask.dtype), guide_mask], dim=2)

        if self._is_nested(chunk.get("samples")):
            chunk["samples"] = comfy.nested_tensor.NestedTensor(out_tensors)
            chunk["noise_mask"] = comfy.nested_tensor.NestedTensor(out_masks)
        else:
            chunk["samples"] = out_tensors[0]
            chunk["noise_mask"] = out_masks[0]

        scale_factors = getattr(vae, "downscale_index_formula", (1, 1, 1))
        keyframe_idxs = self._guide_keyframe_idxs(guide, scale_factors)
        guide_strength = max(0.0, min(1.0, 1.0 - float(guide_mask.float().mean().item())))
        guide_entry = {
            "pre_filter_count": guide.shape[2] * guide.shape[3] * guide.shape[4],
            "strength": guide_strength,
            "pixel_mask": None,
            "latent_shape": list(guide.shape[2:]),
        }
        return (
            chunk,
            self._append_guide_conditioning(positive, keyframe_idxs, guide_entry),
            self._append_guide_conditioning(negative, keyframe_idxs, guide_entry),
            protected,
        )

    def _crop_video_tail(self, latent, count):
        if count <= 0:
            return latent
        result = self._clone_latent(latent)
        tensors = self._latent_tensors(result)
        tensors[0] = tensors[0][:, :, :-count]
        if self._is_nested(result.get("samples")):
            result["samples"] = comfy.nested_tensor.NestedTensor(tensors)
        else:
            result["samples"] = tensors[0]
        return result

    def _blend_tensors(self, left, right, overlap):
        if overlap <= 0:
            return torch.cat([left, right], dim=2)
        if left.shape[:2] != right.shape[:2] or left.shape[3:] != right.shape[3:]:
            raise ValueError("Cannot blend scene latents with different non-time dimensions.")
        alpha = torch.linspace(1.0, 0.0, overlap + 2, device=left.device, dtype=left.dtype)[1:-1]
        shape = [1] * left.dim()
        shape[2] = overlap
        alpha = alpha.reshape(shape)
        blended = alpha * left[:, :, -overlap:] + (1.0 - alpha) * right[:, :, :overlap].to(left.device, left.dtype)
        return torch.cat([left[:, :, :-overlap], blended, right[:, :, overlap:].to(left.device, left.dtype)], dim=2)

    def _blend_latents(self, previous, current, video_overlap):
        result = self._clone_latent(previous)
        previous_tensors = self._latent_tensors(previous)
        current_tensors = self._latent_tensors(current)
        if len(previous_tensors) != len(current_tensors):
            raise ValueError("Cannot blend different latent structures.")

        video_frames = self._tensor_frames(current_tensors[0])
        blended_tensors = []
        for index, tensor in enumerate(current_tensors):
            tensor_frames = self._tensor_frames(tensor)
            overlap = video_overlap if index == 0 else self._derived_overlap(video_overlap, video_frames, tensor_frames)
            blended_tensors.append(self._blend_tensors(previous_tensors[index], tensor, overlap))

        if self._is_nested(previous.get("samples")):
            result["samples"] = comfy.nested_tensor.NestedTensor(blended_tensors)
        else:
            result["samples"] = blended_tensors[0]
        result.pop("noise_mask", None)
        return result

    def _sample_chunk(self, model, sampler, sigmas, seed, cfg, positive, negative, latent):
        if sampler is None:
            raise ValueError("sampler input is required.")
        if not isinstance(sigmas, torch.Tensor):
            raise ValueError("sigmas input must be a SIGMAS tensor.")
        latent = self._clone_latent(latent)
        samples = latent["samples"]
        noise = comfy.sample.prepare_noise(samples, int(seed))
        sampled = comfy.sample.sample_custom(
            model,
            noise,
            float(cfg),
            sampler,
            sigmas,
            positive,
            negative,
            samples,
            noise_mask=latent.get("noise_mask"),
            seed=int(seed),
        )
        latent["samples"] = sampled
        latent.pop("noise_mask", None)
        return latent

    def _scene_text(self, scene_conditioning, index):
        if (
            isinstance(scene_conditioning, (list, tuple))
            and len(scene_conditioning) >= 2
            and isinstance(scene_conditioning[1], dict)
        ):
            text = str(scene_conditioning[1].get("funpack_scene_text", "") or "").strip()
            if text:
                return text
        return f"Scene {index + 1}"

    def sample(self, model, vae, positive, negative, sampler, sigmas, seed, latent_template,
               num_frames_per_scene, frame_overlap, cfg, max_scenes, carry_i2v_guides=False):
        if not isinstance(positive, list) or not positive:
            raise ValueError("positive conditioning must contain at least one scene entry.")
        if negative is None:
            negative = []

        max_scene_count = max(1, int(max_scenes))
        scene_conditionings = positive[:max_scene_count]
        scene_count = len(scene_conditionings)
        time_scale = self._time_scale(vae)
        video_frames = self._validate_template_length(latent_template, num_frames_per_scene, time_scale)
        video_overlap = self._overlap_frames(latent_template, frame_overlap, time_scale)

        output = None
        report_lines = []
        carried_guide_frames = 0
        for scene_index, scene_cond in enumerate(scene_conditionings):
            scene_positive = [scene_cond]
            scene_negative = negative
            scene_seed = int(seed) + scene_index
            carried = 0
            if output is None:
                chunk = self._clone_latent(latent_template)
            else:
                chunk = self._build_continuation_chunk(latent_template, output, video_overlap)
                if carry_i2v_guides:
                    chunk, scene_positive, scene_negative, carried = self._append_i2v_guides(
                        chunk, latent_template, scene_positive, scene_negative, vae,
                    )
                    carried_guide_frames = max(carried_guide_frames, carried)
            sampled = self._sample_chunk(
                model, sampler, sigmas, scene_seed, cfg, scene_positive, scene_negative, chunk,
            )
            if carry_i2v_guides and carried > 0:
                sampled = self._crop_video_tail(sampled, carried)
            output = sampled if output is None else self._blend_latents(output, sampled, video_overlap)
            report_lines.append(f"Scene {scene_index + 1}: seed={scene_seed}, text={self._scene_text(scene_cond, scene_index)}")

        final_frames = self._tensor_frames(self._latent_tensors(output)[0])
        status = (
            f"Scene chain complete: {scene_count} scene(s), "
            f"template={video_frames} latent frames, overlap={video_overlap}, output={final_frames}"
        )
        if carry_i2v_guides and carried_guide_frames > 0:
            status += f", i2v guide tokens={carried_guide_frames} latent frame(s)"
        return (output, status, scene_count, "\n".join(report_lines))
