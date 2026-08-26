import copy
import hashlib
import math
import os
import time as _time
import types

import torch

import comfy.k_diffusion.sampling as k_diffusion_sampling
import comfy.model_management
import comfy.model_sampling
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils

try:
    from . import funpack_log as _log
except ImportError:  # flat import when ComfyUI loads the pack as a top-level module
    import funpack_log as _log


MOTION_PULSE_MODES = ["off", "balanced", "aggressive", "custom"]



VELOCITY_BIAS_MODES = ["off", "capture", "apply", "capture_and_apply"]
# Normalized-sigma targets (current/sigmas[0]) where velocity is captured and rescue acts.
# Chosen to land on the structure AND detail-forming steps of the standard 8-step LTX
# schedule [..,0.909,0.725,0.422,0] (matched within +/-0.065): 0.90->0.909 (structure),
# 0.72->0.725 and 0.42->0.422 (where runs actually go good/bad). The old (0.90, 0.80) only
# ever matched 0.909, so rescue fired once per gen at a near-noise step. Velocity-bias APPLY
# still fires only at the highest target (structure); the extra targets are for rescue.
VELOCITY_BIAS_TARGETS = (0.90, 0.72, 0.42)
# Good-trajectory bank (rating-promoted). Also read by velocity-bias "apply".
VELOCITY_BIAS_MEMORY = {}
# Bad-trajectory bank (promoted from Awful/dislike), used by rescue to steer away.
VELOCITY_BIAS_BAD_MEMORY = {}
# Per-key staging of the most recently captured trajectory, awaiting its rating.
# capture stages here; the refiner commits it to the good or bad bank (or drops it)
# once the user rates that generation — so neither bank is rating-blind.
VELOCITY_BIAS_PENDING = {}
# Refinement keys whose banks have been loaded from disk this process (load-once).
_VELOCITY_LOADED = set()


def _velocity_store_path(refinement_key):
    """On-disk store for a key's velocity/rescue banks, alongside the other per-key
    refinement data. Persisted so the learned good/bad trajectories survive restarts."""
    base = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(base, "refinements", "attn_maps")
    os.makedirs(d, exist_ok=True)
    norm = str(refinement_key or "default").strip() or "default"
    h = hashlib.md5(f"velocity::{norm}".encode()).hexdigest()[:16]
    return os.path.join(d, f"velocity_{h}.pt")


def _serialize_velocity_bank(bank, norm):
    """weights_only-safe representation (lists/dicts/str/int + tensors; no tuple keys)."""
    out = []
    for key, slot in bank.items():
        if not (isinstance(key, tuple) and len(key) == 3 and key[0] == norm):
            continue
        direction = slot.get("direction")
        out.append({
            "refkey": str(key[0]), "target": str(key[1]),
            "shape": [int(v) for v in key[2]],
            "count": int(slot.get("count", 0)),
            "direction": direction.half() if isinstance(direction, torch.Tensor) else None,
            "clusters": [
                {"sig": c["sig"].half() if isinstance(c.get("sig"), torch.Tensor) else None,
                 "direction": c["direction"].half() if isinstance(c.get("direction"), torch.Tensor) else None,
                 "count": int(c.get("count", 0))}
                for c in slot.get("clusters", []) if isinstance(c, dict)
            ],
        })
    return out


def _deserialize_velocity_into(bank, entries):
    for e in entries or []:
        try:
            # Old stores included an "aspect" field — ignored now (key dropped it).
            key = (str(e["refkey"]), str(e["target"]),
                   tuple(int(v) for v in e["shape"]))
            d = e.get("direction")
            bank[key] = {
                "count": int(e.get("count", 0)),
                "direction": d.float() if isinstance(d, torch.Tensor) else None,
                "clusters": [
                    {"sig": c["sig"].float() if isinstance(c.get("sig"), torch.Tensor) else None,
                     "direction": c["direction"].float() if isinstance(c.get("direction"), torch.Tensor) else None,
                     "count": int(c.get("count", 0))}
                    for c in e.get("clusters", []) if isinstance(c, dict)
                ],
            }
        except Exception:
            continue


def _ensure_velocity_loaded(refinement_key):
    """Load a key's persisted banks into memory once per process."""
    norm = str(refinement_key or "default").strip() or "default"
    if norm in _VELOCITY_LOADED:
        return
    _VELOCITY_LOADED.add(norm)
    path = _velocity_store_path(norm)
    if not os.path.exists(path):
        return
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        _deserialize_velocity_into(VELOCITY_BIAS_MEMORY, data.get("good", []))
        _deserialize_velocity_into(VELOCITY_BIAS_BAD_MEMORY, data.get("bad", []))
        ng = sum(1 for k in VELOCITY_BIAS_MEMORY if isinstance(k, tuple) and k[0] == norm)
        nb = sum(1 for k in VELOCITY_BIAS_BAD_MEMORY if isinstance(k, tuple) and k[0] == norm)
        print(f"[FunPack rescue] loaded persisted velocity memory for '{norm}': {ng} good, {nb} bad bucket(s)")
    except Exception as e:
        print(f"[FunPack rescue] load failed: {e}")


def _save_velocity_store(refinement_key):
    """Persist a key's good+bad banks to disk (called after a commit changes them)."""
    norm = str(refinement_key or "default").strip() or "default"
    path = _velocity_store_path(norm)
    try:
        good = _serialize_velocity_bank(VELOCITY_BIAS_MEMORY, norm)
        bad = _serialize_velocity_bank(VELOCITY_BIAS_BAD_MEMORY, norm)
        if not good and not bad:
            if os.path.exists(path):
                os.remove(path)
            return
        torch.save({"good": good, "bad": bad}, path)
    except Exception as e:
        print(f"[FunPack rescue] save failed: {e}")

# Prompt-conditioned rescue clustering. The good-trajectory memory keeps, per
# velocity key, a set of clusters each tagged with a pooled+normalized prompt
# signature. Capture merges into the nearest cluster (or starts a new one);
# rescue builds its reference from clusters whose prompt is similar enough to the
# current run, so it steers toward what was good *for this kind of prompt* rather
# than a prompt-blind global average.
RESCUE_CLUSTER_MERGE_SIM = 0.92   # capture: merge into a cluster at/above this cosine
RESCUE_CLUSTER_MAX = 8            # cap clusters per key (evict lowest-count); each holds a full-shape direction
RESCUE_USE_SIM = 0.5              # rescue: only use clusters at/above this cosine
RESCUE_MIN_COUNT = 2              # rescue: need this many matched captures to act


def _normalize_sig(sig):
    """Pooled prompt vector -> CPU float unit vector, or None."""
    if not isinstance(sig, torch.Tensor) or sig.numel() == 0:
        return None
    s = sig.detach().float().reshape(-1).cpu()
    n = s.norm()
    if not torch.isfinite(n) or float(n) <= 1e-8:
        return None
    return s / n


def _update_prompt_clusters(slot, prompt_sig, direction):
    """Capture-side: fold (prompt_sig, direction) into the slot's prompt clusters."""
    if prompt_sig is None:
        return
    clusters = slot.setdefault("clusters", [])
    best_i, best_cos = -1, -1.0
    for i, c in enumerate(clusters):
        cs = c.get("sig")
        if not isinstance(cs, torch.Tensor) or cs.shape != prompt_sig.shape:
            continue
        cos = float(cs @ prompt_sig)
        if cos > best_cos:
            best_cos, best_i = cos, i
    if best_i >= 0 and best_cos >= RESCUE_CLUSTER_MERGE_SIM:
        c = clusters[best_i]
        n = int(c.get("count", 1))
        if isinstance(c.get("direction"), torch.Tensor) and c["direction"].shape == direction.shape:
            c["direction"] = (c["direction"] * n + direction) / float(n + 1)
        merged = c["sig"] * n + prompt_sig
        mn = merged.norm().clamp_min(1e-8)
        c["sig"] = merged / mn
        c["count"] = min(n + 1, 256)
    else:
        clusters.append({"sig": prompt_sig.clone(), "direction": direction.clone(), "count": 1})
        if len(clusters) > RESCUE_CLUSTER_MAX:
            clusters.sort(key=lambda c: int(c.get("count", 0)), reverse=True)
            del clusters[RESCUE_CLUSTER_MAX:]


def _select_prompt_direction(slot, prompt_sig, x_shape):
    """Rescue-side: similarity-weighted good direction over prompt-relevant clusters.
    Returns None when there is no confident prompt-matching reference (caller then
    skips rescue rather than steering toward a prompt-blind average)."""
    clusters = slot.get("clusters") if isinstance(slot, dict) else None
    if not clusters or prompt_sig is None:
        return None
    num = None
    total_w = 0.0
    matched = 0
    for c in clusters:
        cs = c.get("sig")
        d = c.get("direction")
        if not isinstance(cs, torch.Tensor) or cs.shape != prompt_sig.shape:
            continue
        if not isinstance(d, torch.Tensor) or tuple(d.shape) != tuple(x_shape):
            continue
        cos = float(cs @ prompt_sig)
        if cos < RESCUE_USE_SIM:
            continue
        contrib = d * cos
        num = contrib if num is None else num + contrib
        total_w += cos
        matched += int(c.get("count", 1))
    if num is None or total_w <= 0.0 or matched < RESCUE_MIN_COUNT:
        return None
    return num / total_w


def _nearest_cluster_direction(slot, prompt_sig, x_shape):
    """Option D: the single highest-similarity prompt cluster's direction (>= RESCUE_USE_SIM).

    Returns one cluster's direction rather than a blend across clusters, preserving that
    real good trajectory's high-frequency content instead of averaging several together
    (averaging is what low-passes / softens). None if no cluster matches confidently.
    """
    clusters = slot.get("clusters") if isinstance(slot, dict) else None
    if not clusters or prompt_sig is None:
        return None
    best_d, best_cos = None, RESCUE_USE_SIM
    for c in clusters:
        cs = c.get("sig")
        d = c.get("direction")
        if not isinstance(cs, torch.Tensor) or cs.shape != prompt_sig.shape:
            continue
        if not isinstance(d, torch.Tensor) or tuple(d.shape) != tuple(x_shape):
            continue
        cos = float(cs @ prompt_sig)
        if cos >= best_cos:
            best_cos, best_d = cos, d
    return best_d


def _velocity_direction(slot, prompt_sig, x_shape, source="mean"):
    """Resolve a steering direction from a bank slot.

    source="nearest" (option D): single best-matching prompt cluster — preserves one
      real good trajectory's detail. Falls back to the global mean when no cluster
      matches (e.g. no prompt sig, or memory captured before clustering existed).
    source="mean" (default): prompt-blind global average (legacy behavior).
    """
    if source == "nearest":
        d = _nearest_cluster_direction(slot, prompt_sig, x_shape)
        if d is not None:
            return d
    g = slot.get("direction") if isinstance(slot, dict) else None
    return g if isinstance(g, torch.Tensor) and tuple(g.shape) == tuple(x_shape) else None


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


def clear_velocity_bias_memory(refinement_key):
    """Drop all stored velocity-bias / rescue trajectory memory (global averages and
    prompt clusters) for a key. Called on Studio session reset so rescue and velocity
    bias start from a clean slate instead of steering toward last session's content."""
    norm = str(refinement_key or "default").strip() or "default"
    # Make sure persisted buckets are in memory first, so the on-disk store is rewritten
    # empty (deleted) rather than left behind.
    _ensure_velocity_loaded(norm)
    removed = 0
    for bank in (VELOCITY_BIAS_MEMORY, VELOCITY_BIAS_BAD_MEMORY):
        for k in [k for k in list(bank) if isinstance(k, tuple) and k and k[0] == norm]:
            bank.pop(k, None)
            removed += 1
    VELOCITY_BIAS_PENDING.pop(norm, None)
    # Delete the on-disk store too (this is the deliberate Session-reset path).
    try:
        path = _velocity_store_path(norm)
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"[FunPack rescue] disk clear failed: {e}")
    if removed:
        print(f"[FunPack rescue] cleared {removed} velocity-bias memory bucket(s) "
              f"(good+bad) for key '{norm}' (session reset)")
    return removed


def _velocity_bias_enabled(mode, action):
    mode = (mode or "off").lower()
    if mode == "capture_and_apply":
        return action in {"capture", "apply"}
    return mode == action


def _velocity_bias_key(refinement_key, target, x):
    # (refinement_key, target, latent_shape). Shape already encodes resolution, so a
    # separate aspect bucket was redundant — dropped.
    shape = tuple(int(item) for item in getattr(x, "shape", ()))
    key = str(refinement_key or "default").strip() or "default"
    return (key, f"{float(target):.2f}", shape)


def _sigma_ratio(sigmas, sigma):
    try:
        start = float(sigmas[0].item())
        current = float(sigma.item())
    except Exception:
        return None
    if start <= 0.0:
        return None
    return current / start


def _effective_eta(eta, eta_final, sigmas, sigma):
    """Ancestral eta decayed across the schedule: full ``eta`` at the start
    (sigma == sigmas[0]) ramping linearly to ``eta_final`` near the end (sigma -> 0).

    Anchored to raw schedule progress (sigma/sigmas[0]), NOT the quality-phase boundary.
    The old anchor collapsed to sigma=0 when high_quality_pct=0, silently pinning
    effective_eta at ``eta`` for every step (eta_final ignored) and leaving full-strength
    ancestral noise on the final detail steps with no deterministic phase to resolve it.
    """
    if eta_final >= eta:
        return eta
    ratio = _sigma_ratio(sigmas, sigma)
    if ratio is None:
        return eta
    ratio = max(0.0, min(1.0, ratio))
    return eta_final + (eta - eta_final) * ratio


def _velocity_bias_target(sigmas, sigma):
    ratio = _sigma_ratio(sigmas, sigma)
    if ratio is None:
        return None
    target = min(VELOCITY_BIAS_TARGETS, key=lambda item: abs(float(item) - ratio))
    return target if abs(float(target) - ratio) <= 0.065 else None


def _capture_velocity_bias(refinement_key, target, x, sigma, denoised, prompt_sig=None):
    """Stage this step's trajectory for the current generation. It is NOT committed to
    any bank here — the refiner commits it to good or bad once the gen is rated, so the
    banks never absorb unrated (possibly awful) runs. Staging overwrites per (key, shape,
    target) so only the latest generation for a key is pending."""
    if target is None:
        return
    try:
        direction = k_diffusion_sampling.to_d(x, sigma, denoised).detach().float().cpu()
    except Exception:
        return
    key = _velocity_bias_key(refinement_key, target, x)
    norm = key[0]
    staged = VELOCITY_BIAS_PENDING.setdefault(norm, {})
    staged[key] = {"direction": direction,
                   "sig": prompt_sig.clone() if isinstance(prompt_sig, torch.Tensor) else None}


def _merge_into_bank(bank, key, direction, prompt_sig):
    """Fold one staged trajectory into a bank's global average + prompt clusters."""
    slot = bank.setdefault(key, {"count": 0, "direction": None})
    count = int(slot.get("count", 0))
    previous = slot.get("direction")
    if count <= 0 or not isinstance(previous, torch.Tensor) or tuple(previous.shape) != tuple(direction.shape):
        slot["direction"] = direction
        slot["count"] = 1
    else:
        slot["direction"] = (previous * count + direction) / float(count + 1)
        slot["count"] = min(count + 1, 256)
    _update_prompt_clusters(slot, prompt_sig, direction)


def commit_staged_velocity(refinement_key, verdict):
    """Commit the staged trajectory for a key to the good or bad bank based on the
    generation's rating. verdict: 'good' | 'bad' | 'drop'. Called by the refiner when
    it processes the previous gen's rating. Always clears the staging slot."""
    norm = str(refinement_key or "default").strip() or "default"
    staged = VELOCITY_BIAS_PENDING.pop(norm, None)
    if not staged or verdict == "drop":
        return 0
    _ensure_velocity_loaded(norm)  # merge on top of any persisted buckets
    bank = VELOCITY_BIAS_MEMORY if verdict == "good" else VELOCITY_BIAS_BAD_MEMORY
    n = 0
    for key, entry in staged.items():
        d = entry.get("direction")
        if isinstance(d, torch.Tensor):
            _merge_into_bank(bank, key, d, entry.get("sig"))
            n += 1
    if n:
        _save_velocity_store(norm)  # persist so it survives restarts
        print(f"[FunPack rescue] committed {n} staged trajectory bucket(s) to "
              f"'{verdict}' bank for key '{norm}' (persisted)")
    return n


def _apply_velocity_bias(x, refinement_key, target, strength, sigma_ratio=None,
                         prompt_sig=None, source="mean"):
    """Steer the latent toward the averaged good-trajectory direction.

    Two anti-softening guards (see velocity-bias quality findings):
      C. Structure-only + sigma decay — only fires at the highest velocity target
         (structure-forming sigma, ~0.90 ratio); the lower detail target is left
         untouched so high-frequency detail forms freely. Within that, strength
         decays with the sigma ratio so the push is strongest early and fades as
         sigma drops.
      B. Magnitude-preserving — instead of adding the delta as raw energy
         (x + delta, which pulls the latent toward the low-variance mean and
         progressively softens), we rotate x toward (x + delta) and renormalize
         back to |x|. Direction is steered; no energy is injected.
    """
    if target is None or strength <= 0.0:
        return x
    # C: structure target only — skip the lower (detail) targets entirely.
    if float(target) < max(VELOCITY_BIAS_TARGETS) - 1e-6:
        return x
    _ensure_velocity_loaded(refinement_key)
    key = _velocity_bias_key(refinement_key, target, x)
    slot = VELOCITY_BIAS_MEMORY.get(key)
    if not isinstance(slot, dict):
        return x
    # D: nearest single good trajectory (prompt cluster) vs prompt-blind global mean.
    direction = _velocity_direction(slot, prompt_sig, x.shape, source=source)
    if not isinstance(direction, torch.Tensor):
        return x
    try:
        # C: decay strength with the sigma ratio (full at the structure target, less below).
        decay = 1.0
        if sigma_ratio is not None:
            decay = max(0.0, min(1.0, float(sigma_ratio) / max(VELOCITY_BIAS_TARGETS)))
        eff_strength = max(0.0, min(3.0, float(strength))) * decay
        if eff_strength <= 0.0:
            return x
        direction = direction.to(device=x.device, dtype=x.dtype)
        delta = direction * eff_strength
        x_norm = x.detach().float().norm().clamp_min(1e-8)
        # Cap the rotation toward the remembered direction, scaled by strength so the WHOLE
        # slider is meaningful (the old fixed 4.5% saturated almost immediately, making the
        # upper range a no-op). 0.30*strength keeps the low end ~unchanged (strength~0.15 ==
        # old ~4.5%) while high strength approaches full action replacement — capped at 0.95
        # so it never totally wipes the current gen. Magnitude-preserving renorm keeps |x|.
        max_delta = x_norm * min(0.95, 0.30 * eff_strength)
        delta_norm = delta.detach().float().norm().clamp_min(1e-8)
        if delta_norm > max_delta:
            delta = delta * (max_delta / delta_norm).to(device=x.device, dtype=x.dtype)
        # B: rotate, don't add — renormalize the biased latent back to the original norm.
        biased = x + delta
        biased_norm = biased.detach().float().norm().clamp_min(1e-8)
        return biased * (x_norm / biased_norm).to(device=x.device, dtype=x.dtype)
    except Exception as _e:
        _log.failed("FunPackSceneChain", "velocity bias", _e,
                    "this step runs unbiased — motion will read as it would with the knob at 0")
        return x


# Per-run rescue logging state, reset at the start of each sampler call.
_RESCUE_LOG = {"warned_no_memory": False, "warned_no_prompt_match": False, "fired": 0}


def _rescue_reference(bank, key, prompt_sig, x_shape, source="mean"):
    """Resolve a steering direction from a bank.

    source="nearest" (option D): single best-matching prompt cluster (one real good
      trajectory's detail), falling back to the global average if none matches.
    source="mean" (default): similarity-weighted blend over prompt-relevant clusters
      when a prompt signature is available, else the prompt-blind global average.
    """
    slot = bank.get(key)
    if not isinstance(slot, dict):
        return None
    if source == "nearest":
        return _velocity_direction(slot, prompt_sig, x_shape, source="nearest")
    if prompt_sig is not None:
        return _select_prompt_direction(slot, prompt_sig, x_shape)
    g = slot.get("direction")
    return g if isinstance(g, torch.Tensor) and tuple(g.shape) == tuple(x_shape) else None


def _rescue_denoised(denoised, x, sigma, refinement_key, target, threshold, strength, prompt_sig=None, source="mean"):
    """Reactive in-flight rescue, rating-aware.

    Pulls this step's trajectory toward the GOOD bank (rating-promoted likes) and away
    from the BAD bank (Awful/dislike). Fires when the trajectory has diverged from good
    beyond ``threshold`` OR aligned with bad beyond ``threshold``. Magnitude preserved
    (no energy injected). No matching reference in either bank -> no-op.
    """
    if target is None or strength <= 0.0:
        return denoised
    sig = float(sigma.item()) if isinstance(sigma, torch.Tensor) else float(sigma)
    if sig <= 1e-6:
        return denoised
    _ensure_velocity_loaded(refinement_key)
    key = _velocity_bias_key(refinement_key, target, x)
    good = _rescue_reference(VELOCITY_BIAS_MEMORY, key, prompt_sig, x.shape, source=source)
    bad = _rescue_reference(VELOCITY_BIAS_BAD_MEMORY, key, prompt_sig, x.shape, source=source)
    if good is None and bad is None:
        if not _RESCUE_LOG["warned_no_prompt_match"]:
            print("[FunPack rescue] no rated trajectory for this prompt yet — rate a few gens "
                  "(good builds the target, awful builds what to avoid). Skipping until then.")
            _RESCUE_LOG["warned_no_prompt_match"] = True
        return denoised
    try:
        thr = float(threshold)
        d = k_diffusion_sampling.to_d(x, sigma, denoised).detach().float()
        d_norm = d.norm().clamp_min(1e-8)
        d_unit = d / d_norm

        cos_good = cos_bad = None
        c = d_unit.clone()
        fired_reason = []
        w = max(0.0, min(0.5, float(strength)))

        if good is not None:
            g = good.to(device=x.device, dtype=torch.float32)
            g_unit = g / g.norm().clamp_min(1e-8)
            cos_good = float(d_unit.flatten() @ g_unit.flatten())
            if (1.0 - cos_good) > thr:
                c = c + w * (g_unit - d_unit)            # pull toward good
                fired_reason.append(f"div_good={1.0 - cos_good:.3f}")

        if bad is not None:
            b = bad.to(device=x.device, dtype=torch.float32)
            b_unit = b / b.norm().clamp_min(1e-8)
            cu = c / c.norm().clamp_min(1e-8)
            cos_bad = float(cu.flatten() @ b_unit.flatten())
            if cos_bad > thr:
                c = c - w * cos_bad * b_unit             # remove bad-aligned component
                fired_reason.append(f"sim_bad={cos_bad:.3f}")

        if not fired_reason:
            return denoised  # on-manifold w.r.t. good and clear of bad

        c_norm = c.norm().clamp_min(1e-8)
        d_corr = d_norm * (c / c_norm)
        denoised_corr = (x.float() - sig * d_corr).to(dtype=denoised.dtype, device=denoised.device)
        _RESCUE_LOG["fired"] += 1
        if _RESCUE_LOG["fired"] <= 8:
            refs = f"good={'y' if good is not None else 'n'},bad={'y' if bad is not None else 'n'}"
            print(f"[FunPack rescue] sigma~{sig:.3f} {' '.join(fired_reason)} > thr={thr:.3f} "
                  f"[{refs}] -> steered (w={w:.2f})")
        return denoised_corr
    except Exception as e:
        print(f"[FunPack rescue] skipped: {e}")
        return denoised


def _apply_quality_sharpness(denoised, prev_denoised, sharpness):
    """E: temporal-average unsharp on the x0 prediction during the quality phase.

    The velocity-bias mean-pull (plus order-2 averaging) low-pass the latent and
    soften fine detail. This boosts the high-frequency component of the current
    denoised prediction relative to the previous step's prediction (a cheap
    temporal low-pass), restoring detail with no extra model eval. No-op when
    disabled, on the first quality step, or after a pulse reset (no prev).
    """
    if not sharpness or sharpness <= 0.0 or prev_denoised is None:
        return denoised
    try:
        amount = max(0.0, min(1.0, float(sharpness)))
        prev = prev_denoised.to(device=denoised.device, dtype=denoised.dtype)
        if tuple(prev.shape) != tuple(denoised.shape):
            return denoised
        detail = denoised - 0.5 * (denoised + prev)   # 0.5 * (denoised - prev): temporal high-pass
        return denoised + amount * detail
    except Exception as _e:
        _log.failed("FunPackSceneChain", "quality sharpness", _e,
                    "this step keeps the unsharpened prediction")
        return denoised


# ---------------------------------------------------------------------------
# Blackwell (sm_120) xformers masked-attention fallback
# ---------------------------------------------------------------------------
# On GPUs newer than capability (9, 0), xformers' memory_efficient_attention has no
# kernel that accepts a tensor attn_bias — every backend is rejected ("No operator
# found ... too new ... attn_bias type is <class 'torch.Tensor'>"). Unmasked attention
# still works, which is why i2v ANCHOR scenes generate fine but GUIDE scenes crash: the
# LTX guide path (comfy/ldm/lightricks/model.py _attention_with_guide_mask) passes a
# tensor mask. PyTorch SDPA handles masks on every GPU, so when the active backend is
# xformers on such a device we route only the MASKED calls to attention_pytorch via
# ComfyUI's per-call optimized_attention_override hook (wrap_attn). Unmasked calls stay
# on the fast xformers path. This is the programmatic equivalent of the user's
# --use-pytorch-cross-attention, scoped to masked attention and installed ONLY when this
# exact failure condition is detected, so other setups are untouched.
def _funpack_mask_safe_attention_override(func, *args, **kwargs):
    # `func` is the raw backend ComfyUI selected (attention_xformers). wrap_attn has
    # already stamped _inside_attn_wrapper into kwargs before calling us, so invoking
    # attention_pytorch here cannot re-enter the override (no recursion).
    mask = kwargs.get("mask")
    if mask is None and len(args) >= 5:
        mask = args[4]  # (q, k, v, heads, mask, ...)
    if mask is not None:
        import comfy.ldm.modules.attention as _am
        return _am.attention_pytorch(*args, **kwargs)
    return func(*args, **kwargs)


def _funpack_install_mask_safe_attention(model):
    """When the active attention backend is xformers on a GPU too new for its tensor-bias
    kernels (capability > (9, 0), e.g. Blackwell sm_120), install a per-call override so
    MASKED attention (the LTX guide path) falls back to SDPA. No-op on supported GPUs, when
    ComfyUI already uses a mask-capable backend, or when another override is present.
    Idempotent; never raises."""
    try:
        import comfy.ldm.modules.attention as _am
        if getattr(_am, "optimized_attention", None) is not getattr(_am, "attention_xformers", None):
            return  # SDPA / sage / flash / split selected — masks already handled
        if not torch.cuda.is_available():
            return
        major, minor = torch.cuda.get_device_capability()
        if major <= 9:
            return  # xformers cutlass/flash tensor-bias kernels support cap <= (9, 0)
        to = model.model_options.setdefault("transformer_options", {})
        existing = to.get("optimized_attention_override")
        if existing is _funpack_mask_safe_attention_override:
            return  # already installed (idempotent across scenes/runs)
        if existing is not None:
            return  # respect another tool's override; don't stomp it
        to["optimized_attention_override"] = _funpack_mask_safe_attention_override
        print(f"[FunPack AV] xformers active on capability {major}.{minor} GPU (too new for "
              "its tensor-bias kernels): routing MASKED attention to PyTorch SDPA so guide "
              "scenes don't crash. Unmasked attention stays on xformers.")
    except Exception as _e:
        print(f"[FunPack AV] mask-safe attention install skipped: {_e}")


# ---------------------------------------------------------------------------
# KV-Lock: variance-gated scheduler for BachVid K/V injection (arxiv 2603.09657)
#
# BachVid (ltx_enhancements) injects a blessed identity K/V at a fixed,
# reward-scaled strength. KV-Lock makes that strength adaptive to the model's
# own confidence: track how much the predicted clean latent x0 still jumps
# between steps (relative step-change = a cheap proxy for the paper's variance-
# of-x0 hallucination metric). High instability => the model is still drifting
# => raise the multiplier so the identity K/V is held harder; once x0 settles,
# the multiplier falls back toward the base strength. No DDIM inversion (LTXAV
# inverts poorly) — we lock against the blessed bank, not an inverted source.
#
# No-op unless a BachVid K/V bank installed the shared scale list in
# transformer_options. Never raises — scheduling must not break sampling.
# ---------------------------------------------------------------------------
_KVLOCK_TAU = 0.012   # instability threshold (relative x0 step-change)
_KVLOCK_B = 2.0       # max multiplier (lets injection boost above base strength)
_KVLOCK_W = 5         # sliding window length (steps)


def _kvlock_find_scale_list(model):
    """Locate the shared kvlock_scale list build_enhancements stored in
    transformer_options. Returns the mutable list, or None."""
    for getter in (
        lambda: model.inner_model.model_patcher.model_options,
        lambda: model.inner_model.model_options,
        lambda: model.model_options,
    ):
        try:
            mo = getter()
            to = mo.get("transformer_options") if isinstance(mo, dict) else None
            if isinstance(to, dict):
                lst = to.get("funpack_kvlock_scale")
                if isinstance(lst, list) and lst:
                    return lst
        except Exception:
            pass
    return None


def _kvlock_schedule(model, denoised, prev_denoised, video_mask, state):
    """Update the shared kvlock_scale from x0 prediction instability.

    Sets the multiplier read by BachVid's K/V injector on the NEXT model() call.
    The list lookup is cached (sentinel "miss" = not yet searched). Never raises."""
    try:
        if state.get("list", "miss") == "miss":
            state["list"] = _kvlock_find_scale_list(model)
        scale_list = state["list"]
        if scale_list is None or prev_denoised is None:
            return
        d = denoised.detach().float()
        p = prev_denoised.detach().float().to(d.device)
        if d.shape != p.shape:
            return
        if video_mask is not None and video_mask.shape[-1] == d.shape[-1]:
            m = video_mask.to(d.device)
            num = ((d - p) * m).norm()
            den = (d * m).norm().clamp(min=1e-6)
        else:
            num = (d - p).norm()
            den = d.norm().clamp(min=1e-6)
        instability = float(num / den)
        deltas = state.setdefault("deltas", [])
        deltas.append(instability)
        if len(deltas) > _KVLOCK_W:
            del deltas[:-_KVLOCK_W]
        mean_delta = sum(deltas) / len(deltas)
        scale_list[0] = max(0.0, min(_KVLOCK_B, mean_delta / _KVLOCK_TAU))
    except Exception:
        pass


def _get_latent_shapes(model):
    """Recover comfy's packed-latent stream shapes [video_shape, audio_shape, ...].

    LTXAV packs its video + audio latents into one [B,1,N] tensor for sampling
    (comfy.utils.pack_latents: each stream flattened to [B,1,-1] and concatenated on
    the last dim, video first). comfy threads the per-stream shapes through the guider
    conds as a 'latent_shapes' CONDConstant. We read them back so FunPack perturbations
    can be confined to the video region. None when unavailable or single-stream.
    """
    try:
        guider = getattr(model, "inner_model", None)  # CFGGuider
        conds = getattr(guider, "conds", None)
        if not isinstance(conds, dict):
            return None
        for key in ("positive", "negative"):
            lst = conds.get(key)
            if not lst:
                continue
            for c in lst:
                mc = c.get("model_conds") if isinstance(c, dict) else None
                ls = mc.get("latent_shapes") if isinstance(mc, dict) else None
                val = getattr(ls, "cond", None)
                if val and len(val) > 1:
                    return val
    except Exception:
        return None
    return None


def _packed_video_mask(model, x):
    """[1,1,N] mask over a packed AV latent: 1.0 on the video stream, 0.0 on audio.

    Ancestral re-noising (and other video-tuned perturbations) corrupt the audio stream
    while helping video. With this mask we keep all FunPack steering / ancestral noise on
    video and let audio ride the clean deterministic flow. None when not a packed AV
    latent (single-stream LTXV, or layout we can't verify) -> callers then no-op.
    """
    try:
        shapes = _get_latent_shapes(model)
        if not shapes or len(shapes) <= 1:
            return None
        if not hasattr(x, "shape") or x.dim() < 1:
            return None
        n = int(x.shape[-1])
        sizes = [int(math.prod(tuple(s)[1:])) for s in shapes]
        if sum(sizes) != n:
            return None  # layout doesn't match our assumption -> don't risk masking
        # Video = the highest-dimensional / largest stream (robust to stream ordering).
        video_idx = max(range(len(shapes)), key=lambda i: (len(tuple(shapes[i])), sizes[i]))
        mask = x.new_zeros((1, 1, n))
        off = 0
        for i, sz in enumerate(sizes):
            if i == video_idx:
                mask[..., off:off + sz] = 1.0
            off += sz
        return mask
    except Exception:
        return None


def _video_only(x_new, x_old, mask):
    """Confine a perturbation to the video region: audio keeps its base (x_old) value.

    Works for any perturbation expressed as new-vs-old (additive noise, steering,
    sharpness). For additive noise call as _video_only(x + noise, x, mask) = x + noise*mask.
    """
    if mask is None:
        return x_new
    try:
        return x_old + (x_new - x_old) * mask
    except Exception as _e:
        # The mask is what keeps a video-only perturbation off the audio stream. Losing it
        # does not fail the render, it lets the perturbation reach the waveform.
        _log.failed("FunPackSceneChain", "audio-protection mask", _e,
                    "the perturbation reached the AUDIO stream as well as the video")
        return x_new


def _audio_clock_setup(model, x, sigmas, enabled):
    """Build the per-step audio-clock correction for a MiniMax H3 run, or None.

    Returns (factors, mask) where `factors[i]` scales the audio stream's displacement on
    step i and `mask` is the packed video mask. None whenever the correction cannot be
    made — not H3, no readable packed layout, or shifts that make it a no-op — so callers
    just fall through to comfy's own tangent approximation.
    """
    if not enabled:
        return None
    try:
        try:
            from .minimax_h3 import audio_clock_factors, is_h3_model, resolve_sigma_shifts
        except ImportError:
            from minimax_h3 import audio_clock_factors, is_h3_model, resolve_sigma_shifts
        patcher = getattr(getattr(model, "inner_model", None), "model_patcher", None)
        if not is_h3_model(patcher):
            print("[FunPack AV] h3_audio_clock is on but this is not a MiniMax H3 model — "
                  "the two-schedule correction it applies exists only on H3. Not running.")
            return None
        mask = _packed_video_mask(model, x)
        if mask is None:
            print("[FunPack AV] h3_audio_clock is on but the packed video+audio latent layout "
                  "could not be read, so the audio stream cannot be located. Not running.")
            return None
        shift_v, shift_a = resolve_sigma_shifts(getattr(patcher, "model_options", None))
        factors = audio_clock_factors(sigmas, shift_v, shift_a)
        if not factors or all(abs(f - 1.0) < 1e-6 for f in factors):
            print(f"[FunPack AV] h3_audio_clock is on but shift_video ({shift_v:g}) and "
                  f"shift_audio ({shift_a:g}) put both streams on the same schedule — "
                  f"nothing to correct. Not running.")
            return None
        worst = min(factors)
        print(f"[FunPack AV] h3_audio_clock on (shift_video={shift_v:g}, shift_audio={shift_a:g}) "
              f"— audio integrated on its own schedule over {len(factors)} step(s); the most "
              f"corrected step moves audio to {worst * 100.0:.0f}% of what the video grid alone "
              f"would have moved it.")
        return factors, mask
    except Exception as error:
        print(f"[FunPack AV] h3_audio_clock could not be set up ({error}) — not running.")
        return None


def _audio_clock_step(x_new, x_old, clock, i):
    """Re-scale the audio stream's displacement over one step onto its own schedule.

    Video keeps `x_new` untouched. Audio keeps the same direction but the length the
    audio schedule actually calls for. Confining it to the displacement means any
    ancestral noise or steering the caller already restricted to video rides through
    unchanged. No-op without a clock, or past the end of the factor list.
    """
    if clock is None:
        return x_new
    try:
        factors, mask = clock
        if i >= len(factors):
            return x_new
        factor = factors[i]
        # video region -> multiplier 1, audio region -> multiplier `factor`
        scale = mask + (1.0 - mask) * factor
        return x_old + (x_new - x_old) * scale
    except Exception as _e:
        _log.failed("FunPackSceneChain", "audio clock correction", _e,
                    "audio integrates on the video schedule for this step (h3_audio_clock inert)")
        return x_new


# ── the audio clock on a sampler we do not own ───────────────────────────────
# FunPack's own samplers apply the clock inside their step loop, where the step's start
# and end sigma are both in hand. A stock comfy sampler has no such hook, so the only way
# in is the denoised value it asks for. That works because the correction can be moved
# from the step to the prediction: scaling the audio part of (x - denoised) by the same
# factor makes the sampler's own euler update land the audio exactly where its schedule
# says, since that update is x*r + denoised*(1-r) with r = sigma_next/sigma.
#
# The catch is knowing WHICH step a given call belongs to. A sampler that evaluates the
# model once per step makes call index == step index, and that covers euler, res_multistep,
# er_sde, dpmpp_2m and the rest of the multistep family — including the samplers ComfyUI's
# own H3 template uses. A sampler that evaluates more than once per step (heun, dpm_2,
# dpmpp_2s/sde) does not, and cannot be disambiguated from out here: its corrector call
# lands on the same (sigma, step count) signature as the next step's predictor.
#
# So the wrapper checks its assumption on every call — the sigma it is handed must be the
# one this step's single evaluation would use — and switches itself off, loudly, the moment
# that stops holding. It never guesses.
_AUDIO_CLOCK_SIGMA_TOL = 1e-4

# Measured against comfy's real samplers with a PERFECT rectified-flow predictor, so every
# number is pure schedule error (scratch harness, 2026-08-06; shift 12/3, audio error as a
# % of the stream's full noise->signal span):
#
#   sampler          4 steps      8 steps     20 steps
#   euler          85.4 -> 0.0  38.4 -> 0.0  13.5 -> 0.0     exact, at every step count
#   res_multistep  68.7 -> 20.7 17.5 -> 23.0  1.0 -> 14.8    already self-corrects; clock hurts
#   dpmpp_2m       69.0 -> 20.2 18.2 -> 22.3  0.4 -> 14.2    same
#   heun/dpm_2     unchanged — two evals per step, the proxy stands down
#
# The lesson: this is a FIRST-ORDER integration error, so a higher-order sampler already
# absorbs most of it once the schedule is fine enough — which is why res_multistep and
# er_sde behaved well on H3 at 20 steps and plain-euler paths did not. At 4 steps NOTHING
# absorbs it (everything sits at 68-85%) and only the clock fixes it, which is exactly the
# regime turbo LoRAs put you in.
#
# We do not gate on this — the user picks the sampler and these are the facts they need.
_AUDIO_CLOCK_EXACT_ON = ("sample_euler",)
_AUDIO_CLOCK_HARMED = ("sample_res_multistep", "sample_dpmpp_2m", "sample_gradient_estimation",
                       "sample_ipndm", "sample_ipndm_v", "sample_lms", "sample_deis")


class _AudioClockDenoiser:
    """Denoiser proxy that applies the audio clock to what a foreign sampler asks for.

    Transparent for everything else: samplers reach through to `inner_model`,
    `latent_image` and friends, so attribute access forwards to the wrapped object.
    """

    def __init__(self, inner, clock, sigmas):
        self._inner = inner
        self._clock = clock
        self._sigmas = sigmas
        self._step = 0
        self._live = True

    def __getattr__(self, name):
        # only reached for names not on the proxy itself
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        if name in ("_inner", "_clock", "_sigmas", "_step", "_live"):
            object.__setattr__(self, name, value)
        else:                       # e.g. samplers assigning model.latent_image
            setattr(self._inner, name, value)

    def _step_for(self, sigma):
        """The step this call belongs to, or None if the one-eval-per-step assumption broke."""
        i = self._step
        if i >= len(self._sigmas) - 1:
            return None
        try:
            want = float(self._sigmas[i])
            got = float(sigma.flatten()[0]) if hasattr(sigma, "flatten") else float(sigma)
        except Exception:
            return None
        if abs(got - want) > _AUDIO_CLOCK_SIGMA_TOL * max(1.0, abs(want)):
            return None
        return i

    def __call__(self, x, sigma, **kwargs):
        denoised = self._inner(x, sigma, **kwargs)
        if not self._live:
            return denoised
        i = self._step_for(sigma)
        if i is None:
            self._live = False
            print("[FunPack AV] h3_audio_clock: this sampler does not evaluate the model once "
                  "per step at the scheduled sigma, so a model call cannot be tied to a step "
                  "from outside its loop. Correction switched OFF for the rest of this run — "
                  "the audio is being integrated the way it was before. Use FunPack Distilled "
                  "Flow / Hybrid Euler 2S (the clock runs inside their loop and handles this), "
                  "or a one-eval-per-step sampler such as euler, res_multistep or er_sde.")
            return denoised
        self._step += 1
        # (x - denoised) is the step's direction; scaling its audio part is the same
        # correction the in-loop version applies to the displacement.
        return _audio_clock_step(denoised, x, self._clock, i)


def _audio_clock_wrap_sampler(sampler):
    """A SAMPLER equivalent to `sampler` with the audio clock applied around its model.

    Returns None when the sampler exposes no `sampler_function` to wrap — some SAMPLER
    producers are not KSAMPLER-shaped, and guessing at their internals is not worth it.
    """
    fn = getattr(sampler, "sampler_function", None)
    if fn is None:
        return None

    def _wrapped(model, x, sigmas, extra_args=None, callback=None, disable=None, **options):
        clock = _audio_clock_setup(model, x, sigmas, True)
        if clock is None:      # not H3, unreadable layout, or coincident schedules
            return fn(model, x, sigmas, extra_args=extra_args, callback=callback,
                      disable=disable, **options)
        proxy = _AudioClockDenoiser(model, clock, sigmas)
        return fn(proxy, x, sigmas, extra_args=extra_args, callback=callback,
                  disable=disable, **options)

    _wrapped.__name__ = getattr(fn, "__name__", "sampler") + "_h3_audio_clock"
    return comfy.samplers.KSAMPLER(
        _wrapped,
        extra_options=getattr(sampler, "extra_options", None) or {},
        inpaint_options=getattr(sampler, "inpaint_options", None) or {},
    )


def _video_span(model, x):
    """(offset, size, stream_shape) of the video stream inside a packed AV latent
    [B,1,N], or None when the layout can't be verified (same guards and stream
    heuristic as _packed_video_mask — mismatch means don't risk slicing).
    stream_shape is the per-stream latent shape comfy packed from (e.g.
    (B, C, T, H, W)), so callers can recover frame geometry from the flat span."""
    try:
        shapes = _get_latent_shapes(model)
        if not shapes or len(shapes) <= 1:
            return None
        if not hasattr(x, "shape") or x.dim() < 1:
            return None
        n = int(x.shape[-1])
        sizes = [int(math.prod(tuple(s)[1:])) for s in shapes]
        if sum(sizes) != n:
            return None
        video_idx = max(range(len(shapes)), key=lambda i: (len(tuple(shapes[i])), sizes[i]))
        return sum(sizes[:video_idx]), sizes[video_idx], tuple(shapes[video_idx])
    except Exception:
        return None


# Per-scene model_function_wrappers (embed guidance / score slider / output guidance /
# temporal styles) live on the SHARED ModelPatcher. If a run dies between install and
# restore (interrupt, OOM), the wrapper survives in-process and silently steers every
# later run, stacking one deeper each time — the exact progressive-corruption failure
# mode strip_funpack_block_hooks exists for, so we reuse its tag+strip pattern.
_FUNPACK_SCENE_WRAPPER_TAG = "_funpack_scene_wrapper"


def _make_steer_ramp(sigmas, h3):
    """The late-step gate every rating-driven wrapper shares, as a function of sigma.

    The historical gate is ``max(0, 1 - 2*sigma)``: full authority at sigma 0, nothing
    above 0.5. That reads sigma as "how far through the schedule are we", which holds on
    LTX's hand-authored schedules and is FALSE on H3. H3's schedules are
    ``sigma = shift*t / (1 + (shift-1)*t)`` over uniform t, so a large shift keeps sigma
    high right up to the final leap. Measured coverage of the old gate:

        shift 6 / 4 steps (turbo)   0 of 4 steps
        shift 12 / 12 (H3 default)  0 of 12
        shift 3 / 20                4 of 20, the first two at gate 0.14 and 0.31

    So embed guidance, score_slider, DynaShift and output_guidance were inert or nearly
    inert on H3 -- silently, since each reported itself as active.

    The fix reads the position on the underlying uniform base grid instead, recovered from
    the schedule itself rather than from a shift constant (the shift is only reliable when
    MiniMaxH3SigmaShift is wired, and video sampling does not require that node). For a
    shift-generated schedule the base grid IS the step position, so the gate becomes
    ``max(0, 2k/n - 1)`` -- the same "last half, ramping to full" intent the constant was
    written to express, at any shift and any step count.

    LTX is untouched: it keeps the absolute-sigma gate it was validated with.
    """
    def legacy(sigma):
        return max(0.0, 1.0 - float(sigma) * 2.0)

    if not h3 or sigmas is None:
        return legacy
    try:
        vals = [float(v) for v in sigmas.flatten().tolist()]
    except Exception:
        return legacy
    n = len(vals) - 1
    if n < 1:
        return legacy

    def progress(sigma):
        sigma = float(sigma)
        k = min(range(len(vals)), key=lambda i: abs(vals[i] - sigma))
        return max(0.0, 2.0 * (k / n) - 1.0)

    return progress


def _steer_ramp_coverage(ramp_fn, sigmas):
    """(gated steps, total steps, peak gate) for a schedule -- so a run can say out loud
    how much of it the steering actually reaches instead of only that it is 'active'."""
    try:
        vals = [float(v) for v in sigmas.flatten().tolist()][:-1]  # sigma[-1] is terminal
    except Exception:
        return None
    if not vals:
        return None
    gates = [ramp_fn(v) for v in vals]
    return sum(1 for g in gates if g > 0.0), len(gates), max(gates) if gates else 0.0



def _tag_scene_wrapper(wrapper, prev):
    """Mark a per-scene wrapper (and remember what it wrapped) so a later run can
    identify and unwind leaked ones."""
    setattr(wrapper, _FUNPACK_SCENE_WRAPPER_TAG, True)
    setattr(wrapper, "_funpack_prev_wrapper", prev)
    return wrapper


def _tag_funpack_hook(fn):
    """Mark a forward hook as FunPack's, so the run-start sweep can remove one that leaked.

    Anything installed on the shared diffusion model needs this: the sweep only removes
    what it can prove is ours, so an untagged hook is one nothing will ever clean up.
    """
    try:
        from .ltx_enhancements import _FUNPACK_HOOK_TAG
    except ImportError:
        from ltx_enhancements import _FUNPACK_HOOK_TAG
    setattr(fn, _FUNPACK_HOOK_TAG, True)
    return fn


def _strip_funpack_scene_wrappers(model):
    """Unwind FunPack per-scene wrappers leaked by a previous interrupted/failed run.
    Walks the recorded prev-wrapper chain back to the first non-FunPack wrapper (or
    none). Idempotent; never raises."""
    try:
        w = model.model_options.get("model_function_wrapper")
        stripped = 0
        while w is not None and getattr(w, _FUNPACK_SCENE_WRAPPER_TAG, False):
            w = getattr(w, "_funpack_prev_wrapper", None)
            stripped += 1
        if stripped:
            if w is not None:
                model.model_options["model_function_wrapper"] = w
            else:
                model.model_options.pop("model_function_wrapper", None)
            print(f"[FunPackSceneChain] Stripped {stripped} leaked scene wrapper(s) from a previous run")
        return stripped
    except Exception:
        return 0


def _alg_blur_frames(model, latent_image, kappa, frame_indices=(), tail_count=0):
    """ALG (arXiv:2506.08456): low-pass filter selected frames of the packed video stream.

    I2V models over-expose the sharp anchor frame from step 0, which lets the model take a
    shortcut to a near-static video that just matches the reference. Bilinear downsample-then-
    upsample of the anchor latent (selected frames only, video stream only) at the configured
    factor removes the high-frequency content the shortcut needs; ALG's own per-step schedule
    then swaps this blurred copy back to the sharp original once sigma drops past the threshold.

    `frame_indices` is any iterable of frame indices within the video stream's T dimension to
    blur (each filtered independently) — frame 0 for the genuine i2v anchor, and/or the trailing
    indices of newly-appended guide-attention frames (mid_scene_guide / carry_i2v_guides-as-guide /
    configured guides / JoyAI memory — see EXPERIMENTAL alg_blur_guides) when extending the same
    idea to those. Indices outside the actual frame count are silently skipped.

    Returns a new tensor with the same shape as latent_image, or None on any failure/mismatch
    (caller then leaves ALG off — no anchor, no packed video stream, or unexpected layout).
    """
    if latent_image is None:
        return None
    try:
        shapes = _get_latent_shapes(model)
        if not shapes:
            return None
        video_idx = max(range(len(shapes)), key=lambda i: (len(tuple(shapes[i])), int(math.prod(tuple(shapes[i])[1:]))))
        video_shape = tuple(shapes[video_idx])
        if len(video_shape) != 5:
            return None  # expects [B, C, T, H, W]
        b, c, t, h, w = (int(d) for d in video_shape)
        sizes = [int(math.prod(tuple(s)[1:])) for s in shapes]
        if sum(sizes) != int(latent_image.shape[-1]):
            return None  # packed layout doesn't match our assumption -> don't risk it
        idxs = {int(i) for i in frame_indices if 0 <= int(i) < t}
        if tail_count > 0:
            idxs |= set(range(max(0, t - int(tail_count)), t))
        idxs = sorted(idxs)
        if not idxs:
            return None
        off = sum(sizes[:video_idx])
        sz = sizes[video_idx]
        video = latent_image[..., off:off + sz].reshape(b, c, t, h, w)
        dh, dw = max(1, round(h / kappa)), max(1, round(w / kappa))
        video_blurred = video.clone()
        for idx in idxs:
            frame = video[:, :, idx]  # [B, C, H, W]
            down = torch.nn.functional.interpolate(frame, size=(dh, dw), mode="bilinear", align_corners=False)
            blurred = torch.nn.functional.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
            video_blurred[:, :, idx] = blurred.to(video.dtype)
        out = latent_image.clone()
        out[..., off:off + sz] = video_blurred.reshape(b, 1, sz)
        return out
    except Exception as _e:
        _log.failed("FunPackSceneChain", "ALG anchor blur", _e,
                    "the step runs on the sharp anchor — ALG de-staticking is inert")
        return None


def _alg_prepare(model, extra_args, anchor_on, anchor_strength, tail_frames, tail_strength):
    """Precompute the ALG latent variants for this run.

    Returns ``(sharp, latents, anchor_on, tail_on)``. `latents` is keyed by
    ``(anchor blurred?, tail blurred?)`` — one entry per combination that can occur; the two
    frame sets are disjoint, so the both-blurred variant just composes the two. Anchor and
    guide-tail blur are fully independent (own strength, own sigma window), which is why the
    key is a pair rather than a single flag. Both returned flags are False when ALG cannot
    run at all: no real anchor (no denoise_mask), no latent_image, or a packed layout
    `_alg_blur_frames` refuses to guess at.

    Shared by the in-loop version (Distilled Flow) and the denoiser-proxy version that gives
    every other sampler the same behaviour — one place, so the two can't drift.
    """
    tail_frames = max(0, int(tail_frames))
    anchor_on = bool(anchor_on)
    tail_on = tail_frames > 0
    sharp = getattr(model, "latent_image", None) if (anchor_on or tail_on) else None
    if not (anchor_on or tail_on) or sharp is None or (extra_args or {}).get("denoise_mask") is None:
        return sharp, {}, False, False

    anchor_blurred = _alg_blur_frames(
        model, sharp, max(1.0, float(anchor_strength)), frame_indices=(0,),
    ) if anchor_on else None
    tail_kappa = max(1.0, float(tail_strength))
    tail_blurred = _alg_blur_frames(
        model, sharp, tail_kappa, tail_count=tail_frames,
    ) if tail_on else None
    both_blurred = _alg_blur_frames(
        model, anchor_blurred, tail_kappa, tail_count=tail_frames,
    ) if (anchor_blurred is not None and tail_blurred is not None) else None
    anchor_on = anchor_on and anchor_blurred is not None
    tail_on = tail_on and tail_blurred is not None
    latents = {
        (False, False): sharp,
        (True, False): anchor_blurred,
        (False, True): tail_blurred,
        # both → anchor → tail → None. Explicit None checks, NOT `anchor_blurred or
        # tail_blurred`: `or` calls bool() on the first operand, which is a multi-element
        # tensor when anchor blur is on but tail blur is off (both_blurred is then None) —
        # "Boolean value of Tensor is ambiguous". This value is only a defensive fallback:
        # when (True, True) is actually indexed, anchor_on and tail_on are both True, so
        # both_blurred is non-None and the chain never reaches the singles.
        (True, True): (both_blurred if both_blurred is not None
                       else anchor_blurred if anchor_blurred is not None
                       else tail_blurred),
    }
    return sharp, latents, anchor_on, tail_on


class _ALGDenoiser:
    """Denoiser proxy that runs ALG's blurred/sharp swap for a sampler we can't get inside.

    ALG is a per-step swap of the pinned latent_image, and the swap is decided by ONE thing:
    the sigma of the step. That sigma is an argument of every model call, so the schedule
    does not need the sampler's loop at all — it only needed it in the original implementation
    because that is where the code happened to live. Driving it from here makes ALG work with
    a stock KSampler (any sampler_name), with Hybrid Euler 2S, and with multi-eval samplers
    such as heun, where each evaluation gets the anchor its own sigma calls for.

    Transparent for everything else: attribute access forwards to the wrapped denoiser.
    """

    _OWN = ("_inner", "_latents", "_anchor_on", "_tail_on", "_anchor_thr", "_tail_thr")

    def __init__(self, inner, latents, anchor_on, tail_on, anchor_thr, tail_thr):
        self._inner = inner
        self._latents = latents
        self._anchor_on = anchor_on
        self._tail_on = tail_on
        self._anchor_thr = float(anchor_thr)
        self._tail_thr = float(tail_thr)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        if name in _ALGDenoiser._OWN:
            object.__setattr__(self, name, value)
        else:                       # e.g. samplers assigning model.latent_image
            setattr(self._inner, name, value)

    def __call__(self, x, sigma, **kwargs):
        try:
            s = float(sigma.flatten()[0]) if hasattr(sigma, "flatten") else float(sigma)
            self._inner.latent_image = self._latents[(
                self._anchor_on and s > self._anchor_thr,
                self._tail_on and s > self._tail_thr,
            )]
        except Exception:
            pass                    # never cost the step over the guidance
        return self._inner(x, sigma, **kwargs)


def _alg_wrap_sampler(sampler, anchor_on, anchor_strength, anchor_threshold,
                      tail_frames, tail_strength, tail_threshold):
    """A SAMPLER equivalent to `sampler` with ALG applied around its model.

    Returns None when the sampler exposes no `sampler_function` to wrap — some SAMPLER
    producers are not KSAMPLER-shaped, and guessing at their internals is not worth it.
    """
    fn = getattr(sampler, "sampler_function", None)
    if fn is None:
        return None

    def _wrapped(model, x, sigmas, extra_args=None, callback=None, disable=None, **options):
        sharp, latents, a_on, t_on = _alg_prepare(
            model, extra_args, anchor_on, anchor_strength, tail_frames, tail_strength)
        if not (a_on or t_on):     # no anchor, or a layout we won't guess at — run untouched
            return fn(model, x, sigmas, extra_args=extra_args, callback=callback,
                      disable=disable, **options)
        print(f"[FunPack AV] ALG on (anchor: "
              f"{f'strength={anchor_strength}, sigma_threshold={anchor_threshold}' if a_on else 'off'}; "
              f"guide tail: "
              f"{f'{tail_frames} frame(s), strength={tail_strength}, sigma_threshold={tail_threshold}' if t_on else 'off'}) "
              f"— driven from outside {getattr(fn, '__name__', 'the sampler')}'s loop, off the "
              f"sigma of each model call")
        proxy = _ALGDenoiser(model, latents, a_on, t_on, anchor_threshold, tail_threshold)
        try:
            return fn(proxy, x, sigmas, extra_args=extra_args, callback=callback,
                      disable=disable, **options)
        finally:
            model.latent_image = sharp

    _wrapped.__name__ = getattr(fn, "__name__", "sampler") + "_alg"
    return comfy.samplers.KSAMPLER(
        _wrapped,
        extra_options=getattr(sampler, "extra_options", None) or {},
        inpaint_options=getattr(sampler, "inpaint_options", None) or {},
    )


class _SharpenDenoiser:
    """Denoiser proxy that runs the quality-sharpness unsharp for a sampler we can't get inside.

    Same mechanism as the in-loop version on Hybrid Euler 2S / Distilled Flow: boost the
    x0 prediction's high-frequency component against the previous prediction. Nothing about
    it needs the sampler's loop — it reads only the current denoised, the previous one, and
    the step's sigma, and sigma is an argument of every model call. So a stock KSampler
    (euler, res_multistep, dpmpp_2m, …) can have the detail recovery that until now only
    FunPack's own samplers had.

    Two honest differences from the in-loop version:

    * On a multi-eval sampler (heun, dpmpp_2m_sde, …) "previous prediction" means the
      previous *evaluation*, which may be the second half of the same step rather than the
      step before. The high-pass is still a high-pass; its magnitude is not identical.
    * `prev` holds the SHARPENED result, matching the in-loop samplers, where
      `prev_denoised = denoised` runs after the sharpen. That makes the boost mildly
      self-limiting instead of compounding.

    Transparent for everything else: attribute access forwards to the wrapped denoiser, so
    this composes with the ALG proxy in either order.
    """

    _OWN = ("_inner", "_amount", "_thr", "_mask", "_prev", "_mask_done")

    def __init__(self, inner, amount, threshold, mask):
        self._inner = inner
        self._amount = float(amount)
        self._thr = float(threshold)
        self._mask = mask
        self._prev = None

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        if name in _SharpenDenoiser._OWN:
            object.__setattr__(self, name, value)
        else:                       # e.g. samplers (or the ALG proxy) assigning latent_image
            setattr(self._inner, name, value)

    def __call__(self, x, sigma, **kwargs):
        denoised = self._inner(x, sigma, **kwargs)
        try:
            s = float(sigma.flatten()[0]) if hasattr(sigma, "flatten") else float(sigma)
        except Exception as _e:  # noqa: BLE001
            # Once, not per evaluation — but never nothing. Silently returning here would
            # leave quality_sharpness switched on, reported as active, and doing nothing for
            # the whole run.
            _log.failed("FunPackStudio", "quality sharpness (unreadable sigma)", _e,
                        "this sampler's steps are NOT being sharpened")
            return denoised
        if s > self._thr:
            # Outside the window: nothing to sharpen, but the prediction is still KEPT. The
            # in-loop samplers carry prev_denoised across the phase boundary, so the first
            # quality step sharpens against the last pre-quality one; dropping it here would
            # make the same setting mean something different on a KSampler. The cost is one
            # latent (~40 MB on a 97-frame 768x512 H3 clip) for the length of the run, which
            # is not a memory problem worth breaking parity over.
            try:
                self._prev = denoised.detach()
            except Exception:  # noqa: BLE001
                self._prev = None
            return denoised
        try:
            denoised = _video_only(
                _apply_quality_sharpness(denoised, self._prev, self._amount),
                denoised, self._mask)
        except Exception as _e:  # noqa: BLE001
            _log.failed("FunPackStudio", "quality sharpness", _e,
                        "this evaluation keeps the unsharpened prediction")
        try:
            self._prev = denoised.detach()
        except Exception:  # noqa: BLE001
            self._prev = None
        return denoised

    def release(self):
        """Drop the retained prediction. Called when the sampler returns, so the tensor is
        gone before the decode asks for its peak rather than waiting on garbage collection."""
        self._prev = None
        self._mask = None


def _sharpen_wrap_sampler(sampler, sharpness, start_pct):
    """A SAMPLER equivalent to `sampler` with quality sharpness applied around its model.

    Returns `sampler` unchanged when sharpness is off, and None when the sampler exposes no
    `sampler_function` to wrap.
    """
    sharpness = max(0.0, min(1.0, float(sharpness or 0.0)))
    if sharpness <= 0.0:
        return sampler
    fn = getattr(sampler, "sampler_function", None)
    if fn is None:
        return None

    def _wrapped(model, x, sigmas, extra_args=None, callback=None, disable=None, **options):
        # The window is a fraction of the schedule, converted to the sigma that starts it —
        # exactly how Hybrid Euler 2S derives its quality phase from high_quality_pct, so the
        # same number means the same thing on both samplers.
        try:
            sched_steps = max(1, int(sigmas.shape[0]) - 1)
            late_start = _get_late_start_index(sched_steps, start_pct)
            thr = float(sigmas[late_start].item()) if late_start < int(sigmas.shape[0]) else None
        except Exception:
            thr = None
        if thr is None:
            return fn(model, x, sigmas, extra_args=extra_args, callback=callback,
                      disable=disable, **options)
        mask = _packed_video_mask(model, x)
        print(f"[FunPack] quality sharpness {sharpness:.2f} on the last "
              f"{start_pct * 100:.0f}% of the schedule (sigma <= {thr:.4f}) — driven from "
              f"outside {getattr(fn, '__name__', 'the sampler')}'s loop, off the sigma of "
              f"each model call"
              f"{'' if mask is not None else '; single-stream latent, no audio to protect'}")
        proxy = _SharpenDenoiser(model, sharpness, thr, mask)
        try:
            return fn(proxy, x, sigmas, extra_args=extra_args, callback=callback,
                      disable=disable, **options)
        finally:
            proxy.release()

    _wrapped.__name__ = getattr(fn, "__name__", "sampler") + "_sharpen"
    return comfy.samplers.KSAMPLER(
        _wrapped,
        extra_options=getattr(sampler, "extra_options", None) or {},
        inpaint_options=getattr(sampler, "inpaint_options", None) or {},
    )


class _JoyAIMemoryBank:
    """JoyAI-Echo cross-shot memory: a rolling set of clean prior-shot latent frames.

    Mirrors PairedAudioVideoMemoryBank's policy (JoyAI_Echo configs/inference.yaml): the first
    ``num_fix`` entries are pinned permanently (the opening shots = global story anchor); the rest
    is a most-recent window so the total never exceeds ``max_size``. Each entry is a paired
    (video_frame, audio_frame) of clean latents from a finished scene; the audio half is None until
    joyai_audio_memory is on. Pure bookkeeping — injection is done by the sampler (video via LTX
    guide attention, audio via a protected prefix), so this holds no model state and is unit-testable.
    """

    def __init__(self, max_size=7, num_fix=3):
        self.max_size = max(1, int(max_size))
        self.num_fix = max(0, min(int(num_fix), self.max_size))
        self.entries = []  # list of (video_frame, audio_frame) tuples, oldest first

    def add(self, video_frame, audio_frame=None):
        if video_frame is None:
            return
        self.entries.append((video_frame, audio_frame))
        if len(self.entries) <= self.max_size:
            return
        fixed = self.entries[:self.num_fix]
        recent = self.entries[self.num_fix:]
        room = self.max_size - len(fixed)
        self.entries = fixed + (recent[-room:] if room > 0 else [])

    def frames(self):
        """Video latent frames, oldest first (the injection order for guide attention)."""
        return [v for v, _ in self.entries]

    def audio(self):
        """Paired audio latent frames aligned 1:1 with frames(); entries hold None when audio is off."""
        return [a for _, a in self.entries]


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


def _sample_const_rf_full(model, x, sigmas, extra_args, callback, disable,
                          eta, s_noise, eta_final, high_quality_pct, correction_blend,
                          motion_pulse_mode, motion_pulse_start_pct, motion_pulse_count,
                          motion_pulse_spacing_pct, motion_pulse_strength,
                          motion_pulse_noise, motion_pulse_steps,
                          velocity_bias_mode, velocity_bias_strength,
                          velocity_refinement_key,
                          rescue_mode, rescue_threshold, rescue_strength, rescue_prompt_sig,
                          quality_sharpness=0.0, velocity_bias_source="mean",
                          normalize_strength=0.0, normalize_start_sigma=0.9,
                          h3_audio_clock=False):
    """Full-feature rectified-flow sampler for CONST models (LTXAV).

    Rectified-flow-correct port of the hybrid sampler so its features actually run on
    LTXAV (which stock comfy routes to euler_ancestral_RF, bypassing the custom loop):
      - early phase: ancestral RF euler with eta decay, AB2 (order-2) denoised
        extrapolation, and anti-stiffness motion pulses
      - late (quality) phase: deterministic Heun corrector — the RF-correct 2nd order;
        DPM++ 2S is eps-only and intentionally not used here — with the hybrid's
        progressive correction_blend schedule
      - velocity-bias capture/apply and reactive rescue around every model eval
    With motion/velocity/rescue off AND high_quality_pct=0 it reduces to stock
    euler_ancestral_RF. The base RF step matches comfy.sample_euler_ancestral_RF.
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = k_diffusion_sampling.default_noise_sampler(x, seed=seed)
    try:
        s_noise = s_noise * getattr(
            model.inner_model.model_patcher.get_model_object('model_sampling'), "noise_scale", 1.0)
    except Exception:
        pass

    total_steps = max(0, len(sigmas) - 1)
    if total_steps <= 0:
        return x

    high_quality_pct = max(0.0, min(1.0, float(high_quality_pct)))
    correction_blend = max(0.0, min(1.0, float(correction_blend)))
    eta_final = max(0.0, min(float(eta), float(eta_final)))

    sched_steps = max(1, int(len(sigmas)) - 1)

    if not motion_pulse_steps:
        _, _, motion_pulse_steps, motion_pulse_noise = _prepare_dynamic_sigmas(
            sigmas, high_quality_pct, motion_pulse_mode, motion_pulse_start_pct,
            motion_pulse_count, motion_pulse_spacing_pct, motion_pulse_strength)
    motion_pulse_noise = max(0.0, float(motion_pulse_noise or 0.0))
    motion_step_noise = {
        int(item.get("step_index", -1)): max(0.0, float(item.get("noise", motion_pulse_noise)))
        for item in (motion_pulse_steps or []) if isinstance(item, dict)
    }

    late_start = _get_late_start_index(sched_steps, high_quality_pct)
    quality_sigma_start = float(sigmas[late_start].item()) if late_start < sigmas.shape[0] else None
    num_quality_steps = sched_steps - late_start

    s_in = x.new_ones([x.shape[0]])
    prev_denoised = None
    prev_h = None
    _kvlock_state = {}

    # Audio-safe sampling: on a packed LTXAV latent, keep ancestral noise + steering on the
    # video stream and let audio ride the clean deterministic flow (ancestral re-noising
    # corrupts audio). None for single-stream LTXV -> all of this is a no-op.
    video_mask = _packed_video_mask(model, x)
    if video_mask is not None:
        n_aud = int((video_mask < 0.5).sum().item())
        print(f"[FunPack AV] packed audio+video latent detected -> audio-safe sampling "
              f"(audio held deterministic on {n_aud} of {video_mask.shape[-1]} packed dims)")

    normalize_strength = max(0.0, min(1.0, float(normalize_strength)))
    ref_scale = [None]  # video-only latent-normalization reference (anti-overbake), opt-in
    if normalize_strength > 0.0:
        print(f"[FunPack AV] latent normalization on (strength={normalize_strength}, "
              f"start_sigma={normalize_start_sigma}, video-only)")

    audio_clock = _audio_clock_setup(model, x, sigmas, h3_audio_clock)

    for i in comfy.utils.model_trange(total_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        in_quality = quality_sigma_start is not None and float(sigma.item()) <= quality_sigma_start
        velocity_target = _velocity_bias_target(sigmas, sigma)

        if not in_quality:
            pulse = motion_step_noise.get(int(i), 0.0)
            if pulse > 0.0:
                x = _video_only(_apply_motion_pulse(x, sigma, sigma_next, pulse, noise_sampler), x, video_mask)
                prev_denoised = None
                prev_h = None

        if _velocity_bias_enabled(velocity_bias_mode, "apply"):
            x_pre = x
            x = _apply_velocity_bias(x, velocity_refinement_key, velocity_target, velocity_bias_strength,
                                     sigma_ratio=_sigma_ratio(sigmas, sigma),
                                     prompt_sig=rescue_prompt_sig, source=velocity_bias_source)
            x = _video_only(x, x_pre, video_mask)

        denoised = model(x, sigma * s_in, **extra_args)
        _kvlock_schedule(model, denoised, prev_denoised, video_mask, _kvlock_state)

        if rescue_mode or _velocity_bias_enabled(velocity_bias_mode, "capture"):
            _capture_velocity_bias(velocity_refinement_key, velocity_target, x, sigma, denoised, prompt_sig=rescue_prompt_sig)
        if rescue_mode and velocity_target is not None:
            denoised = _video_only(_rescue_denoised(
                denoised, x, sigma, velocity_refinement_key,
                velocity_target, rescue_threshold, rescue_strength, prompt_sig=rescue_prompt_sig,
                source=velocity_bias_source,
            ), denoised, video_mask)
        # E: restore high-frequency detail lost to the velocity-bias mean-pull (quality phase only).
        if in_quality:
            denoised = _video_only(_apply_quality_sharpness(denoised, prev_denoised, quality_sharpness), denoised, video_mask)

        # Video-only latent normalization (opt-in, anti-overbake) — stacks on top of the RF loop.
        denoised = _normalize_video_denoised(
            denoised, video_mask, sigma, ref_scale, normalize_strength, normalize_start_sigma,
        )

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})

        h = float((sigma - sigma_next).abs().item())
        # AB2 (order-2) extrapolation on the x0/denoised sequence — RF-valid.
        if prev_denoised is not None and prev_h is not None and prev_h > 1e-7 and h > 1e-7:
            r = max(0.1, min(5.0, h / prev_h))
            try:
                denoised_eff = (1.0 + r / 2.0) * denoised - (r / 2.0) * prev_denoised.to(device=denoised.device, dtype=denoised.dtype)
            except Exception:
                denoised_eff = denoised
        else:
            denoised_eff = denoised
        # Audio rides plain euler — exclude it from AB2 (2nd-order extrapolation corrupts audio).
        denoised_eff = _video_only(denoised_eff, denoised, video_mask)
        prev_denoised = denoised.detach()
        prev_h = h

        if sigma_next == 0:
            # Identical displacement to the euler-RF step below at sigma_next = 0, so the
            # audio clock corrects this final step like any other.
            x = _audio_clock_step(denoised_eff, x, audio_clock, i)
            continue

        if in_quality:
            # Deterministic Heun corrector (RF-correct 2nd order). Progressive blend:
            # first half of quality steps = euler, second half = full Heun, matching
            # the hybrid sampler's correction schedule. Note: to_d-based euler is
            # identical to the RF flow update, so this is consistent with the early phase.
            if num_quality_steps <= 1:
                effective_blend = correction_blend
            else:
                q_idx = max(0, i - late_start)
                effective_blend = 0.0 if q_idx < (num_quality_steps // 2) else correction_blend
            dt = sigma_next - sigma
            d1 = k_diffusion_sampling.to_d(x, sigma, denoised_eff)
            if effective_blend > 0.0:
                # Audio in the predicted state gets the clock too: the corrector's audio
                # output is discarded, but video and audio share one packed sequence, so
                # an over-stepped audio half skews the video correction via joint attention.
                x_pred = _audio_clock_step(x + d1 * dt, x, audio_clock, i)
                denoised_pred = model(x_pred, sigma_next * s_in, **extra_args)
                d2 = k_diffusion_sampling.to_d(x_pred, sigma_next, denoised_pred)
                d_use = d1 + effective_blend * ((d1 + d2) * 0.5 - d1)
            else:
                d_use = d1
            # Audio rides the plain euler direction (d1); only video gets the Heun correction.
            d_use = _video_only(d_use, d1, video_mask)
            x = _audio_clock_step(x + d_use * dt, x, audio_clock, i)
            # Heun changed x with a corrected direction; invalidate AB2 history.
            prev_denoised = None
            prev_h = None
        else:
            # Early phase: ancestral RF euler (matches sample_euler_ancestral_RF), with
            # eta decay toward the quality boundary, using the AB2 denoised estimate.
            effective_eta = _effective_eta(eta, eta_final, sigmas, sigma)
            # Deterministic euler-RF step to sigma_next (this is what audio rides — no
            # ancestral re-noising). For single-stream video (mask None) it is only used
            # as the eta==0 fallback; the full ancestral result is kept for video.
            er = sigma_next / sigma
            x_det = er * x + (1 - er) * denoised_eff
            if effective_eta > 0:
                downstep_ratio = 1 + (sigma_next / sigma - 1) * effective_eta
                sigma_down = sigma_next * downstep_ratio
                alpha_ip1 = 1 - sigma_next
                alpha_down = 1 - sigma_down
                sigma_down_i_ratio = sigma_down / sigma
                x_anc = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised_eff
                renoise_coeff = (sigma_next ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2) ** 0.5
                x_anc = (alpha_ip1 / alpha_down) * x_anc + noise_sampler(sigma, sigma_next) * s_noise * renoise_coeff
                # Video gets full ancestral noise; audio stays on the deterministic step.
                # The clock is applied to the combined result: its audio half is exactly
                # x_det's (the noise above is video-only), so it scales the deterministic
                # audio displacement and leaves the noised video untouched.
                x = _audio_clock_step(_video_only(x_anc, x_det, video_mask), x, audio_clock, i)
            else:
                x = _audio_clock_step(x_det, x, audio_clock, i)

    return x


def sample_funpack_hybrid_euler_2s(model, x, sigmas, extra_args=None, callback=None,
                                   disable=None, eta=1.0, s_noise=1.0,
                                   high_quality_pct=0.35, correction_blend=1.0,
                                   quality_sharpness=0.0,
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
                                   velocity_bias_source="mean",
                                   velocity_refinement_key="default",
                                   rescue_mode=False,
                                   rescue_threshold=0.15,
                                   rescue_strength=0.2,
                                   rescue_prompt_sig=None,
                                   eta_final=1.0,
                                   normalize_strength=0.0,
                                   normalize_start_sigma=0.9,
                                   h3_audio_clock=False):
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
    # Normalize velocity/rescue params up front — both the CONST (rectified-flow)
    # branch and the main hybrid loop rely on them.
    velocity_bias_mode = (velocity_bias_mode or "off").lower()
    if velocity_bias_mode not in VELOCITY_BIAS_MODES:
        velocity_bias_mode = "off"
    velocity_bias_strength = max(0.0, min(3.0, float(velocity_bias_strength or 0.0)))
    velocity_bias_source = (velocity_bias_source or "mean").lower()
    if velocity_bias_source not in ("mean", "nearest"):
        velocity_bias_source = "mean"
    rescue_mode = bool(rescue_mode)
    rescue_threshold = max(0.0, min(1.0, float(rescue_threshold or 0.0)))
    rescue_strength = max(0.0, min(0.5, float(rescue_strength or 0.0)))
    rescue_prompt_sig = _normalize_sig(rescue_prompt_sig)  # capture also uses this
    if rescue_mode:
        _RESCUE_LOG["warned_no_memory"] = False
        _RESCUE_LOG["warned_no_prompt_match"] = False
        _RESCUE_LOG["fired"] = 0

    if bool(rescue_mode):
        # One-shot startup diagnostic so it is unambiguous why rescue does or does
        # not fire: confirms the option reached the sampler, whether this model
        # bypasses the custom loop (CONST), and which steps are eligible.
        try:
            _is_const = isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST)
        except Exception:
            _is_const = "unknown"
        try:
            _ts = max(0, len(sigmas) - 1)
            _elig = [(i, round(float(sigmas[i].item()), 4)) for i in range(_ts)
                     if _velocity_bias_target(sigmas, sigmas[i]) is not None]
        except Exception:
            _elig = []
        print(f"[FunPack rescue] requested — strength={float(rescue_strength)}, "
              f"threshold={float(rescue_threshold)}, "
              f"prompt_sig={'yes' if isinstance(rescue_prompt_sig, torch.Tensor) else 'no'}, "
              f"model_is_CONST={_is_const}, eligible_steps={_elig}")
        if _is_const is True:
            print("[FunPack rescue] CONST model -> using rectified-flow rescue path (velocity/rescue active).")
        if not _elig:
            print("[FunPack rescue] WARNING: no sigma step matches a velocity target "
                  "(normalized ~0.90/0.80 +/-0.065) -> rescue/capture cannot fire on this schedule.")

    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        # CONST (rectified-flow, e.g. LTXAV): stock comfy routes to euler_ancestral_RF,
        # bypassing the entire custom loop. Run the RF-correct full-feature port so the
        # hybrid sampler's behavior (motion pulses, order-2, quality correction, eta
        # decay, velocity bias, rescue) actually applies on LTXAV.
        return _sample_const_rf_full(
            model, x, sigmas, extra_args, callback, disable, eta, s_noise, eta_final,
            high_quality_pct, correction_blend,
            motion_pulse_mode, motion_pulse_start_pct, motion_pulse_count,
            motion_pulse_spacing_pct, motion_pulse_strength,
            motion_pulse_noise, motion_pulse_steps,
            velocity_bias_mode, velocity_bias_strength,
            velocity_refinement_key,
            rescue_mode, rescue_threshold, rescue_strength, rescue_prompt_sig,
            quality_sharpness=quality_sharpness,
            velocity_bias_source=velocity_bias_source,
            normalize_strength=normalize_strength,
            normalize_start_sigma=normalize_start_sigma,
            h3_audio_clock=h3_audio_clock,
        )

    # Below here is the eps-parameterised loop, which MiniMax H3 (CONST) never reaches —
    # h3_audio_clock is deliberately not wired into it rather than added as dead code.
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = k_diffusion_sampling.default_noise_sampler(x, seed=seed)

    total_steps = max(0, len(sigmas) - 1)
    if total_steps <= 0:
        return x

    high_quality_pct = max(0.0, min(1.0, float(high_quality_pct)))
    correction_blend = max(0.0, min(1.0, float(correction_blend)))
    eta_final = max(0.0, min(float(eta), float(eta_final)))

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
    _kvlock_state = {}
    video_mask = _packed_video_mask(model, x)

    for i in comfy.utils.model_trange(total_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        in_quality_phase = quality_sigma_start is not None and float(sigma.item()) <= quality_sigma_start
        # Velocity-bias capture and rescue must run wherever a velocity target lands,
        # regardless of phase. With short distilled schedules the only target step
        # (norm sigma ~0.9) often falls inside the quality phase, so gating these to
        # the early branch alone would silently disable them.
        velocity_target = _velocity_bias_target(sigmas, sigma)

        if not in_quality_phase:
            # Adaptive eta: decay from eta toward eta_final as sigma
            # approaches the quality boundary.
            effective_eta = _effective_eta(eta, eta_final, sigmas, sigma)

            pulse_noise = motion_step_noise.get(int(i), 0.0)
            if pulse_noise > 0.0:
                x = _apply_motion_pulse(x, sigma, sigma_next, pulse_noise, noise_sampler)
                # Pulse modifies x; previous denoised is no longer a valid
                # second-order estimate for the next step.
                prev_denoised = None
                prev_h = None

            if _velocity_bias_enabled(velocity_bias_mode, "apply"):
                x = _apply_velocity_bias(x, velocity_refinement_key, velocity_target, velocity_bias_strength,
                                         sigma_ratio=_sigma_ratio(sigmas, sigma),
                                     prompt_sig=rescue_prompt_sig, source=velocity_bias_source)
            denoised = model(x, sigma * s_in, **extra_args)
            _kvlock_schedule(model, denoised, prev_denoised, video_mask, _kvlock_state)
            if rescue_mode or _velocity_bias_enabled(velocity_bias_mode, "capture"):
                _capture_velocity_bias(velocity_refinement_key, velocity_target, x, sigma, denoised, prompt_sig=rescue_prompt_sig)
            if rescue_mode and velocity_target is not None:
                denoised = _rescue_denoised(
                    denoised, x, sigma, velocity_refinement_key,
                    velocity_target, rescue_threshold, rescue_strength, prompt_sig=rescue_prompt_sig,
                    source=velocity_bias_source,
                )

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

            if _velocity_bias_enabled(velocity_bias_mode, "apply"):
                x = _apply_velocity_bias(x, velocity_refinement_key, velocity_target, velocity_bias_strength,
                                         sigma_ratio=_sigma_ratio(sigmas, sigma),
                                     prompt_sig=rescue_prompt_sig, source=velocity_bias_source)
            denoised = model(x, sigma * s_in, **extra_args)
            _kvlock_schedule(model, denoised, prev_denoised, video_mask, _kvlock_state)
            if rescue_mode or _velocity_bias_enabled(velocity_bias_mode, "capture"):
                _capture_velocity_bias(velocity_refinement_key, velocity_target, x, sigma, denoised, prompt_sig=rescue_prompt_sig)
            if rescue_mode and velocity_target is not None:
                denoised = _rescue_denoised(
                    denoised, x, sigma, velocity_refinement_key,
                    velocity_target, rescue_threshold, rescue_strength, prompt_sig=rescue_prompt_sig,
                    source=velocity_bias_source,
                )
            # E: restore high-frequency detail lost to the velocity-bias mean-pull.
            denoised = _apply_quality_sharpness(denoised, prev_denoised, quality_sharpness)

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
                "quality_sharpness": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Restores fine detail lost to velocity-bias softening. Temporal-average unsharp on the x0 prediction during the quality phase only. 0 disables; 0.2-0.4 typical when velocity bias is on. Free (no extra model eval)."
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
                    "max": 3.0,
                    "step": 0.05,
                    "tooltip": "Strength of the remembered velocity (action) injected at the structure sigma. 0 disables. ~0.15 = subtle spice; 0.3-1.0 = clear action crossover; 2-3 approaches full action replacement (capped so the current gen isn't wiped). Creative tool, not for consistency."
                }),
                "velocity_bias_source": (["mean", "nearest"], {
                    "default": "mean",
                    "tooltip": "How velocity bias / rescue pick a good direction. 'mean' = prompt-blind global average (legacy). 'nearest' = single best-matching prompt cluster — preserves one real good gen's detail instead of a washed-out average (less softening). Affects both apply and rescue."
                }),
                "velocity_refinement_key": ("STRING", {
                    "default": "default",
                    "multiline": False,
                    "tooltip": "Memory key used to capture/apply early velocity bias."
                }),
                "rescue_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reactive in-flight rescue, rating-gated. Steers each eligible step toward trajectories you rated good and away from ones you rated Awful (matched to the current prompt). Learns automatically from ratings while on — no separate capture step needed. A no-op until you've rated a few gens for this prompt/key (a positive rating builds the target, an Awful builds what to avoid)."
                }),
                "rescue_threshold": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Fires when the step has diverged from the good trajectory by more than this (1 - cosine) OR aligned with a bad trajectory by more than this (cosine). Lower = corrects more eagerly. 0.10-0.20 typical; raise toward 0.4+ for only severe cases."
                }),
                "rescue_strength": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "How hard to pull toward good / push away from bad when triggered (magnitude preserved, no energy injected). Keep moderate; 0.5 is a strong correction."
                }),
            },
            "optional": {
                "sigmas": ("SIGMAS",),
                # Appended last on purpose so existing saved nodes keep their widget order.
                "normalize_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Video-only latent normalization (anti-overbake / oversaturation / colour drift) stacked on this RF loop. 0 = off. 0.5 = gentle. Audio is never touched (LTXAV/CONST path). ~zero overhead.",
                }),
                "normalize_start_sigma": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.025,
                    "tooltip": "Sigma at/below which latent normalization activates and anchors its reference. Only used when normalize_strength > 0.",
                }),
            }
        }

    RETURN_TYPES = ("SAMPLER", "SIGMAS")
    RETURN_NAMES = ("sampler", "sigmas")
    FUNCTION = "get_sampler"
    CATEGORY = "FunPack/Sampling"
    DESCRIPTION = (
        "Hybrid sampler: early Euler ancestral with order-2 denoised extrapolation for motion, "
        "late DPM-Solver++(2S) ODE for quality with progressive correction blending. "
        "Optional eta decay, anti-stiffness motion pulses, velocity bias, and reactive rescue. "
        "On rectified-flow models (CONST, e.g. LTXAV) it runs an RF-correct port of the same "
        "features (Heun corrector in place of 2S) instead of falling back to plain euler-ancestral."
    )

    def get_sampler(self, eta, s_noise, high_quality_pct, correction_blend,
                    quality_sharpness=0.0,
                    motion_pulse_mode="off", motion_pulse_start_pct=0.30,
                    motion_pulse_count=2, motion_pulse_spacing_pct=0.22,
                    motion_pulse_strength=0.85, velocity_bias_mode="off",
                    velocity_bias_strength=0.0, velocity_bias_source="mean",
                    velocity_refinement_key="default", rescue_mode=False,
                    rescue_threshold=0.15, rescue_strength=0.2, rescue_prompt_sig=None,
                    sigmas=None, eta_final=1.0,
                    normalize_strength=0.0, normalize_start_sigma=0.9):
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
                "quality_sharpness": quality_sharpness,
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
                "velocity_bias_source": velocity_bias_source,
                "velocity_refinement_key": velocity_refinement_key,
                "rescue_mode": rescue_mode,
                "rescue_threshold": rescue_threshold,
                "rescue_strength": rescue_strength,
                "rescue_prompt_sig": rescue_prompt_sig,
                "eta_final": eta_final,
                "normalize_strength": normalize_strength,
                "normalize_start_sigma": normalize_start_sigma,
            }
        )
        return (sampler, prepared_sigmas)


def _rf_ancestral(model):
    """True when this model is a rectified-flow (CONST) one — H3 and LTXAV both are.

    Decides which ancestral formulation is correct; a wrong answer only changes how noise is
    added, never whether sampling runs, so an unreadable model falls back to the eps form.
    """
    try:
        return isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST)
    except Exception:
        return False


def sample_funpack_distilled_flow(model, x, sigmas, extra_args=None, callback=None,
                                   disable=None, order=2, s_noise=0.0,
                                   final_correction_steps=0, ab2_ramp=False,
                                   normalize_strength=0.0, normalize_start_sigma=0.9,
                                   velocity_bias_mode="off", velocity_bias_strength=0.0,
                                   velocity_bias_source="mean", velocity_refinement_key="default",
                                   rescue_mode=False, rescue_threshold=0.15, rescue_strength=0.2,
                                   rescue_prompt_sig=None,
                                   alg_enabled=False, alg_strength=2.0, alg_sigma_threshold=0.975,
                                   alg_guide_tail_frames=0,
                                   alg_guide_blur_strength=2.0, alg_guide_blur_sigma_threshold=0.975,
                                   mg_enabled=False, mg_strength=0.5, mg_decay=0.5, mg_sigma_threshold=0.975,
                                   quality_sharpness=0.0, h3_audio_clock=False):
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
    - Velocity-bias capture/apply and reactive rescue around each model eval —
      the same memory machinery the Hybrid/RF samplers use. On few-step schedules
      the velocity targets may only match a step or two, so apply/rescue fire less
      often than on an 8-step run; they no-op cleanly when no target matches.
    - EXPERIMENTAL alg_enabled: Adaptive Low-Pass Guidance (arXiv:2506.08456) for i2v
      anchored chunks. Blurs the anchor frame the model sees while sigma is above
      alg_sigma_threshold (the near-pure-noise steps), then swaps back to the sharp
      anchor for the rest of the schedule — counters the model's tendency to shortcut
      to a near-static video that just matches the sharp reference. No-op when there's
      no i2v anchor (denoise_mask is None) or the packed latent layout can't be read.
    - EXPERIMENTAL alg_guide_tail_frames (>0 extends the same idea, not from the paper):
      additionally blurs that many trailing video frames — the newly-appended guide-
      attention frames this scene (mid_scene_guide / carry_i2v_guides-as-guide /
      configured guides / JoyAI memory), set by the Scene Chain Sampler's alg_blur_guides
      toggle right before this chunk's sample call. 0 = untouched (default). Standalone:
      works with alg_enabled off (anchor stays sharp, only the guide tail is blurred), with
      its own independent controls — alg_guide_blur_strength / alg_guide_blur_sigma_threshold
      (also set per-chunk by the Scene Chain Sampler), separate from the anchor's
      alg_strength / alg_sigma_threshold.
    - EXPERIMENTAL mg_enabled: Momentum Guidance (arXiv:2602.20360). Keeps a running EMA
      (decay=mg_decay) of the per-step ODE direction and blends the current step's
      direction toward it (weight=mg_strength) while sigma is BELOW mg_sigma_threshold —
      the complementary window to ALG's blur, which only acts above its threshold. The EMA
      itself accumulates every step (free) so it has real history by the time it starts
      being applied; only the application is gated. Audio is never touched (video-only).
    - Optional quality_sharpness: temporal-average unsharp on the x0 prediction, applied only
      during the final Heun-correction steps (same mechanism as the Hybrid Euler 2S sampler's
      quality-phase sharpening, ported here). Free (no extra model eval), video-only.
    - Optional h3_audio_clock: on MiniMax H3, integrate the audio stream on its own flow
      schedule instead of the tangent approximation the DiT hands the sampler. Matters most
      here — this sampler exists for the few-step schedules where that tangent is worst.
      Free (a scalar per step), no-op off H3.
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = k_diffusion_sampling.default_noise_sampler(x, seed=seed)

    total_steps = max(0, len(sigmas) - 1)
    if total_steps <= 0:
        return x

    sched_steps = max(1, int(len(sigmas)) - 1)

    order = max(1, min(2, int(order)))
    s_noise = max(0.0, min(1.0, float(s_noise)))
    # The TERMINAL step (sigma_next == 0) cannot be Heun-corrected: the corrector evaluates
    # the model at sigma_next, and sigma 0 is degenerate. It also returns early below, so a
    # window measured from the end of the schedule started ON that step and the corrector was
    # never reached — `final_correction_steps=1` did nothing at all on any schedule ending at
    # 0, which is all of them. The window is measured over the CORRECTABLE steps instead, so
    # N means N. Sharing the index with quality_sharpness is deliberate: its own tooltip
    # defines its window as the Heun-correction steps.
    correctable = sum(1 for i in range(sched_steps) if float(sigmas[i + 1]) > 0)
    final_correction_steps = max(0, min(correctable, int(final_correction_steps)))
    correction_start_idx = correctable - final_correction_steps

    _RESCUE_LOG["warned_no_memory"] = False
    _RESCUE_LOG["warned_no_prompt_match"] = False
    _RESCUE_LOG["fired"] = 0

    # Audio-safe steering: on a packed LTXAV latent keep velocity bias / rescue on the
    # video stream only (perturbing the audio stream corrupts it via joint attention).
    # None for single-stream LTXV -> _video_only is a no-op.
    video_mask = _packed_video_mask(model, x)
    if video_mask is not None and (rescue_mode or velocity_bias_mode != "off"):
        n_aud = int((video_mask < 0.5).sum().item())
        print(f"[FunPack AV] packed audio+video latent detected -> audio-safe steering "
              f"(audio held deterministic on {n_aud} of {video_mask.shape[-1]} packed dims)")

    normalize_strength = max(0.0, min(1.0, float(normalize_strength)))
    if normalize_strength > 0.0:
        print(f"[FunPack AV] latent normalization on (strength={normalize_strength}, "
              f"start_sigma={normalize_start_sigma}, video-only)")
    audio_clock = _audio_clock_setup(model, x, sigmas, h3_audio_clock)
    s_in = x.new_ones([x.shape[0]])
    prev_denoised = None
    prev_h = None
    ref_scale = [None]  # video-only latent-normalization reference (anti-overbake), opt-in
    _kvlock_state = {}

    # EXPERIMENTAL ALG (arXiv:2506.08456): blur the i2v anchor for the earliest/near-pure-
    # noise steps, then swap back to the sharp anchor. No-op without a real anchor (no
    # denoise_mask) or if the packed-latent layout can't be read (helper returns None).
    # alg_guide_tail_frames (>0) extends the same idea, beyond the paper, to the trailing
    # guide-attention frames this scene appended (alg_blur_guides on the Scene Chain Sampler).
    # Anchor and guide-tail blur are fully independent: each has its own strength and sigma
    # window (alg_strength/alg_sigma_threshold vs alg_guide_blur_*), so one precomputed
    # latent per (anchor blurred?, tail blurred?) combination that can occur — the frame
    # sets are disjoint, so the both-blurred variant just composes the two.
    alg_guide_tail_frames = max(0, int(alg_guide_tail_frames))
    alg_sharp_latent_image, alg_latents, alg_anchor_on, alg_tail_on = _alg_prepare(
        model, extra_args, bool(alg_enabled), alg_strength,
        alg_guide_tail_frames, alg_guide_blur_strength,
    )
    alg_active = alg_anchor_on or alg_tail_on
    if alg_active:
        anchor_desc = (f"strength={alg_strength}, sigma_threshold={alg_sigma_threshold}"
                       if alg_anchor_on else "off")
        tail_desc = (f"{alg_guide_tail_frames} frame(s), strength={alg_guide_blur_strength}, "
                     f"sigma_threshold={alg_guide_blur_sigma_threshold}" if alg_tail_on else "off")
        print(f"[FunPack AV] ALG on (anchor: {anchor_desc}; guide tail: {tail_desc}) "
              f"— blurred while sigma > threshold")

    # EXPERIMENTAL Momentum Guidance (arXiv:2602.20360): EMA of the per-step ODE direction,
    # blended into the direction actually used once sigma drops below mg_sigma_threshold —
    # the complementary window to ALG's blur. EMA accumulates every step regardless of
    # whether it's being applied yet, so there's no cold-start when it switches on.
    mg_active = bool(mg_enabled)
    mg_decay = max(0.0, min(0.999, float(mg_decay)))
    mg_strength = max(0.0, min(1.0, float(mg_strength)))
    mg_ema = None
    if mg_active:
        print(f"[FunPack AV] Momentum Guidance on (strength={mg_strength}, decay={mg_decay}, "
              f"sigma_threshold={mg_sigma_threshold}) — applied while sigma < threshold")

    def _mg_step(d, video_mask):
        nonlocal mg_ema
        if not mg_active:
            return d
        d_detached = d.detach()
        mg_ema = d_detached if mg_ema is None else (mg_decay * mg_ema + (1.0 - mg_decay) * d_detached)
        if float(sigma) >= float(mg_sigma_threshold) or mg_strength <= 0.0:
            return d
        try:
            blended = d * (1.0 - mg_strength) + mg_ema.to(device=d.device, dtype=d.dtype) * mg_strength
        except Exception as _e:
            _log.failed("FunPackSceneChain", "momentum guidance", _e,
                        "this step uses the raw derivative, unsmoothed")
            return d
        return _video_only(blended, d, video_mask)

    try:
        for i in comfy.utils.model_trange(total_steps, disable=disable):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            velocity_target = _velocity_bias_target(sigmas, sigma)

            if alg_active:
                model.latent_image = alg_latents[(
                    alg_anchor_on and float(sigma) > float(alg_sigma_threshold),
                    alg_tail_on and float(sigma) > float(alg_guide_blur_sigma_threshold),
                )]

            if _velocity_bias_enabled(velocity_bias_mode, "apply"):
                x_pre = x
                x = _apply_velocity_bias(x, velocity_refinement_key, velocity_target, velocity_bias_strength,
                                         sigma_ratio=_sigma_ratio(sigmas, sigma),
                                         prompt_sig=rescue_prompt_sig, source=velocity_bias_source)
                x = _video_only(x, x_pre, video_mask)

            denoised = model(x, sigma * s_in, **extra_args)
            _kvlock_schedule(model, denoised, prev_denoised, video_mask, _kvlock_state)

            if rescue_mode or _velocity_bias_enabled(velocity_bias_mode, "capture"):
                _capture_velocity_bias(velocity_refinement_key, velocity_target, x, sigma, denoised, prompt_sig=rescue_prompt_sig)
            if rescue_mode and velocity_target is not None:
                denoised = _video_only(_rescue_denoised(
                    denoised, x, sigma, velocity_refinement_key,
                    velocity_target, rescue_threshold, rescue_strength, prompt_sig=rescue_prompt_sig,
                    source=velocity_bias_source,
                ), denoised, video_mask)

            # Restore high-frequency detail during the final Heun-correction steps (free,
            # video-only) — same unsharp mechanism as the Hybrid Euler 2S sampler.
            if i >= correction_start_idx:
                denoised = _video_only(_apply_quality_sharpness(denoised, prev_denoised, quality_sharpness), denoised, video_mask)

            # Video-only latent normalization (opt-in, anti-overbake) — stacks on top of the ODE.
            denoised = _normalize_video_denoised(
                denoised, video_mask, sigma, ref_scale, normalize_strength, normalize_start_sigma,
            )

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

            # Graduated 2nd order (opt-in, free): ramp the AB2 contribution linearly 0->1 across
            # the schedule. Early/high-sigma steps stay near 1st-order euler (the denoised estimate
            # is rough there and full AB2 overshoots); late/detail steps get full AB2. Reuses the
            # already-computed AB2 estimate, so no extra model evals. No-op at order=1.
            if ab2_ramp and sched_steps > 1:
                w = i / (sched_steps - 1)  # 0 at first step -> 1 at last
                denoised_eff = denoised + (denoised_eff - denoised) * w

            # Audio rides plain 1st-order euler: keep the (ramped) AB2 estimate for video,
            # raw denoised for audio (2nd-order extrapolation corrupts the audio stream).
            denoised_eff = _video_only(denoised_eff, denoised, video_mask)

            # Store current denoised for the next step's multistep correction.
            # Reset after a Heun step since x was updated with a corrected direction.
            prev_denoised = denoised.detach()
            prev_h = h

            if sigma_next == 0:
                # Same displacement the euler branch below would produce at dt = -sigma,
                # so the audio clock corrects this final step like any other.
                x = _audio_clock_step(denoised_eff, x, audio_clock, i)
                continue

            dt = sigma_next - sigma  # negative: sigmas decrease

            if i >= correction_start_idx:
                # Heun predictor-corrector.
                # Predictor: Euler step using the (multistep-corrected) denoised.
                d1 = k_diffusion_sampling.to_d(x, sigma, denoised_eff)
                # The corrector's own audio output is discarded (audio rides d1), but the
                # state it is evaluated at is shared: audio and video are one packed
                # sequence, so an over-stepped audio half would skew the VIDEO correction
                # through joint attention. Predict the audio where the clock says it lands.
                x_pred = _audio_clock_step(x + d1 * dt, x, audio_clock, i)
                # Corrector: evaluate model at the predicted x and sigma_next.
                denoised_pred = model(x_pred, sigma_next * s_in, **extra_args)
                d2 = k_diffusion_sampling.to_d(x_pred, sigma_next, denoised_pred)
                # Audio rides the plain euler direction (d1); only video gets the Heun correction.
                d_use = _video_only((d1 + d2) / 2.0, d1, video_mask)
                d_use = _mg_step(d_use, video_mask)
                x = _audio_clock_step(x + d_use * dt, x, audio_clock, i)
                # Heun updates x differently; invalidate multistep history.
                prev_denoised = None
                prev_h = None
            else:
                d = k_diffusion_sampling.to_d(x, sigma, denoised_eff)
                d = _mg_step(d, video_mask)
                x_det = _audio_clock_step(x + d * dt, x, audio_clock, i)
                if s_noise > 0.0 and float(sigma_next) > 0 and _rf_ancestral(model):
                    # Rectified-flow ancestral step, identical to comfy's
                    # sample_euler_ancestral_RF (which is what `euler_ancestral` becomes on a
                    # CONST model such as MiniMax H3 or LTXAV).
                    #
                    # The previous version stepped the FULL distance to sigma_next and then
                    # added sqrt(sigma^2 - sigma_next^2) of noise on top. Both halves are wrong
                    # for a flow model: that variance formula is the VP one, and adding noise
                    # without shortening the deterministic step leaves the latent noisier than
                    # the schedule says it should be at sigma_next. It over-noised, which is
                    # why s_noise here never behaved like euler_ancestral.
                    #
                    # The correct step lands SHORT (sigma_down), then renoises with the alpha
                    # rescaling flow matching requires. At s_noise=1.0 this is euler_ancestral.
                    downstep_ratio = 1 + (sigma_next / sigma - 1) * s_noise
                    sigma_down = sigma_next * downstep_ratio
                    alpha_ip1 = 1 - sigma_next
                    alpha_down = 1 - sigma_down
                    ratio = sigma_down / sigma
                    x_anc = ratio * x + (1 - ratio) * denoised_eff
                    renoise = (sigma_next ** 2
                               - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2) ** 0.5
                    x_anc = ((alpha_ip1 / alpha_down) * x_anc
                             + noise_sampler(sigma, sigma_next) * renoise)
                    # Video only: ancestral noise on the audio stream corrupts it, and the
                    # audio half of x_det already carries the clock correction.
                    x = _video_only(x_anc, x_det, video_mask)
                elif s_noise > 0.0 and float(sigma_next) > 0:
                    # Non-CONST (eps/VP) model: the original formulation, unchanged.
                    sigma_up = math.sqrt(max(0.0, float(sigma.item()) ** 2 - float(sigma_next.item()) ** 2))
                    x = _video_only(x_det + noise_sampler(sigma, sigma_next) * s_noise * sigma_up,
                                    x_det, video_mask) if sigma_up > 0.0 else x_det
                else:
                    x = x_det
    finally:
        if alg_active:
            model.latent_image = alg_sharp_latent_image

    return x


# FunPack sampler functions that accept h3_audio_clock. The hybrid entry point is here
# because on a CONST model (which MiniMax H3 is) it delegates to _sample_const_rf_full,
# which carries the correction; its own eps loop is unreachable on H3.
_AUDIO_CLOCK_SAMPLERS = (sample_funpack_distilled_flow, sample_funpack_hybrid_euler_2s)


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
                    "default": 0,
                    "min": 0,
                    "max": 3,
                    "step": 1,
                    "tooltip": "Number of correctable steps that use a Heun predictor-corrector pass. EACH COSTS ONE EXTRA MODEL CALL (on a 7-step run, 1 = ~14% more time). Counted over the steps that CAN be corrected: the terminal step lands on sigma 0, where the corrector has nothing to evaluate, so it is never one of them. Was 1 by default and did nothing — the window used to start on that terminal step, which returns before the corrector runs — so 0 is what every run has actually been doing. Set it to 1 to get the correction the knob always advertised, on the last real step. quality_sharpness shares this window and needs this above 0.",
                }),
                "s_noise": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.00,
                    "step": 0.01,
                    "tooltip": "Ancestral noise: the fraction of each step that is taken stochastically rather than deterministically (comfy calls this eta). 0 = fully deterministic ODE. 1.0 on a rectified-flow model (MiniMax H3, LTXAV) is exactly euler_ancestral. Free — no extra model calls. BEHAVIOUR CHANGED: this used to step the full distance and then add sqrt(sigma^2-sigma_next^2) of noise ON TOP, which is the VP variance formula and leaves the latent noisier than the schedule says — it never matched euler_ancestral at any value. It now lands short (sigma_down) and renoises with the alpha rescaling flow matching needs. A value carried over from before does NOT mean what it used to: re-tune it. If you liked euler_ancestral, start at 1.0.",
                }),
                "velocity_bias_mode": (VELOCITY_BIAS_MODES, {
                    "default": "off",
                    "tooltip": "Experimental: capture/apply averaged early model velocity around normalized sigma 0.9/0.72/0.42. Off preserves the plain distilled ODE. Note: few-step schedules may only land on a target or two, so it fires less often than on an 8-step run.",
                }),
                "velocity_bias_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Strength of the remembered velocity (action) injected at the structure sigma. 0 disables. ~0.15 = subtle spice; 0.3-1.0 = clear action crossover; 2-3 approaches full action replacement (capped so the current gen isn't wiped). Creative tool, not for consistency.",
                }),
                "velocity_bias_source": (["mean", "nearest"], {
                    "default": "mean",
                    "tooltip": "How velocity bias / rescue pick a good direction. 'mean' = prompt-blind global average. 'nearest' = single best-matching prompt cluster — preserves one real good gen's detail instead of a washed-out average. Affects both apply and rescue.",
                }),
                "velocity_refinement_key": ("STRING", {
                    "default": "default", "multiline": False,
                    "tooltip": "Memory key used to capture/apply early velocity bias and rescue trajectories.",
                }),
                "rescue_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reactive in-flight rescue, rating-gated. Steers each eligible step toward trajectories you rated good and away from ones you rated Awful (matched to the current prompt). A no-op until you've rated a few gens for this prompt/key.",
                }),
                "rescue_threshold": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Fires when the step has diverged from the good trajectory by more than this (1 - cosine) OR aligned with a bad trajectory by more than this (cosine). Lower = corrects more eagerly. 0.10-0.20 typical.",
                }),
                "rescue_strength": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "How hard to pull toward good / push away from bad when triggered (magnitude preserved, no energy injected). Keep moderate; 0.5 is a strong correction.",
                }),
            },
            "optional": {
                "sigmas": ("SIGMAS",),
                # Appended last on purpose: a new widget inserted earlier shifts every saved
                # widgets_values slot in existing Distilled Flow nodes.
                "ab2_ramp": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Graduated 2nd order (free). Instead of full AB2 on every step, ramp the AB2 contribution linearly 0->1 across the schedule: early/noisy steps stay near 1st-order euler (less overshoot), late/detail steps get full AB2. No extra model calls. Helps low-step distilled runs. No effect at order=1.",
                }),
                "normalize_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Video-only latent normalization (anti-overbake / oversaturation / colour drift) stacked on this ODE. 0 = off. 0.5 = gentle. Audio is never touched. ~zero overhead.",
                }),
                "normalize_start_sigma": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.025,
                    "tooltip": "Sigma at/below which latent normalization activates and anchors its reference (above it the x0 estimate is meaningless). Only used when normalize_strength > 0.",
                }),
                # Appended last on purpose, same rule as ab2_ramp above.
                "alg_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL: Adaptive Low-Pass Guidance (arXiv:2506.08456). Blurs the i2v anchor frame while sigma is above alg_sigma_threshold, then swaps back to the sharp anchor — counters the model's tendency to shortcut to a near-static video that just matches the reference image. No-op without an i2v anchor.",
                }),
                "alg_strength": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1,
                    "tooltip": "Downsample factor for the anchor blur (paper default 2.5, but 2.0 held character/i2v consistency noticeably better in testing here). Higher = blurrier anchor during the affected steps. Only used when alg_enabled; guide-frame blur has its own controls on the Scene Chain Sampler (alg_blur_guides + alg_guide_blur_*).",
                }),
                "alg_sigma_threshold": ("FLOAT", {
                    "default": 0.975, "min": 0.5, "max": 0.999, "step": 0.005,
                    "tooltip": "Anchor stays blurred while sigma is above this value (the near-pure-noise steps), then swaps to sharp. Higher = narrower blurred window. Only used when alg_enabled; guide-frame blur has its own controls on the Scene Chain Sampler (alg_blur_guides + alg_guide_blur_*).",
                }),
                "mg_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL: Momentum Guidance (arXiv:2602.20360). Keeps a running average of the per-step direction and blends the current step toward it once sigma drops below mg_sigma_threshold — the complementary window to ALG's blur. Smooths the fine-motion/refinement steps; may damp motion as a side effect, untested for video.",
                }),
                "mg_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend weight toward the momentum average when active (0 = no effect, 1 = fully replace the step's direction with the average). Only used when mg_enabled.",
                }),
                "mg_decay": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 0.99, "step": 0.05,
                    "tooltip": "EMA decay for the momentum average (higher = longer memory of past steps' directions). On our 8-step schedule, high decay (e.g. 0.9) keeps the EMA anchored to the very first, near-pure-noise step's direction for nearly the whole run — wrong scale, produces garbage regardless of mg_strength. Tested safe (and good) at 0.5 even at mg_strength=1.0. Only used when mg_enabled.",
                }),
                "mg_sigma_threshold": ("FLOAT", {
                    "default": 0.975, "min": 0.5, "max": 0.999, "step": 0.005,
                    "tooltip": "Momentum guidance applies while sigma is BELOW this value (the opposite window from alg_sigma_threshold) — defaults to the same boundary as ALG for a clean handoff. Only used when mg_enabled.",
                }),
                # Appended last on purpose, same rule as ab2_ramp above.
                "quality_sharpness": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Restores fine detail via temporal-average unsharp on the x0 prediction, applied only during the final Heun-correction steps (final_correction_steps > 0 required to have any effect). 0 disables. 0.2-0.4 typical. Free (no extra model eval), video-only.",
                }),
            }
        }

    RETURN_TYPES = ("SAMPLER", "SIGMAS")
    RETURN_NAMES = ("sampler", "sigmas")
    FUNCTION = "get_sampler"
    CATEGORY = "FunPack/Sampling"
    DESCRIPTION = (
        "ODE sampler for distilled few-step video models (e.g. LTX2.3 distilled LoRA). "
        "Adams-Bashforth 2-step multistep for better trajectory accuracy across large sigma jumps, "
        "Heun predictor-corrector on final steps for quality, optional controlled noise for diversity, "
        "optional velocity bias + reactive rescue (shared with the Hybrid/RF samplers), "
        "experimental ALG anchor de-staticking for i2v chunks, and experimental Momentum "
        "Guidance smoothing for the complementary (fine-motion) sigma window."
    )

    def get_sampler(self, order=2, final_correction_steps=0, s_noise=0.0,
                    velocity_bias_mode="off", velocity_bias_strength=0.0,
                    velocity_bias_source="mean", velocity_refinement_key="default",
                    rescue_mode=False, rescue_threshold=0.15, rescue_strength=0.2,
                    rescue_prompt_sig=None, sigmas=None, ab2_ramp=False,
                    normalize_strength=0.0, normalize_start_sigma=0.9,
                    alg_enabled=False, alg_strength=2.0, alg_sigma_threshold=0.975,
                    mg_enabled=False, mg_strength=0.5, mg_decay=0.5, mg_sigma_threshold=0.975,
                    quality_sharpness=0.0):
        prepared_sigmas = sigmas.detach().clone() if isinstance(sigmas, torch.Tensor) else sigmas
        sampler = comfy.samplers.KSAMPLER(
            sample_funpack_distilled_flow,
            extra_options={
                "order": order,
                "final_correction_steps": final_correction_steps,
                "ab2_ramp": ab2_ramp,
                "normalize_strength": normalize_strength,
                "normalize_start_sigma": normalize_start_sigma,
                "s_noise": s_noise,
                "velocity_bias_mode": velocity_bias_mode,
                "velocity_bias_strength": velocity_bias_strength,
                "velocity_bias_source": velocity_bias_source,
                "velocity_refinement_key": velocity_refinement_key,
                "rescue_mode": rescue_mode,
                "rescue_threshold": rescue_threshold,
                "rescue_strength": rescue_strength,
                "rescue_prompt_sig": rescue_prompt_sig,
                "alg_enabled": alg_enabled,
                "alg_strength": alg_strength,
                "alg_sigma_threshold": alg_sigma_threshold,
                "mg_enabled": mg_enabled,
                "mg_strength": mg_strength,
                "mg_decay": mg_decay,
                "mg_sigma_threshold": mg_sigma_threshold,
                "quality_sharpness": quality_sharpness,
            }
        )
        return (sampler, prepared_sigmas)


def _normalize_video_denoised(denoised, video_mask, sigma, ref, strength,
                              start_sigma, tolerance=0.05, beta=0.9):
    """Keep the VIDEO latent's spread from drifting into overbaked / oversaturated ranges.

    Operates on the denoised (x0) prediction, video stream only — audio is never touched, so it
    is audio-safe by construction. Once x0 is meaningful (sigma <= start_sigma) it anchors a
    reference spread (robust std over the video region) and gently compresses any later step
    whose spread inflates past it; an EMA lets the reference follow legitimate detail growth
    while resisting abrupt overbake spikes. Benefit-only: it never amplifies and is a no-op when
    the latent stays healthy. Cost is a few masked reductions per step (~zero overhead).

    `ref` is a 1-element list holding the reference scale across steps (mutable state).
    """
    if strength <= 0.0:
        return denoised
    try:
        s = float(sigma.item()) if hasattr(sigma, "item") else float(sigma)
    except Exception:
        s = 1.0
    if s > start_sigma:
        return denoised  # skip the pure-noise plateau where the x0 estimate is not yet meaningful
    try:
        # Stats over the video region (whole tensor for single-stream LTXV, where mask is None).
        if video_mask is not None:
            m = video_mask
            w = m.sum().clamp_min(1.0)
            mean = (denoised * m).sum() / w
            var = (((denoised - mean) ** 2) * m).sum() / w
        else:
            mean = denoised.mean()
            var = ((denoised - mean) ** 2).mean()
        std_v = float(torch.sqrt(var.clamp_min(1e-12)).item())
        if ref[0] is None:
            ref[0] = std_v
            return denoised
        target = ref[0] * (1.0 + tolerance)
        if std_v > target and std_v > 1e-6:
            factor = 1.0 - strength * (1.0 - target / std_v)  # = lerp(1, target/std, strength)
            normalized = mean + (denoised - mean) * factor
            denoised = _video_only(normalized, denoised, m)
            ref[0] = ref[0] * beta + target * (1.0 - beta)
        else:
            ref[0] = ref[0] * beta + std_v * (1.0 - beta)
    except Exception:
        return denoised
    return denoised


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
                "frame_overlap": ("INT", {
                    "default": 16, "min": 0, "max": 512, "step": 8,
                    "tooltip": "Pixel frames copied from the previous scene into the next chunk and preserved during denoising. 0 disables overlap blending entirely. WARNING: combining frame_overlap=0 with carry_i2v_guides=True is confirmed to produce bad results — use only for testing.",
                }),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "max_scenes": ("INT", {"default": 8, "min": 1, "step": 1}),
                "use_same_seed": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use the first provided scene seed for every scene. Off uses per-scene metadata seeds or seed + scene index.",
                }),
                "carry_i2v_guides": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Carry protected frames from latent_template noise_mask into each continuation chunk as a style guide.",
                }),
                "mid_scene_guide": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Experimental: append the middle frame of the previous scene as a guide for the current scene via LTX guide attention. Helps maintain character positioning across scenes.",
                }),
                "mid_scene_guide_strength": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Guide attention strength for mid-scene anchor. Full 0..1 range. 0.25-0.35 is the measured sweet spot: below 0.25 audio degrades and character appearance drifts, above 0.35 spatial conflicts appear when scene composition shifts. Outside that band is yours to explore — 0 disables the guide's pull entirely.",
                }),
                "embed_guidance": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply the Refiner's learned quality direction at each denoising step, not just once before sampling. Requires refinement_key_input and enough liked generations to have a direction. Adds ~20-30% inference overhead.",
                }),
                "embed_guidance_strength": ("FLOAT", {
                    "default": 0.02, "min": 0.005, "max": 0.1, "step": 0.005,
                    "tooltip": "Per-step nudge strength toward the liked conditioning direction. Keep small — the direction is applied at every step so it compounds. 0.01-0.03 is typical.",
                }),
                "score_slider": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "FreeSliders-style taste guidance in SCORE space. Instead of nudging the conditioning once (embed_guidance), it runs 3 forward passes on quality-phase steps — base, taste+, taste- — and steers the noise prediction along eps_+ minus eps_-. Stronger, prompt-faithful taste push; ~2x cost on late steps. Uses the same learned direction + source + refinement_key_input as embed_guidance (needs 3+ liked generations). Contrastive pair: once 3+ disliked/awful gens are rated, the minus pole switches from a mirror of liked to the real learned BAD direction, so the axis becomes good-vs-bad and actively steers away from what produced rated-bad gens. Video-only (audio unaffected).",
                }),
                "score_slider_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.25,
                    "tooltip": "Slider amount (eta). How hard to push the noise prediction along the learned taste axis. 1.0 is a clear, safe push; raise toward 3.0 for a stronger effect (paper's saturation range). 0 = off.",
                }),
                "transition_duration": ("INT", {
                    "default": 16, "min": 0, "max": 128, "step": 2,
                    "tooltip": "Extra pixel frames of fade beyond the blend zone on each side of a scene boundary. 0 = disable all transition effects.",
                }),
            },
            "optional": {
                # Widget inputs first, then connection/forceInput sockets LAST. A forceInput
                # input placed between widgets desyncs ComfyUI's widgets_values mapping on reload
                # (the combo value lands on decode_tile_size -> NaN/"relative" in the INT field).
                "decode_tile_size": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 64,
                    "tooltip": "Tile size for VAE decode (0 = no tiling). Set to e.g. 512 if decode OOMs.",
                }),
                "decode_noise_scale": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Noise injected at VAE decode to restore fine detail/grain (the LTX VAE decoder is itself a tiny diffusion model). 0 = off (clean decode). ~0.025 is a gentle detail restore. Applied to this node's IMAGES decode only.",
                }),
                "decode_timestep": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Decode timestep fed to the VAE decoder when decode_noise_scale > 0. ~0.05 adds subtle detail; higher gives the decoder more freedom (more deviation from the latent).",
                }),
                "embed_guidance_source": (["relative", "absolute"], {
                    "default": "relative",
                    "tooltip": "Which learned direction embed_guidance steers toward. Relative: this prompt's liked direction (needs refinement_key_input). Absolute: the global, prompt-agnostic taste direction the Refiner accumulates across all prompts — works with no key.",
                }),
                # NOTE: append-only. New widgets MUST go at the END of the widget sequence (after
                # every widget the reference workflow already has a positional value for, before the
                # forceInput sockets below). Inserting earlier desyncs extract_widgets' positional
                # mapping and lands a numeric on a combo (e.g. joyai_frame_select got 0).
                "joyai_memory": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "JoyAI-Echo cross-shot memory bank. REQUIRES the JoyAI-Echo LoRA — without it the injected memory frames change nothing, because the base model was never trained to read them as memory. Generalizes mid_scene_guide from one anchor to a managed set of clean prior-shot frames injected into each scene via LTX guide attention, so character/scene identity carries across the whole chain (JoyAI-Echo's story-level consistency). The first joyai_fix_frames scenes are pinned permanently as a global anchor; the rest is a rolling most-recent window capped at joyai_memory_size. Supersedes mid_scene_guide when on. Video memory only; pair it with joyai_audio_memory for the soundtrack.",
                }),
                "joyai_memory_size": ("INT", {
                    "default": 7, "min": 1, "max": 32,
                    "tooltip": "Max total memory entries injected per scene (JoyAI default 7). Higher = stronger long-range consistency but more guide tokens and slower scenes.",
                }),
                "joyai_fix_frames": ("INT", {
                    "default": 3, "min": 0, "max": 16,
                    "tooltip": "Number of opening scenes pinned permanently in the bank as a global anchor (JoyAI default 3). They are never pruned; entries beyond them are a rolling most-recent window.",
                }),
                "joyai_frame_select": (["center", "first", "random"], {
                    "default": "center",
                    "tooltip": "Which frame of each finished scene to store in the bank (JoyAI default 'center').",
                }),
                "joyai_memory_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.25, "max": 10.0, "step": 0.05,
                    "tooltip": "Guide-attention strength for each memory frame. 0.25 floor as mid_scene_guide (below it audio degrades and identity drifts). Uncapped at the top: 0.25-0.5 is the audio-safe band, higher values push identity harder but may degrade audio/over-constrain motion.",
                }),
                # NOTE: append-only — keep new sampler widgets at the END of this block so the
                # builder's positional reference-workflow mapping (extract_widgets) stays aligned.
                "joyai_audio_memory": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "JoyAI-Echo PAIRED AUDIO memory. Needs the JoyAI-Echo LoRA, like the video memory it accompanies. Alongside each video memory frame, pin the prior shot's clean audio latent into the audio stream so voice/timbre/ambience carry across shots the way the face now does. Deliberately breaks the audio pass-through invariant — off by default. Requires joyai_memory on; no effect on single-stream (video-only) LTXV.",
                }),
                "v2a_grad_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.25,
                    "tooltip": "JoyAI-Echo video->audio coupling. Scales the model's trained video-to-audio cross-attention so the carried audio tracks the new shot's visuals (JoyAI uses 2.0). 1.0 = native model behavior (no change, zero overhead); 0.0 = audio ignores video this run. Only applies when joyai_audio_memory is on.",
                }),
                "audio_vae": ("VAE", {
                    "tooltip": "MiniMax H3 only: the audio VAE, needed to encode AUDIO reference media for ref2va (a voice or ambience clip the generation should sound like). Studio lists the references and bakes their <Audio j> labels into the prompt; this VAE turns them into the latent blocks the DiT packs. Image references need only the main vae and work without this. Ignored entirely on LTX.",
                }),
                "h3_keyframes": ("CONDITIONING", {
                    "tooltip": "MiniMax H3 only: the CONDITIONING output of a MiniMax H3 Image to Video node, wired here purely so its first_frame / last_frame pins survive. That node's conditioning is otherwise discarded (the sampler's positive comes from Studio), which silently drops the image. Only the keyframe pins are read — the prompt encoded by that node is ignored, so write your prompt in Studio as usual. A first-frame pin lands on scene 1, a last-frame pin on the last scene's final frame. Ignored entirely on LTX.",
                }),
                "refinement_key_input": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "forceInput": True,
                    "tooltip": "Connect to the same refinement key as your V2 Refiner. When wired, the sampler writes carry_i2v_guides, frame_overlap, and scene count into the refinement state so the Refiner can reason about what changed between rated runs.",
                }),
                "funpack_scene_guides": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional JSON from Movie Editor when guide_settings.stack_enabled: per-scene guide lists with source, frame_idx, apply_at, strength. When empty, carry_i2v_guides uses the Studio default (scene 1 template at frame 0).",
                }),
                "funpack_scene_anchors": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional JSON map of scene_index → {filename, strength} for mixed-source i2v anchors (LTXVImgToVideoInplace starting latent). Distinct from i2v guides.",
                }),
                "funpack_scene_media_refs": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional media_ref → filename map for image-type i2v guides in custom guide stacks.",
                }),
                "alg_blur_guides": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL: extends ALG (see alg_anchor) from just the i2v anchor to also blur newly-appended guide-attention frames this scene (mid_scene_guide / carry_i2v_guides-as-guide / configured per-scene guides / JoyAI memory), for the same early steps. Standalone: works even with the anchor blur off (anchor stays sharp), with its own alg_guide_blur_strength / alg_guide_blur_sigma_threshold controls below. Works with ANY wired sampler — inside the loop on FunPack Distilled Flow, and through a denoiser proxy (same sigma schedule, same result) on everything else. No effect if no guide frames were appended this scene.",
                }),
                "bounded_attention_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL: Bounded Attention (arXiv:2403.16990) + Structured Diffusion Guidance (arXiv:2212.05032)-style exact split. Studio splits multi-sentence scene prompts by sentence count and encodes each half SEPARATELY (no shared tokenization, exact boundary), then this masks text cross-attention so the left half of the frame can only attend to subject-1's tokens and the right half only to subject-2's — aims to stop attribute/anatomy bleed between two figures in one frame. No-op on single-sentence prompts or single-subject scenes. Works on any sampler (model-level hook, not sampler-specific).",
                }),
                # NOTE: append-only — keep new sampler widgets at the END of this block so the
                # builder's positional reference-workflow mapping (extract_widgets) stays aligned.
                "output_guidance": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL: sibling of embed_guidance, but the learned quality direction is trained on and applied to the model's own predicted OUTPUT (x0_hat) instead of the input conditioning — a separate value function (needs its own 10+ rated generations to activate; see refinement key's *.x0_snapshot.pt / *.value_fn_x0.pt). Same near-zero mechanism as embed_guidance (one backward pass through a small MLP, no extra model forward pass), applied post-prediction rather than pre-input. Requires refinement_key_input. Cost unmeasured yet — treat as embed_guidance-shaped until benchmarked, not assumed cheaper.",
                }),
                "output_guidance_strength": ("FLOAT", {
                    "default": 0.02, "min": 0.005, "max": 0.1, "step": 0.005,
                    "tooltip": "Per-step correction strength applied to the model's predicted output. Same scale/units as embed_guidance_strength — start there and adjust.",
                }),
                "dynashift": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL DynaShift: a negative prompt at CFG=1, driven by YOUR bad ratings instead of text. Bad-outcome ratings ('Awful', 'Wrong appearance', and the quality-missing family: 'Missing quality' / '+details' / '+action' combos) store that run's video latent in a per-key negative bank; near-miss ratings with positive reward ('Missing details', 'Missing action') deliberately do NOT. During sampling, frames that start to look like a banked bad generation are steered away (projection removal) until the match drops below the threshold. Alignment-free in time (chain position / guide tails don't matter); negatives from a different resolution are skipped; each negative is weighted by prompt similarity so unrelated bad gens steer less. Requires refinement_key_input; silent until the bank has at least one entry. No extra model pass — near-zero overhead. Audio untouched.",
                }),
                "dynashift_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "Fraction of the matched negative component removed per fully-gated late step (accumulates over ~4 quality-phase steps). 0.3 is a gentle nudge; 1.0 removes the matched component outright each step.",
                }),
                "dynashift_threshold": ("FLOAT", {
                    "default": 0.6, "min": 0.3, "max": 0.95, "step": 0.05,
                    "tooltip": "Frame-similarity gate: a current frame must match a banked negative frame above this cosine similarity before any steering applies. Steering strength ramps from 0 at the threshold to full at similarity 1.0, so it self-releases once the unwanted feature is gone. Lower = more aggressive (risks steering away from legitimately similar content).",
                }),
                "alg_guide_blur_strength": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1,
                    "tooltip": "Downsample factor for the guide-frame blur (alg_blur_guides). Higher = blurrier guide/JoyAI frames during the affected steps. Independent of the sampler's anchor alg_strength.",
                }),
                "alg_guide_blur_sigma_threshold": ("FLOAT", {
                    "default": 0.975, "min": 0.5, "max": 0.999, "step": 0.005,
                    "tooltip": "Guide frames stay blurred while sigma is above this value (the near-pure-noise steps), then swap to sharp. Higher = narrower blurred window. Independent of the sampler's anchor alg_sigma_threshold.",
                }),
                "identity_transfer_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL Best-FaceID compatibility: full native port of the overlap+source_phase+ArcFace conditioning Best-FaceID-style identity LoRAs were trained on. Replaces Continuity's 'Identity pin' guide (Engine settings) with separate, non-rendered reference tokens (never blended into frame 0) plus optional ArcFace projector tokens on the text context. No-op without an identity pin image set. Load the LoRA itself the normal way (Models -> add a LoRA loader onto the model path).",
                }),
                "identity_projector": (cls._identity_projector_choices(), {
                    "default": "None",
                    "tooltip": "ArcFace projector .safetensors (from models/loras). 'None' = overlap only (the projector is a weak secondary channel; the overlap latent carries the bulk of identity).",
                }),
                "source_id": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 8.0, "step": 1.0,
                    "tooltip": "source_phase segment id for the overlap reference tokens (ltx-trainer's overlap+source_phase convention used 2). 0 disables the RoPE rotation while leaving the overlap tokens active.",
                }),
                "phase_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1,
                    "tooltip": "Multiplier on source_id before the RoPE rotation.",
                }),
                "id_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 50.0, "step": 0.5,
                    "tooltip": "Multiplies the ArcFace projector tokens (only when identity_projector is set). Weak channel; push high (5-20) to test, very high may add artifacts.",
                }),
                "arcface_mode": (["auto_adjust", "as_is", "disable"], {
                    "default": "auto_adjust",
                    "tooltip": "auto_adjust: retry face detection with zoom-out/upscale, skip projector tokens if none found. as_is: detect on the image only. disable: skip ArcFace, use only the overlap latent.",
                }),
                "debug_log": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print per-scene identity-transfer shape/status logs to the console.",
                }),
                # NOTE: append-only — keep new sampler widgets at the END of this block so the
                # builder's positional reference-workflow mapping (extract_widgets) stays aligned.
                "carry_overlap_through_anchor": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When a scene switches to its own i2v anchor (funpack_scene_anchors — a different reference image/character than the previous scene), still carry frame_overlap latent frames from the previous scene's tail into the frames right after the anchor image, instead of the default hard cut with no carried context. Preserves background/environment continuity through an anchor change (e.g. a Best-FaceID identity_transfer scene swapping the reference face mid-chain). The anchor image's own leading frame is never touched by the carried tail. No effect on scenes without a per-scene anchor.",
                }),
                "plateau_cache": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL speed (MixCache/Chorus-family step-caching, adapted to LTX2.3's distilled schedule). IGNORED while context_windows is on (the cache can't tell one window from another within a step) — the scene report says so. Mechanism: the near-pure-noise plateau steps (sigma above plateau_cache_threshold) carry almost no signal, so the model's output barely changes across them. This computes the full transformer forward once at the top of the plateau, then REUSES that output for the remaining plateau steps instead of recomputing — skipping those transformer passes entirely. On the default 8-step schedule (sigmas 1.0→0.975 are the plateau) that's ~3-4 of 8 forwards skipped. DETERMINISTIC given seed (no diversity/rating impact, safe in Batch Training) but an APPROXIMATION — validate A/B before trusting on final renders. Note: much of wall-clock time is outside the sampler (encode/decode), so sampler speedup ≠ total speedup. Off by default. UNVALIDATED LIVE.",
                }),
                "plateau_cache_threshold": ("FLOAT", {
                    "default": 0.975, "min": 0.5, "max": 0.999, "step": 0.005,
                    "tooltip": "Steps whose sigma is at or above this value count as the reusable plateau (matches the alg_guide_blur_sigma_threshold convention). Higher = fewer steps cached (safer, less speedup); lower = more steps cached (faster, more approximation). 0.975 catches the documented near-pure-noise plateau (schedule steps 1-5) while leaving structure formation (sigma 0.909 onward) fully computed. Only used when plateau_cache is on.",
                }),
                "taste_nearest_prompt": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL: source embed_guidance / score_slider from the taste direction learned on the prompts NEAREST this scene's prompt, instead of the single global liked-direction average. On every liked rating the Refiner records (prompt fingerprint -> that run's liked direction); with this on, each scene retrieves the similarity-weighted direction of its closest matches (a forest prompt pulls what worked on forests, not the mean across all prompts). Non-parametric retrieval — no extra model forward, just a cosine lookup + vector mean, and it can't collapse into a spurious attractor the way a value function can. Falls back to the global liked direction when no rated prompt is close enough (or the index is empty). Only affects embed_guidance / score_slider; needs refinement_key_input (or embed_guidance_source=absolute). UNVALIDATED LIVE.",
                }),
                "segmented_detailing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL ADetailer-for-video: after each scene finishes denoising, CLIPSeg (text-prompted segmentation) locates the regions named in detail_targets on a few decoded keyframes; the matched region is cut out of the video latent as a spatiotemporal tube, pushed through Lightricks' trained latent upsampler (2x spatial — the official two-stage pipeline's stage-2 model), then either used directly (detail_mode='sharpen', near-free) or re-noised + re-denoised for a 3-step tail (detail_mode='repair', default — costs ~4x tube area fraction x 3 steps, hands ~+15%). Downscaled back to its ORIGINAL latent size and pasted through the feathered CLIPSeg silhouette either way. Final resolution never changes. Tubes over detail_max_area (default 35%) are refused as a cost guard, not a content judgment. Audio untouched by construction. detail_upsampler 'auto' finds or downloads the official Lightricks upsampler (~1 GB, once) when the model's latents are LTX-width, and otherwise uses an installed upsampler without downloading anything; skips are reported loudly in console + scene report. UNVALIDATED LIVE.",
                }),
                "detail_targets": ("STRING", {
                    "default": "hands",
                    "tooltip": "Comma-separated regions to detail, in plain words ('hands', 'hands, feet', 'face'). Each becomes a CLIPSeg text query; matched regions merge into one tube per scene. CLIPSeg matches broad CLIP semantics, so malformed anatomy still lights up for its name.",
                }),
                "detail_upsampler": (cls._detail_upsampler_choices(), {
                    "default": "auto",
                    "tooltip": "Latent upsampler checkpoint from models/latent_upscale_models (the LTX 2.3 spatial upsampler used by the official two-stage workflows). 'auto' picks the newest installed spatial upscaler, or downloads the official file (~1 GB, once) when the folder is empty. Pick a file explicitly to pin it.",
                }),
                "detail_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend of the refined region into the frame at paste-back (through the feathered CLIPSeg mask). 1.0 = full replacement inside the silhouette; 0 disables the pass entirely.",
                }),
                "detail_threshold": ("FLOAT", {
                    "default": 0.35, "min": 0.05, "max": 0.9, "step": 0.05,
                    "tooltip": "CLIPSeg match confidence (post-sigmoid) required before a region counts as found. CLIPSeg's raw score for a real, correctly-named region is often well under 0.5 — if the scene report shows 'no match: max CLIPSeg score X < threshold', lower this toward X (or just below it) rather than assuming nothing is there. Lower = more permissive (more false positives on unrelated regions); higher = stricter.",
                }),
                "detail_max_area": ("FLOAT", {
                    "default": 0.35, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "Ceiling on how much of the frame the detected region may cover before the pass refuses it, as a fraction of frame area. This is a COST guardrail only (cost ~= 4x area x 3 steps, so a large region starts to rival a second full render), never a judgment about whether the region is worth detailing — if the scene report shows a region refused at some %, raise this above that % to detail it anyway (up to 1.0 = no cap, full-frame allowed).",
                }),
                "detail_denoise": ("FLOAT", {
                    "default": 0.85, "min": 0.3, "max": 0.99, "step": 0.05,
                    "tooltip": "Only used in 'repair' mode. How much noise the crop is re-noised to before the 3-step refine tail (the official LTX 2.3 two-stage recipe's own value, 0.85, is the default). Higher = more freedom for the model to genuinely reconstruct the region (fix bad anatomy) at the cost of possibly drifting from the surrounding frame; lower = closer to a plain upscale (looks 'detailed' as interpolation, but doesn't actually repair the region — if that's what you're seeing, raise this).",
                }),
                "detail_mode": (["repair", "sharpen"], {
                    "default": "repair",
                    "tooltip": "'repair' (default): upsample the crop, then re-denoise it through the video model for 3 extra steps — can genuinely fix wrong structure (bad anatomy) but costs real compute (~4x region area x 3 steps). 'sharpen': stop after the upsampler's own forward pass — no video-model calls at all, close to free — good for a region that's blurry/under-resolved but already correctly shaped; it CANNOT fix wrong structure (an extra finger stays an extra finger, just sharper), since a super-resolution net only adds detail consistent with what's already there.",
                }),
                "cut_opening_frames": ("INT", {
                    "default": 0, "min": 0, "max": 512, "step": 8,
                    "tooltip": "Let the i2v anchor do its work, then cut it out of the clip: generate the scene exactly as normal (anchor pinned at full strength the whole way, nothing weakened, no extra sampling), then drop this many frames off the FRONT of the finished clip. The anchor is a pinned latent frame at position 0 — it transfers identity, style and composition better than anything that softens it on the way in (ALG blurs it and loses character detail; Best-FaceID tokens approximate it and lose some too), but it is also literally the first frame you see, so every i2v scene opens on the exact reference still. Cutting it afterwards keeps the transfer and removes the tell: an i2v generation that reads as t2v. 0 (default) = off. The value is in REAL frames and is EXACT — N means N, with no rounding to the latent grid. The anchor itself is only the first ~8 real frames, which is usually NOT enough: it is followed by a settling-in stretch where the shot is still leaving the reference still and little is happening yet, and on a prompt that asks for immediate action that dead time is exactly what you want gone (48 was the value that worked on a 768x768x305@30 i2v chain with a quick-cut prompt — a starting point for this pipeline, not a universal default). NOTHING IS REGROWN: the scene comes out that much SHORTER than the length you asked for, and the audio is cropped to match. That is the trade — every surviving frame was generated as part of one continuous shot, with no invented ending. HOW IT IS CUT: on the DECODED frames, never on the latent, on LTX and MiniMax H3 alike. The video VAE is causal — latent frame 0 is the temporal origin — so slicing the front off the latent promoted a continuation frame to position 0 and it decoded with origin handling it was never generated for, which came out as a noisy first frame. Decoding everything first and dropping pixels afterwards leaves every surviving frame in the context it was sampled in. Consequence: the LATENT output keeps its FULL video stream, so take video from the IMAGES output on a cut run (audio from the latent as usual — its audio stream IS cropped, so sound and picture still start together). The IMAGES output must be connected or there is nothing to crop, and the run says so. Needs a pinned i2v anchor; skipped with the reason in the scene report on continuation scenes and on scenes carrying guide frames or JoyAI audio memory. On MiniMax H3 only the chain's opening is cut, not each scene's, because H3's anchor is a keyframe condition row rather than a pinned latent prefix.",
                }),
                "context_windows": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL: denoise a scene LONGER than the model's native window as overlapping context windows instead of one giant pass (ComfyUI core's own comfy.context_windows, LTX2 branch — nothing ported). Each step splits the scene into windows of context_window_length frames, denoises each, and fuses the overlaps back together. Core's LTXAV path is genuinely audio-aware: it unpacks the packed AV latent, maps each video window onto its proportional AUDIO window, and re-slices denoise masks, keyframe_idxs and guide_attention_entries per window — so i2v anchors, mid-scene guides and JoyAI memory keep working inside a window. Cost at the defaults (145/40): about 1.45x the per-frame work, because each window re-does its 40-frame overlap, offset against attention getting CHEAPER the longer the scene is (attention is quadratic in one pass, near-flat when windowed) - roughly break-even around 200 frames, a net win past ~300. Engages ONLY when the scene is longer than context_window_length — shorter scenes are untouched and pay nothing. Off by default. UNVALIDATED LIVE.",
                }),
                "context_window_length": ("INT", {
                    "default": 145, "min": 9, "max": 2049, "step": 8,
                    "tooltip": "Window size in REAL frames (must be 8n+1; core rounds down to latent frames). A scene at or below this length skips windowing entirely, so this doubles as the engage threshold. Keep it at or under the length the model actually generates well in one pass — the whole point is to stay inside that range while the scene as a whole goes past it.",
                }),
                "context_window_overlap": ("INT", {
                    "default": 40, "min": 0, "max": 512, "step": 8,
                    "tooltip": "How many real frames consecutive windows share. This is the ONLY thing carrying motion/appearance continuity across a window boundary, and it is also the only extra compute this feature costs (overlap/length = the redundant fraction). Too low and boundaries show as a seam or a motion hitch; too high and you pay for frames you already have.",
                }),
                # The three legacy spellings stay in the LIST, not just in the alias map:
                # ComfyUI validates combo values at QUEUE time, before the node ever runs, so
                # a project saved with the old name would be rejected outright and the alias
                # would never get a chance. They resolve to core's names below.
                "context_window_schedule": (["standard_uniform", "standard_static", "looped_uniform", "batched",
                                             "uniform_standard", "static_standard", "uniform_looped"], {
                    "default": "standard_uniform",
                    "tooltip": "How the windows are laid out across the scene, per step. These are ComfyUI core's own schedule names (comfy.context_windows). 'standard_uniform' (default, core's own LTXV default) shifts the window grid between steps so boundaries land in different places each step and never bake in — the safest general choice. 'standard_static' keeps the same fixed cut points every step (cheapest, but a bad boundary stays bad). 'looped_uniform' wraps the last window into the first, for seamless looping content. 'batched' denoises disjoint chunks with no overlap logic (fastest, weakest continuity). Projects saved with the old reversed spellings (uniform_standard / static_standard / uniform_looped) are still accepted and mapped onto these.",
                }),
                "context_window_fuse": (["pyramid", "relative", "flat", "overlap-linear"], {
                    "default": "pyramid",
                    "tooltip": "Weighting used to blend overlapping windows back together. 'pyramid' (default) fades each window toward its edges, so the middle of a window dominates and seams get soft. 'flat' averages equally (can smear). 'relative' and 'overlap-linear' weight by position within the overlap. Change this if boundaries look soft/ghosted rather than merely misaligned.",
                }),
                "context_window_freenoise": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Shuffle (rather than redraw) the initial noise between windows so overlapping regions start from correlated noise. Costs nothing — it is a one-time permutation of the starting noise — and is core's default for LTXV because it measurably improves how well windows blend. Turn it off only to A/B whether it is helping.",
                }),
                "context_window_retain_first": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Pin latent frame 0 (the i2v anchor) into EVERY window, in both the conditioning and the noise latent, instead of only the first window. Helps when later windows drift away from the reference image. Off by default because on a CONTINUATION scene frame 0 is the carried tail of the previous scene, not the anchor — pinning it there re-shows the same content in every window and can read as the scene going static. Turn it on if later windows lose the reference; turn it off if the scene stops moving.",
                }),
                # NOTE: append-only — keep new sampler widgets at the END of this block so the
                # builder's positional reference-workflow mapping (extract_widgets) stays aligned.
                "second_pass": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Sample each scene in TWO passes. Pass 1 runs the main sigmas schedule in full, exactly as written; pass 2 then runs the second_pass_sigmas schedule in full, exactly as written, starting from pass 1's finished clip. Nothing is cut short and nothing is derived — to make pass 1 shorter, shorten the main schedule. Total steps are simply the two schedules added up (a 9-step main plus a 4-step second pass is 13). Pass 1's finished latent is simply handed to pass 2 as its latent_image and the sampler noises it to the schedule's first sigma itself, exactly as any img2img does — there is no extra step in between. That first sigma is therefore the strength dial, and it is literal (CONST scaling: x = s*noise + (1-s)*picture): at 0.8 pass 2 starts from 80% fresh noise over 20% of the pass-1 picture and reworks the shot, looking soft if the schedule has few steps to resolve it; at 0.4 it is 40/60 and polishes; at 0.2 it is nearly pure detail work. Requires a second_pass_sigmas schedule; without one the pass is skipped with a note.",
                }),
                "second_pass_upscale": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05,
                    "tooltip": "How far the between-pass operation resamples the latent. Cost "
                               "is the SQUARE of this on upscale: 2x is four times the pixels "
                               "for pass 2, 4x is sixteen. Only upsamplers that take a factor "
                               "honour it — Lightricks' LTX one is a fixed 2x network and says "
                               "so in the scene report; MiniMax H3's resizer takes any factor "
                               "in 1.0-4.0. On 'sharpen' it is how far up the latent goes "
                               "before coming straight back, so it buys detail rather than "
                               "resolution. Latent width and height snap to even numbers, "
                               "because a patchified model cannot take an odd one."}),
                "second_pass_op": (["none", "sharpen", "upscale_2x"], {
                    "default": "none",
                    "tooltip": "OPTIONAL latent-space operation applied between the two passes — 'none' by default, nothing runs unless you pick one. 'sharpen': one forward of Lightricks' trained 2x latent upsampler, resampled straight back to the original size. No video-model calls at all, so it costs a fraction of a step; pass 2 then re-denoises the sharpened latent, which is what makes it stick. It adds detail consistent with what is already there and CANNOT fix structure that is wrong (an extra finger stays an extra finger, just sharper) — the same limit segmented detailing's sharpen mode documents. 'upscale_2x': the same upsampler, but the result is KEPT at 2x, so pass 2 runs at four times the pixels and the scene decodes at double resolution. That is 3-5x the sampling cost of the second half. The i2v pin SURVIVES it: the pinned frames are carried through the upsampler with everything else and the mask is scaled to the new grid, so pass 2 still holds the anchor — as the upscaled anchor rather than the encoded source image, which the scene report states. Guide keyframes do not survive, because they are token indices into the old grid; pass 1 uses them in full. MULTI-SCENE works with it: a scene finishes at 2x while every later scene is still built from the latent template at the original size, so everything that crosses a scene boundary (carried overlap frames, the anchor's continuation, the soft join, JoyAI memory, per-scene guide sources) is brought back to the template's grid on the way. Each scene still samples and OUTPUTS at 2x — only the carried material is resampled, and only downwards, which is the direction that survives it: those frames exist to say 'continue from here', which a resample preserves far better than invented detail would. Both ops use the same upsampler file as segmented detailing (detail_upsampler, 'auto' downloads the official Lightricks one on first use). Works on any model family, not just LTX — what it needs is an upsampler whose latents are the same width as this model's, so on MiniMax H3 install an H3 latent upsampler and pick it here; 'auto' will not download the LTX file for a model it cannot fit, and a mismatch is reported as a skip with both channel counts rather than failing the render. Video stream only; audio is never reshaped.",
                }),
                "h3_audio_clock": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL, MiniMax H3 only: integrate the audio stream on its OWN flow schedule. H3 denoises video and audio on two different schedules (the MiniMax H3 Sigma Shift node's shift_video 12 / shift_audio 3), but only one sigma grid reaches a sampler, so the model reconciles them by scaling the audio velocity by the slope between the two schedules at the START of each step. That is exact for infinitesimal steps and increasingly wrong as steps get bigger: on a 4-step schedule the last step drives audio roughly 2.5x past where its own schedule puts it, which is heard as distortion. This replaces that start-of-step slope with the one that actually spans the step, so audio lands where its schedule says. Costs one scalar multiply per step — no extra model call. Aimed at few-step schedules (turbo/distilled LoRAs, 4-8 steps), where nothing else fixes this. NO-OP when shift_video and shift_audio are EQUAL — the streams are then on one schedule and there is nothing to correct (it says so on the console). SAMPLER MATTERS, measured against a perfect predictor so the numbers are pure schedule error (audio error as a % of the stream's full range, 4/8/20 steps): WORKS BEST — FunPack Distilled Flow and Hybrid Euler 2S (runs inside their step loop, exact), and stock `euler` (85/38/14% -> 0/0/0%, exact at every step count). PERFORMS POORLY — the higher-order multistep family (`res_multistep`, `dpmpp_2m`, `gradient_estimation`, `ipndm`, `lms`, `deis`): they already absorb most of this error themselves, so the clock helps them at 4 steps (69% -> 21%) but HURTS at 20 (1% -> 15%); leave it off there. NO EFFECT — two-evals-per-step samplers (`heun`, `dpm_2`, `dpmpp_2s_ancestral`, `dpmpp_sde`, `seeds_2`): a model call cannot be tied to a step from outside their loop, so the wrapper detects that on the first call and switches itself off with a console note rather than guessing. Ancestral/SDE samplers additionally add noise to the audio stream, which this does not address. The clock never touches the video stream directly, though on H3 the two share one attention sequence, so a changed audio latent can still shift the video slightly.",
                }),
                "h3_gain_video": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "EXPERIMENTAL, MiniMax H3 only. Per-block write gain on the VIDEO rows. Every DiT block carries three sets of AdaLN modulation — one per modality (video / prompt / audio) — and the GATE is what scales that block's attention and MLP output as it is written back into a row range. This multiplies the video gate on all 50 blocks: below 1.0 each block contributes less to the picture (softer, calmer, less detail and less motion), above 1.0 more (harder, busier, and past ~1.3 it overcooks). 1.0 = untouched, and the model is not even cloned. Free — one small vector multiply per block, no extra model calls. KEYFRAME PINS AND REFERENCE IMAGES RIDE THE VIDEO TAG, so this moves them with the picture; they are separated from the target video by timestep row, not by modality, and only early in the schedule. Attached with add_object_patch, which ComfyUI restores on unpatch, so nothing survives the run.",
                }),
                "h3_gain_prompt": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "EXPERIMENTAL, MiniMax H3 only. Per-block write gain on the PROMPT rows (see h3_gain_video for the mechanism). H3 keeps the text in the same sequence as the picture and lets every block rewrite it, so the prompt the model reads at block 50 is not the one the encoder produced. 0.0 freezes the text at its encoded value for the whole forward — the video still attends to it, but it can no longer drift toward what is being drawn. Lower values are the lever to try when a scene slowly stops matching what you asked for; higher values let the text move further with the picture. 1.0 = untouched. Free. The <Picture N> vision rows inside the text span carry the VIDEO tag, not this one, so a reference is not affected.",
                }),
                "h3_gain_audio": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "EXPERIMENTAL, MiniMax H3 only. Per-block write gain on the AUDIO rows (see h3_gain_video for the mechanism). This is the one modality whose gate can be moved without touching the other two at the point of writing — though the streams still share one attention pass, so a changed audio row is read by the video on the NEXT block. Below 1.0 the soundtrack is built more conservatively. Free. 1.0 = untouched.",
                }),
                "h3_taste_bias": ("FLOAT", {
                    "default": 0.0, "min": -0.30, "max": 0.30, "step": 0.01,
                    "tooltip": "EXPERIMENTAL, MiniMax H3 only. Pushes the prompt toward what you have RATED WELL, in the model's own space. The Refiner already learns a 'liked direction' from your ratings and steers the conditioning along it — but that edit then goes through H3's token refiner, whose two attention blocks mix it across tokens and whose final norm scales its magnitude away, so what the model reads is not what was learned. This adds the same direction AFTER the refiner, where it lands as sent. Measured as a fraction of a typical prompt row's magnitude, so it means the same thing on every prompt and every checkpoint. Negative pushes AWAY from what you liked. Needs a refinement key with at least 3 liked runs; without one it does nothing and says so. Free. IGNORED unless h3_gain_mode is 'manual' — in 'learned' mode this is learned from your ratings like the other gains. 0.0 = off.",
                }),
                "h3_gain_mode": (["learned", "manual"], {
                    "default": "learned",
                    "tooltip": "Where the seven H3 render strengths come from. 'learned' (default) takes them from the refinement key: Studio learns them from your ratings alone and tags them onto the conditioning, and the seven widgets below are IGNORED — nothing to tune by hand, which is the point. Six of the seven are learned (h3_prompt_time is not — its off value sits at the end of its own range, not the middle, so the loop cannot explore around it safely; it stays a manual dial). Six scalars is a smaller search than the sigma schedule already learns from ratings, so it converges in tens of rated runs. With no key wired, or before the first rating, 'learned' renders at the model's trained strengths (untouched, and the taste push at 0). 'manual' ignores the learned values and uses the widgets below exactly as set — for deliberately probing one strength, or for a graph with no Refiner in it.",
                }),
                "h3_prompt_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "EXPERIMENTAL, MiniMax H3 only. How LOUD the prompt is when the picture reads it. H3 puts the Qwen text through condition_proj + a 2-block token refiner (ending in an RMSNorm) before the DiT sees it, and the blocks then read the text through attention — so the magnitude of those refined rows is how strongly the prompt competes for attention against the picture and the reference. This multiplies them: above 1.0 the prompt is harder to ignore, below 1.0 it recedes and the reference or the anchor gets more say. Applied AFTER the refiner's final norm, so it lands as set rather than being renormalized away. 1.0 = untouched, and the model is not cloned. Free — one multiply on the text rows, no extra model calls. Confined to the PROMPT rows: the <Picture N> label and vision block sit ahead of them and are left alone (read from minimax_token_tags; if those cannot be read the whole text span is scaled and the console says so). DIFFERENT from h3_gain_prompt, which controls how much each block WRITES BACK into the text rows — this controls how loudly they are READ.",
                }),
                "h3_prompt_time": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "EXPERIMENTAL, MiniMax H3 only. Makes the picture stick closer to the prompt, and drift away from it less as a scene goes on. 0.0 = off (default). Try 0.9-1.0; the effect gets stronger the closer to 1.0. HOW: H3 tells every part of its sequence how finished it is, and it tells the prompt whatever it tells the picture — so early on, while the picture is still noise, the prompt is treated as unreliable too. This tells the prompt it is finished no matter where the picture is, so the model leans on it from the first step. Free: one extra row through a small projection per block, no extra model calls. Does not touch the reference image or a keyframe pin. UNVALIDATED.",
                }),
                "h3_video_detail": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "EXPERIMENTAL, MiniMax H3 only. Makes the picture crisper or softer WITHOUT changing the sound at all. Above 1.0 = more detail and contrast (past ~1.3 it overcooks), below 1.0 = softer and calmer. 1.0 = untouched, and the model is not cloned. This is the audio-safe twin of h3_gain_video: everything earlier in the model shares one attention pass, so a change to the picture reaches the soundtrack, but this runs after the last one — there is no path left for it to travel. Free. UNVALIDATED.",
                }),
                "alg_anchor": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL: run ALG's i2v anchor blur (arXiv:2506.08456) on WHATEVER sampler is wired — a stock KSampler with any sampler_name, Hybrid Euler 2S, a two-evals-per-step sampler like heun, anything. The blur de-statics an anchored scene by hiding the anchor's high-frequency detail during the near-pure-noise steps, so the model cannot shortcut to a video that just matches the still. It is the same guidance as the FunPack Distilled Flow sampler's own alg_enabled, and this switch drives that one too when Distilled Flow is the wired sampler, so there is one control wherever you are. The swap is decided by the step's sigma alone, which is an argument of every model call, so it does not need to run inside a sampler's loop. No effect on a scene with no i2v anchor.",
                }),
                "alg_anchor_strength": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1,
                    "tooltip": "Downsample factor for the anchor blur (alg_anchor). Paper default is 2.5; 2.0 held character/i2v consistency noticeably better in testing here. Higher = blurrier anchor during the affected steps.",
                }),
                "alg_anchor_sigma_threshold": ("FLOAT", {
                    "default": 0.975, "min": 0.5, "max": 0.999, "step": 0.005,
                    "tooltip": "The anchor stays blurred while sigma is above this value (the near-pure-noise steps), then swaps to sharp. Higher = narrower blurred window. Independent of the guide-frame window (alg_guide_blur_sigma_threshold).",
                }),
                # A connection socket, never a widget — safe at the end, and it must stay after
                # every widget above (see the widgets_values note at the top of this block).
                "second_pass_sigmas": ("SIGMAS", {
                    "tooltip": "The schedule pass 2 runs — required for second_pass, and it is run EXACTLY as written, high to low, ending at 0. Wire any scheduler here, or type sigmas in the Editor. Pass 1 has already finished the main schedule by this point, so pass 2 starts from a clean clip and re-enters by re-noising it up to this schedule's FIRST sigma: that value is the strength dial (near 1.0 reworks the shot, low values only polish it), and the rest of the schedule sets how many steps it gets. A schedule that ascends, or that stops above 0, is refused with the reason — both would silently produce a distorted or under-denoised clip rather than fail loudly.",

                }),
                "second_pass_sampler": ("SAMPLER", {
                    "tooltip": "Optional: a DIFFERENT sampler for pass 2. Left unwired, pass 2 reuses the sampler above — the old behaviour. Wiring one lets the two passes use different algorithms, usually because what builds a shot well is not what finishes it: a distilled few-step sampler for pass 1 and an ordinary KSampler with more steps for the polish, or the reverse. It changes the algorithm only; pass 2's schedule is still second_pass_sigmas. No effect when second_pass is off.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("latent", "images", "status", "scene_count", "scene_report", "scene_boundaries")
    FUNCTION = "sample"
    CATEGORY = "FunPack/Sampling"
    DESCRIPTION = (
        "Samples multi-entry scene conditioning as a smooth LTXV/LTXAV continuation chain. "
        "Use with FunPack Studio split-by-transitions output."
    )

    @classmethod
    def _identity_projector_choices(cls):
        try:
            import folder_paths
            return ["None"] + folder_paths.get_filename_list("loras")
        except Exception:
            return ["None"]

    @classmethod
    def _detail_upsampler_choices(cls):
        # "auto": prefer an installed spatial upscaler, else download the official
        # Lightricks file on first use. ("None" from older saved workflows/projects is
        # treated as auto too — the enable toggle is the only off switch.)
        try:
            import folder_paths
            return ["auto"] + folder_paths.get_filename_list("latent_upscale_models")
        except Exception:
            return ["auto"]

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

    # Time axis per latent stream, in comfy's pack order (video first). LTXAV puts time
    # on dim 2 for BOTH streams (its audio latent is [B, C, T, freq]); MiniMax H3 puts the
    # STEREO CHANNEL there and time last ([B, 32, 2, T]), so slicing an H3 audio stream on
    # dim 2 empties it instead of trimming its tail. Both are 4-D, so the tensor cannot
    # tell you which — it is set per run from the model family in sample().
    _time_dims = (2, 2, 2, 2)

    def _stream_dim(self, stream=0):
        dims = self._time_dims
        idx = max(0, int(stream))
        return dims[idx] if idx < len(dims) else dims[-1]

    def _set_stream_axes(self, model):
        """Record each stream's time axis for this run. Returns True when H3 was detected."""
        try:
            from .minimax_h3 import is_h3_model, stream_time_dims
        except ImportError:
            from minimax_h3 import is_h3_model, stream_time_dims
        h3 = bool(is_h3_model(model))
        self._time_dims = tuple(stream_time_dims(4, h3))
        if not h3:
            # Teach the conditioning module this model's real video/audio text-context split
            # instead of letting it guess from a static width table. From LTX 2.5 on the widths
            # come out of the checkpoint, so an unseen pair would silently drop audio protection.
            try:
                from .conditioning import register_ltxav_split_from_model
            except ImportError:
                from conditioning import register_ltxav_split_from_model
            try:
                register_ltxav_split_from_model(model)
            except Exception:
                pass
        return h3

    def _tensor_frames(self, tensor, stream=0):
        if not isinstance(tensor, torch.Tensor) or tensor.dim() < 3:
            raise ValueError("Scene chain latents must have a time dimension at index 2.")
        return int(tensor.shape[self._stream_dim(stream)])

    def _context_scene_latent_frames(self, chunk):
        """Video latent frames in this scene's chunk, or None if it can't be read.

        Reported only — core decides for itself whether a scene is long enough to window.
        Read off the VIDEO tensor (index 0 of the nested AV latent), while the chunk is
        still unpacked 5D; after comfy packs it the time axis is gone.
        """
        try:
            return self._tensor_frames(self._latent_tensors(chunk)[0])
        except Exception:
            return None

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

    def _expected_latent_frames(self, pixel_frames, time_scale, vae=None):
        """Pixel frames -> video latent frames.

        The uniform ((n - 1) // time_scale) + 1 is LTXAV's 8k+1 grid. It is not universal:
        MiniMax H3 is 17k+5 -> 5k+2, and its VAE reports downscale_index_formula[0] = 4
        (the INDEX map, not the count map), so the uniform form silently under-counts by
        about a fifth. When the VAE exposes its own count map we ask it instead — which
        reproduces LTXAV's answer exactly, so nothing changes for existing projects.
        """
        if vae is not None:
            try:
                from .minimax_h3 import latent_frames_from_vae
            except ImportError:
                from minimax_h3 import latent_frames_from_vae
            counted = latent_frames_from_vae(vae, pixel_frames)
            if counted is not None:
                return counted
        return ((max(1, int(pixel_frames)) - 1) // max(1, int(time_scale))) + 1

    def _validate_template_length(self, latent_template, num_frames_per_scene, time_scale, vae=None):
        video_frames = self._tensor_frames(self._latent_tensors(latent_template)[0])
        expected = self._expected_latent_frames(num_frames_per_scene, time_scale, vae=vae)
        if video_frames != expected:
            detail = ""
            if self._is_h3:
                # By this point num_frames_per_scene is already snapped to 17k+5, so a
                # mismatch means the latent node was built for a different length entirely.
                try:
                    from .minimax_h3 import FRAME_GRID, FRAME_BASE
                except ImportError:
                    from minimax_h3 import FRAME_GRID, FRAME_BASE
                detail = (f" On MiniMax H3 the latent node's `length` must be the SAME "
                          f"{FRAME_GRID}k+{FRAME_BASE} frame count as num_frames_per_scene "
                          f"({num_frames_per_scene}) — set Empty MiniMax H3 AV Latent to that.")
            raise ValueError(
                f"latent_template has {video_frames} video latent frames, expected {expected} "
                f"from num_frames_per_scene={num_frames_per_scene} and time scale={time_scale}."
                + detail
            )
        return video_frames

    def _overlap_frames(self, latent_template, frame_overlap, time_scale, vae=None):
        video_frames = self._tensor_frames(self._latent_tensors(latent_template)[0])
        overlap = self._expected_latent_frames(frame_overlap + 1, time_scale, vae=vae) - 1
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

    def _replace_start(self, target, source_tail, overlap, stream=0):
        if overlap <= 0:
            return target
        target = target.clone()
        source_tail = source_tail.to(device=target.device, dtype=target.dtype)
        return self._set_time_slice(target, 0, overlap, source_tail, stream=stream)

    def _tail(self, tensor, overlap, stream=0):
        if overlap <= 0:
            return self._time_slice(tensor, 0, 0, stream=stream)
        return self._time_slice(tensor, -overlap, None, stream=stream)

    def _time_slice(self, tensor, start, end, stream=0):
        slices = [slice(None)] * tensor.dim()
        slices[self._stream_dim(stream) % tensor.dim()] = slice(start, end)
        return tensor[tuple(slices)]

    def _set_time_slice(self, tensor, start, end, value, stream=0):
        slices = [slice(None)] * tensor.dim()
        slices[self._stream_dim(stream) % tensor.dim()] = slice(start, end)
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

    def _make_mask_tensor(self, tensor, overlap, stream=0):
        mask = torch.ones_like(tensor)
        if overlap > 0:
            self._set_time_slice(mask, 0, overlap, 0, stream=stream)
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

    def _match_template_resolution(self, latent, template):
        """Resample a finished scene's VIDEO stream back onto `template`'s spatial grid.

        This is what makes multi-scene work with a second pass. A resolution-changing
        second_pass_op (upscale_2x) hands back a scene at twice the size, but every LATER
        scene is still built from `latent_template` at the original size — so the carried
        overlap frames, the anchor's continuation and the JoyAI memory frame no longer fit
        the chunk they are spliced into, and the chain died on the shape mismatch. Bringing
        the carried material back to the template's grid is the whole fix: each scene still
        SAMPLES and OUTPUTS at the upscaled size, only what crosses a scene boundary is
        brought back.

        Downscaling is the right direction to lose information in: the carried frames exist
        to say "continue from here", and that survives a resample far better than the detail
        the second pass added would survive being invented at the wrong scale.

        Audio has no spatial axes and is never touched. Returns the input object unchanged
        when the grids already agree, so any run without a resolution-changing op is
        bit-identical.
        """
        try:
            tensors = self._latent_tensors(latent)
            video = tensors[0]
            th, tw = self._latent_tensors(template)[0].shape[-2:]
        except Exception as _e:
            _log.failed("FunPackSceneChain", "template resolution match", _e,
                        "the scene is spliced at its own resolution — a mismatch here is the "
                        "source of a size error later in the chain")
            return latent
        if video.shape[-2:] == (th, tw):
            return latent
        try:
            try:
                from .detailing import _downscale_to
            except ImportError:
                from detailing import _downscale_to
            if video.shape[-2] >= th and video.shape[-1] >= tw:
                resized = _downscale_to(video, int(th), int(tw))
            else:
                # Not reachable from any current op (upscale_2x only grows), but a scene
                # SMALLER than the template would corrupt the splice just as surely, so
                # handle it rather than pass a wrong shape through. Antialiasing is a
                # downscale-only concept, hence the separate call.
                b, c, f, _h, _w = video.shape
                resized = torch.nn.functional.interpolate(
                    video.permute(0, 2, 1, 3, 4).reshape(b * f, c, _h, _w),
                    size=(int(th), int(tw)), mode="bicubic", align_corners=False,
                ).reshape(b, f, c, int(th), int(tw)).permute(0, 2, 1, 3, 4)
        except Exception as exc:  # noqa: BLE001
            print(f"[FunPackSceneChain] could not bring the previous scene back to the "
                  f"template's {th}x{tw} grid ({type(exc).__name__}: {exc}) — this scene's "
                  f"continuity (carried frames / anchor overlap / JoyAI memory) will fail.")
            return latent
        result = self._clone_latent(latent)
        out = self._latent_tensors(result)
        out[0] = resized.to(device=video.device, dtype=video.dtype)
        if self._is_nested(result.get("samples")):
            result["samples"] = comfy.nested_tensor.NestedTensor(out)
        else:
            result["samples"] = out[0]
        # A mask describes the OLD grid; the caller rebuilds one for the chunk it is
        # splicing into, and a stale one here would be applied at the wrong scale.
        result.pop("noise_mask", None)
        return result

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
            tensor_frames = self._tensor_frames(tensor, stream=index)
            overlap = video_overlap if index == 0 else self._derived_overlap(video_overlap, video_frames, tensor_frames)
            prev_tail = self._tail(previous_tensors[index], overlap, stream=index)
            out_tensor = self._replace_start(tensor, prev_tail, overlap, stream=index)
            mask_tensor = self._make_mask_tensor(tensor, overlap, stream=index)
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


    def _prepend_soft_continuation(self, chunk, previous, mask_value=0.4, n_frames=4):
        chunk_tensors = self._latent_tensors(chunk)
        previous_tensors = self._latent_tensors(previous)
        if not chunk_tensors or not previous_tensors:
            return chunk, 0
        prev_len = self._tensor_frames(previous_tensors[0])
        count = min(n_frames, prev_len)
        if count <= 0:
            return chunk, 0
        soft_frames = self._tail(previous_tensors[0], count).to(
            device=chunk_tensors[0].device, dtype=chunk_tensors[0].dtype,
        )
        out_tensors = list(chunk_tensors)
        out_masks = self._latent_masks(chunk, len(out_tensors))
        if out_masks[0] is None:
            out_masks[0] = torch.ones_like(out_tensors[0])
        soft_mask = torch.full_like(soft_frames, mask_value)
        out_tensors[0] = torch.cat([soft_frames, out_tensors[0]], dim=2)
        out_masks[0] = torch.cat([soft_mask, out_masks[0].to(soft_mask.device, soft_mask.dtype)], dim=2)
        if self._is_nested(chunk.get("samples")):
            chunk["samples"] = comfy.nested_tensor.NestedTensor(out_tensors)
            chunk["noise_mask"] = comfy.nested_tensor.NestedTensor(out_masks)
        else:
            chunk["samples"] = out_tensors[0]
            chunk["noise_mask"] = out_masks[0]
        return chunk, count

    def _h3_carry_i2v_guide(self, chunk, template, positive, negative):
        """carry_i2v_guides on H3: hold the ORIGINAL reference, as a frame-0 pin.

        The LTX path below prepends the template's protected frames to the chunk. On H3 that
        adds video latent frames the packed layout never accounted for — off its 5k+2 grid —
        and attaches LTX conditioning this model does not read, so it damaged the latent
        instead of carrying anything. The same intent (every scene keeps the opening
        reference) is a keyframe pin, and nothing is appended, so no tail is cropped later.

        It claims frame 0 ahead of the continuation pin: continuing from the previous shot is
        the default, holding the original reference is what the user asked for by name.
        """
        tensors = self._latent_tensors(template)
        masks = self._latent_masks(template, len(tensors))
        if not tensors or getattr(tensors[0], "ndim", 0) < 5:
            return chunk, positive, negative, 0
        protected = self._protected_prefix_frames(masks[0], self._tensor_frames(tensors[0]))
        if protected <= 0:
            return chunk, positive, negative, 0
        frame = self._time_slice(tensors[0], 0, 1)[:1].clone()
        positive = self._h3_add_keyframes(
            positive, [{"resolved_frame_index": 0, "latent": frame}], self._h3_frame_count)
        return chunk, positive, negative, 0

    def _append_i2v_guides(self, chunk, template, positive, negative):
        if self._is_h3:
            return self._h3_carry_i2v_guide(chunk, template, positive, negative)
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
        target_mask = self._time_slice(out_masks[0], 0, protected).to(guide_mask.device, guide_mask.dtype)
        guide_mask = self._expand_mask_like(guide_mask, target_mask)
        # Prepend so guide is temporal pos 0, overlap frames follow at 1, 9, 17... — no conflict.
        out_tensors[0] = torch.cat([guide, out_tensors[0]], dim=2)
        out_masks[0] = torch.cat([guide_mask, out_masks[0].to(guide_mask.device, guide_mask.dtype)], dim=2)

        if self._is_nested(chunk.get("samples")):
            chunk["samples"] = comfy.nested_tensor.NestedTensor(out_tensors)
            chunk["noise_mask"] = comfy.nested_tensor.NestedTensor(out_masks)
        else:
            chunk["samples"] = out_tensors[0]
            chunk["noise_mask"] = out_masks[0]

        return chunk, positive, negative, protected

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

    def _crop_video_head(self, latent, count):
        if count <= 0:
            return latent
        result = self._clone_latent(latent)
        tensors = self._latent_tensors(result)
        tensors[0] = tensors[0][:, :, count:]
        if self._is_nested(result.get("samples")):
            result["samples"] = comfy.nested_tensor.NestedTensor(tensors)
        else:
            result["samples"] = tensors[0]
        return result

    def _crop_audio_tail(self, latent, count):
        """Drop the trailing `count` injected JoyAI audio-memory frames from the audio stream
        (tensors[1]) after sampling — the audio analogue of _crop_video_tail. No-op for
        single-stream LTXV (no audio tensor)."""
        if count <= 0:
            return latent
        result = self._clone_latent(latent)
        tensors = self._latent_tensors(result)
        if len(tensors) < 2 or tensors[1] is None:
            return latent
        tensors[1] = self._time_slice(tensors[1], 0, -count, stream=1)
        if self._is_nested(result.get("samples")):
            result["samples"] = comfy.nested_tensor.NestedTensor(tensors)
        return result

    def _slerp_block(self, a, b, t):
        """Spherical interpolation of two latent blocks along the feature volume, per (batch, frame).

        A linear crossfade of two different latents superimposes them and lets their magnitude
        collapse near alpha~0.5 -> ghosting + washed-out seam. Slerp interpolates the DIRECTION on
        the hypersphere and the MAGNITUDE separately, so the seam keeps its energy. Reduces over the
        feature dims (C,H,W), keeping batch (0) and time (2). `t` is the fraction toward `b`,
        broadcastable over the time axis. Math runs in float32 for stability, cast back to a.dtype."""
        out_dtype = a.dtype
        a = a.float()
        b = b.to(a.device).float()
        dims = tuple(d for d in range(a.dim()) if d not in (0, 2))
        eps = 1e-8
        # vector_norm (not Tensor.norm) — the latter treats a 3-tuple dim as a matrix norm.
        a_norm = torch.linalg.vector_norm(a, dim=dims, keepdim=True)
        b_norm = torch.linalg.vector_norm(b, dim=dims, keepdim=True)
        a_u = a / a_norm.clamp_min(eps)
        b_u = b / b_norm.clamp_min(eps)
        dot = (a_u * b_u).sum(dim=dims, keepdim=True).clamp(-1.0, 1.0)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        denom = sin_theta.clamp_min(eps)
        s0 = torch.sin((1.0 - t) * theta) / denom
        s1 = torch.sin(t * theta) / denom
        direction = torch.where(
            sin_theta.abs() < 1e-4,                      # nearly collinear -> plain lerp of dirs
            (1.0 - t) * a_u + t * b_u,
            s0 * a_u + s1 * b_u,
        )
        mag = (1.0 - t) * a_norm + t * b_norm            # magnitude interpolated linearly
        return (direction * mag).to(out_dtype)

    def _blend_tensors(self, left, right, overlap, use_slerp=True, stream=0):
        dim = self._stream_dim(stream) % left.dim()
        if overlap <= 0:
            return torch.cat([left, right], dim=dim)
        if [s for d, s in enumerate(left.shape) if d != dim] != \
                [s for d, s in enumerate(right.shape) if d != dim]:
            raise ValueError("Cannot blend scene latents with different non-time dimensions.")
        right = right.to(left.device, left.dtype)
        left_ov = self._time_slice(left, -overlap, None, stream=stream)
        right_ov = self._time_slice(right, 0, overlap, stream=stream)
        shape = [1] * left.dim()
        shape[dim] = overlap
        if use_slerp:
            # Smoothstep ramp 0->1 (eases in/out so less dwell at the 50/50 ghost point), then
            # slerp so the crossfade preserves latent magnitude instead of washing out.
            lin = torch.linspace(0.0, 1.0, overlap + 2, device=left.device, dtype=left.dtype)[1:-1]
            t = (lin * lin * (3.0 - 2.0 * lin)).reshape(shape)
            blended = self._slerp_block(left_ov, right_ov, t)
        else:
            # Audio (and any non-video stream) keeps the original linear crossfade untouched.
            alpha = torch.linspace(1.0, 0.0, overlap + 2, device=left.device, dtype=left.dtype)[1:-1].reshape(shape)
            blended = alpha * left_ov + (1.0 - alpha) * right_ov
        return torch.cat([self._time_slice(left, 0, -overlap, stream=stream), blended,
                          self._time_slice(right, overlap, None, stream=stream)], dim=dim)

    def _blend_latents(self, previous, current, video_overlap):
        result = self._clone_latent(previous)
        previous_tensors = self._latent_tensors(previous)
        current_tensors = self._latent_tensors(current)
        if len(previous_tensors) != len(current_tensors):
            raise ValueError("Cannot blend different latent structures.")

        video_frames = self._tensor_frames(current_tensors[0])
        blended_tensors = []
        for index, tensor in enumerate(current_tensors):
            tensor_frames = self._tensor_frames(tensor, stream=index)
            overlap = video_overlap if index == 0 else self._derived_overlap(video_overlap, video_frames, tensor_frames)
            # index 0 is the video latent -> slerp+smoothstep; any further stream (audio) stays
            # on the untouched linear crossfade (audio-safety: never reshape audio nonlinearly).
            blended_tensors.append(self._blend_tensors(previous_tensors[index], tensor, overlap,
                                                      use_slerp=(index == 0), stream=index))

        if self._is_nested(previous.get("samples")):
            result["samples"] = comfy.nested_tensor.NestedTensor(blended_tensors)
        else:
            result["samples"] = blended_tensors[0]
        result.pop("noise_mask", None)
        return result

    def _set_phase(self, label):
        """Publish what is running right now for the editor's progress readout.

        The ComfyUI progress channel is (value, max) only, so without this a multi-scene
        chain — especially one sampling some scenes twice — is a single anonymous bar.
        Best-effort in every direction: a missing module or a failed write must never
        affect the render.
        """
        try:
            try:
                from . import run_phase as _rp
            except ImportError:  # loaded as a top-level module (tests, direct import)
                import run_phase as _rp
            _rp.set_phase(label)
        except Exception:  # noqa: BLE001
            pass

    def _current_phase_label(self):
        """The progress label this chunk is running under ("scene 2/3 · pass 1 of 2"), for
        error messages. Best-effort like _set_phase — never let a readout break a render."""
        try:
            try:
                from . import run_phase as _rp
            except ImportError:
                import run_phase as _rp
            return str((_rp.current() or {}).get("label") or "")
        except Exception:  # noqa: BLE001
            return ""

    def _nonfinite_report(self, samples):
        """['video (12/1000 values, dtype=...)'] for each stream carrying NaN/Inf. Empty when
        clean, and empty when the latent is not readable — the check must never be the thing
        that kills a render."""
        try:
            tensors = self._latent_tensors({"samples": samples})
        except Exception:  # noqa: BLE001
            return []
        names = ("video", "audio")
        bad = []
        for i, t in enumerate(tensors):
            if not isinstance(t, torch.Tensor):
                continue
            finite = torch.isfinite(t)
            if bool(finite.all()):
                continue
            n_bad = int((~finite).sum().item())
            label = names[i] if i < len(names) else f"stream {i}"
            bad.append(f"{label} ({n_bad}/{t.numel()} values, dtype={t.dtype})")
        return bad

    def _assert_finite_inputs(self, samples, sigmas):
        """Check what the chunk is ABOUT TO be sampled from, so a corrupt run says whether the
        damage arrived or was produced here.

        Without this the output check alone cannot tell "the model computed garbage" from "the
        model was handed garbage and faithfully propagated it" — and those have disjoint
        suspect lists. The schedule is checked too because it is hand-typed: our own solvers
        divide by sigma (er = sigma_next / sigma), so an INTERIOR zero or a repeat is an
        instant Inf that then poisons every value in the tensor. A trailing zero is fine — it
        is only ever the target of the last step, never a divisor.
        """
        where = self._current_phase_label() or "this chunk"
        bad = self._nonfinite_report(samples)
        if bad:
            raise RuntimeError(
                f"[FunPackSceneChain] {where}: the latent handed to the sampler is ALREADY "
                f"non-finite before any sampling — {'; '.join(bad)}. The model has not run "
                f"yet, so this is not the checkpoint, the LoRA or the sampler. Look at what "
                f"produced this latent: the empty-latent node, the i2v anchor encode, or (on "
                f"a later scene) the previous scene's output."
            )
        if not isinstance(sigmas, torch.Tensor) or sigmas.numel() < 2:
            return
        s = sigmas.detach().float().reshape(-1).cpu()
        if not bool(torch.isfinite(s).all()):
            raise RuntimeError(
                f"[FunPackSceneChain] {where}: the sigma schedule contains NaN/Inf "
                f"({[float(v) for v in s]}). Fix the schedule — nothing downstream can."
            )
        interior = s[:-1]
        if bool((interior <= 0).any()):
            raise RuntimeError(
                f"[FunPackSceneChain] {where}: the sigma schedule has a zero or negative "
                f"value before its last entry ({[float(v) for v in s]}). Every value except "
                f"the final one is divided by, so this produces Inf on that step and NaN "
                f"everywhere after it. Only the LAST sigma may be 0."
            )
        if bool((s[1:] >= s[:-1]).any()):
            raise RuntimeError(
                f"[FunPackSceneChain] {where}: the sigma schedule is not strictly "
                f"descending ({[float(v) for v in s]}). A repeated value makes a zero-length "
                f"step the solvers divide by; an ascending one integrates backwards."
            )

    def _assert_finite_sample(self, sampled):
        """Stop the run the moment a chunk comes back with NaN/Inf in it.

        Nothing downstream notices: the blend spreads it into every previously finished
        scene, the VAE decodes it, and the first thing that objects is ffmpeg's AAC encoder
        with a message naming nothing that produced it ("Input contains (near) NaN/+-Inf") —
        after the whole montage has been paid for. The video half is worse: NaN through
        `astype(np.uint8)` is undefined but silent, so it degrades without complaining.

        A non-finite latent is unrecoverable, so this raises rather than warns. One
        isfinite() per chunk against a multi-second sample is not a cost worth weighing.
        """
        bad = self._nonfinite_report(sampled)
        if not bad:
            return
        where = self._current_phase_label() or "this chunk"
        raise RuntimeError(
            f"[FunPackSceneChain] {where}: the sampler returned a non-finite latent — "
            f"{'; '.join(bad)}. Its inputs were checked and were clean, so this was produced "
            f"HERE, by the model or the sampler math. Usual causes, in the order worth "
            f"testing: a LoRA that does not match this checkpoint (ComfyUI logs the keys "
            f"it could not apply at load — check the log), a base checkpoint whose layout "
            f"the loader mis-inferred (third-party repacks of quantised weights are the "
            f"common case), or a VAE/model dtype the model cannot hold. Stopping here "
            f"rather than blending this into the finished scenes and failing later in "
            f"ffmpeg with nothing to point at."
        )

    def _sample_chunk(self, model, sampler, sigmas, seed, cfg, positive, negative, latent,
                      pbar=None, step_offset=0, alg_guide_tail_frames=0,
                      alg_guide_blur_strength=2.0, alg_guide_blur_sigma_threshold=0.975,
                      alg_anchor=False, alg_anchor_strength=2.0,
                      alg_anchor_sigma_threshold=0.975,
                      bounded_attention_enabled=False, h3_audio_clock=False):
        if sampler is None:
            raise ValueError("sampler input is required.")
        if not isinstance(sigmas, torch.Tensor):
            raise ValueError("sigmas input must be a SIGMAS tensor.")
        latent = self._clone_latent(latent)
        samples = latent["samples"]
        self._assert_finite_inputs(samples, sigmas)
        noise = comfy.sample.prepare_noise(samples, int(seed))

        def _progress_cb(step, _denoised, _x, _total_steps):
            if pbar is not None:
                pbar.update_absolute(step_offset + int(step) + 1)

        # EXPERIMENTAL alg_blur_guides: this chunk-specific guide-tail count has to reach
        # sample_funpack_distilled_flow somehow, but `sampler` is a single KSAMPLER object
        # built once (per Studio pass) and reused across every chunk — its extra_options dict
        # is read fresh on every .sample() call (see comfy KSAMPLER.sample), so mutating it
        # right before this call (and always resetting it, even to 0) is safe and chunk-scoped,
        # with no risk of leaking into the next chunk or a differently-configured sampler.
        extra_options = getattr(sampler, "extra_options", None)
        _wired_fn = getattr(sampler, "sampler_function", None)
        _alg_in_sampler = _wired_fn is sample_funpack_distilled_flow
        if isinstance(extra_options, dict) and _alg_in_sampler:
            extra_options["alg_guide_tail_frames"] = int(alg_guide_tail_frames)
            extra_options["alg_guide_blur_strength"] = float(alg_guide_blur_strength)
            extra_options["alg_guide_blur_sigma_threshold"] = float(alg_guide_blur_sigma_threshold)
            # alg_anchor is the same guidance as this sampler's own alg_enabled, so on
            # Distilled Flow it drives that rather than a second, competing blur. Only when
            # ON — off leaves the sampler node's own setting exactly as the user left it.
            if alg_anchor:
                extra_options["alg_enabled"] = True
                extra_options["alg_strength"] = float(alg_anchor_strength)
                extra_options["alg_sigma_threshold"] = float(alg_anchor_sigma_threshold)
        # h3_audio_clock reaches a FunPack sampler the same way, but it belongs to every one
        # of them that can run on H3 (see _AUDIO_CLOCK_SAMPLERS), not just this one.
        if isinstance(extra_options, dict) and getattr(sampler, "sampler_function", None) in _AUDIO_CLOCK_SAMPLERS:
            extra_options["h3_audio_clock"] = bool(h3_audio_clock)
        elif h3_audio_clock:
            # A stock comfy sampler has no such option, so wrap its model instead. The wrapper
            # verifies per call that this sampler evaluates once per step and stands down if
            # not, so handing it an unsuitable sampler costs the correction, never the run.
            wrapped = _audio_clock_wrap_sampler(sampler)
            if wrapped is not None:
                if not self._h3_clock_unreachable_noted:
                    self._h3_clock_unreachable_noted = True
                    _fn_name = getattr(getattr(sampler, "sampler_function", None), "__name__", "")
                    if _fn_name in _AUDIO_CLOCK_HARMED:
                        print(f"[FunPackSceneChain] h3_audio_clock is on with {_fn_name} — a "
                              f"higher-order sampler, which already absorbs most of this error "
                              f"on its own once the schedule is fine. Measured: it HELPS at ~4 "
                              f"steps (69% -> 20% audio error) but HURTS at ~20 (1% -> 15%). "
                              f"At 20 steps leave the clock off with this sampler; for few-step "
                              f"turbo runs prefer euler, where the clock is exact.")
                    elif _fn_name not in _AUDIO_CLOCK_EXACT_ON:
                        print(f"[FunPackSceneChain] h3_audio_clock is on with {_fn_name}. Only "
                              f"euler (and FunPack's own samplers) are exactly corrected; this "
                              f"one is unmeasured, and ancestral/SDE samplers also add noise to "
                              f"the audio stream, which the clock does not address.")
                sampler = wrapped
            elif not self._h3_clock_unreachable_noted:
                self._h3_clock_unreachable_noted = True
                print("[FunPackSceneChain] h3_audio_clock is on, but the wired SAMPLER exposes "
                      "no sampler_function to wrap, so the correction cannot be applied to it. "
                      "Sampling continues without it.")
        # EXPERIMENTAL ALG on a sampler whose loop we cannot get inside. ALG lived in the
        # Distilled Flow loop only because that is where it was written: the blurred/sharp
        # swap is decided by the step's sigma, and sigma is an argument of every model call,
        # so a denoiser proxy can drive it from outside any sampler. That is what makes the
        # anchor blur (and the guide-tail blur, same mechanism) available with a stock
        # KSampler, with Hybrid Euler 2S, and with two-evals-per-step samplers. Wrapped LAST
        # so h3_audio_clock above still sees the real sampler function and takes its exact
        # in-loop path where it has one.
        _alg_tail = int(alg_guide_tail_frames)
        if not _alg_in_sampler and (alg_anchor or _alg_tail > 0):
            _alg_wrapped = _alg_wrap_sampler(
                sampler, bool(alg_anchor), alg_anchor_strength, alg_anchor_sigma_threshold,
                _alg_tail, alg_guide_blur_strength, alg_guide_blur_sigma_threshold,
            )
            if _alg_wrapped is not None:
                sampler = _alg_wrapped
            elif not self._alg_unreachable_noted:
                self._alg_unreachable_noted = True
                print("[FunPackSceneChain] ALG is on, but the wired SAMPLER exposes no "
                      "sampler_function to wrap, so the anchor/guide blur cannot be applied "
                      "to it. Sampling continues without it.")
        # EXPERIMENTAL Bounded Attention: model-level attention hooks (sampler-agnostic, unlike
        # the toggles above which only work on Distilled Flow), so install/remove here rather
        # than via extra_options. Cheap to attempt (no-ops fast without the right metadata).
        _ba_handles = self._install_bounded_attention(model, latent, positive) if bounded_attention_enabled else []
        model = self._install_h3_token_weights(model, positive)
        # Per-modality AdaLN gain. Sampler-side and self-contained: it reads no refinement
        # key, no rating and no Studio state, so it works on a graph with the Refiner absent
        # entirely. 1.0 on all three does not even clone the model.
        model = self._install_h3_adaln_gains(model, positive)
        model = self._install_h3_final_layer(model, positive)
        model = self._install_h3_token_refiner(model, positive)

        try:
            sampled = comfy.sample.sample_custom(
                model, noise, float(cfg), sampler, sigmas, positive, negative, samples,
                noise_mask=latent.get("noise_mask"), seed=int(seed),
                callback=_progress_cb if pbar is not None else None,
            )
        finally:
            self._remove_bounded_attention(_ba_handles)
        self._assert_finite_sample(sampled)
        latent["samples"] = sampled
        latent.pop("noise_mask", None)
        return latent

    # ---------------------------------------------------------------------------
    # cut_opening_frames — let the i2v anchor do its work, then cut it out of the clip.
    #
    # The anchor is a real, PINNED latent frame at temporal position 0: it constrains
    # identity/style/composition all the way down the schedule, and it is also literally
    # the first frame you see. The scene is sampled exactly as normal — the anchor is
    # never weakened, blurred or approximated — and the opening is then cut off the
    # FINISHED latent, so it can be the former without being the latter. An i2v
    # generation that reads as t2v, for no extra sampling at all.
    #
    # Nothing is regrown to replace what is cut: the scene simply comes out that much
    # shorter. Regrowing it was tried (slide the frames left, refill the freed tail,
    # finish the schedule on the result) and the invented ending consistently came out
    # with worse or missing movement, which is not worth paying steps for.
    # ---------------------------------------------------------------------------

    def _anchor_pinned_frames(self, chunk):
        """How many leading latent frames this chunk pins (mask < 1) — i.e. the i2v anchor.

        0 means a genuine t2v scene with no anchor image: there is nothing to shift, and
        sliding anyway would just discard real generated frames. Reading the CHUNK's own
        mask rather than the template's is deliberate — it reflects what this scene actually
        got, including per-scene anchors.
        """
        try:
            tensors = self._latent_tensors(chunk)
            masks = self._latent_masks(chunk, len(tensors))
            if not masks or masks[0] is None:
                return 0
            return self._protected_prefix_frames(masks[0], self._tensor_frames(tensors[0]))
        except Exception:
            return 0

    @staticmethod
    def _is_companion_conditioning(entry):
        """True for an entry that rides WITH a scene rather than being one.

        Studio tags every entry past the first of a wired multi-entry CONDITIONING: an r2v
        node emits the reference block and the encoded prompt separately, and both describe
        one generation. Untagged, each would be counted as its own scene.
        """
        try:
            return bool(entry[1].get("funpack_companion_conditioning"))
        except (AttributeError, IndexError, KeyError, TypeError):
            return False

    def _second_pass_schedule(self, alt_sigmas):
        """Validate the pass-2 schedule. Returns (sigmas, reason); one of them is None.

        There is nothing to derive here and nothing to cut. Pass 1 runs the main schedule in
        full, exactly as written; pass 2 then runs THIS schedule in full, exactly as written.
        Pass 2 therefore starts from a FINISHED clip, which is the ordinary img2img re-entry
        (comfy's CONST scaling, x = s*noise + (1-s)*clean, valid precisely because the input
        is clean). To make pass 1 shorter, shorten the main schedule.

        A hand-typed schedule is the one input that can be malformed, and both ways it can
        be are silent in the OUTPUT rather than loud at runtime, so they are caught here.
        """
        if not isinstance(alt_sigmas, torch.Tensor) or alt_sigmas.numel() < 2:
            return None, ("no second-pass schedule — set one (comma-separated sigmas, high to "
                          "low, ending at 0) or turn the second pass off")
        vals = [float(v) for v in alt_sigmas.tolist()]
        eps = 1e-4
        if any(b > a + eps for a, b in zip(vals, vals[1:])):
            return None, ("second-pass schedule must descend — "
                          f"{', '.join(f'{v:g}' for v in vals)} goes back up, which walks the "
                          f"trajectory backwards. List the sigmas high to low")
        if vals[-1] > eps:
            return None, (f"second-pass schedule ends at {vals[-1]:g}, not 0 — the scene would "
                          f"come out partially denoised (noise artefacts). End it at 0")
        if vals[0] <= eps:
            return None, "second-pass schedule starts at 0 — there is nothing for it to denoise"
        return alt_sigmas, None

    def _second_pass_operate(self, latent, op, upsampler, vae, scale=2.0):
        """Apply the chosen latent-space operation between the two passes.

        All of these are OPTIONS — "none" is the default and the whole feature is opt-in.
        Everything here touches the VIDEO stream only: the audio stream sits alongside it in
        the same nested latent and has no spatial axes to scale, and reshaping it is exactly
        the class of change that has corrupted audio before.

        - "sharpen": Lightricks' trained 2x LatentUpsampler forward, then an antialiased
          bicubic downscale back to the original size. No video-model calls at all, so it
          costs a fraction of one step. It adds detail consistent with what is already there; it cannot fix structure
          that is wrong (the same limit segmented detailing's 'sharpen' mode documents).
          Pass 2 then re-denoises the sharpened latent, which is what makes it stick.
        - "upscale_2x": the same upsampler, but the result is KEPT at 2x. Pass 2 therefore
          runs at four times the pixels, and the scene decodes at double resolution. This is
          the expensive one by a wide margin — it is here because it was asked for, not
          because it is a good default.

        Returns (latent, note). The latent is returned unchanged with a note when the op
        cannot run, so a missing upsampler degrades to "second pass, no operation" rather
        than failing the whole render.
        """
        if op in (None, "none", ""):
            return latent, None
        # Validate the name BEFORE touching the upsampler: an unrecognised op should cost
        # nothing and say so, not pay for a full upsampler forward and then be discarded.
        if op not in ("sharpen", "upscale_2x"):
            return latent, f"second_pass_op={op} skipped: unknown operation"
        if upsampler is None:
            return latent, f"second_pass_op={op} skipped: no latent upsampler could be loaded"
        try:
            from . import detailing as _d
        except ImportError:  # loaded as a top-level module (tests, direct import)
            import detailing as _d
        result = self._clone_latent(latent)
        tensors = self._latent_tensors(result)
        video = tensors[0]
        h, w = int(video.shape[-2]), int(video.shape[-1])
        # An upsampler trained on a different model's latents fails inside a convolution,
        # minutes in, with a shape error that names neither model. Say it here instead.
        want = _d.upsampler_in_channels(upsampler)
        have = int(video.shape[1]) if video.ndim >= 5 else None
        if want and have and want != have:
            return latent, (f"second_pass_op={op} skipped: the selected latent upsampler takes "
                            f"{want}-channel latents and this model's are {have}-channel — "
                            f"install an upsampler trained for this model")
        # Lightricks' upsampler is a fixed 2x network — its final PixelShuffle decides that,
        # not a parameter — so a factor it cannot honour is reported rather than silently
        # rounded to 2x. H3's resizer interpolates to a requested size and takes any factor.
        scale, fixed = float(scale), ""
        if abs(scale - 2.0) > 1e-6 and not _d.upsampler_takes_a_scale(upsampler):
            fixed = f", {scale:g}x ignored: this upsampler is fixed at 2x"
            scale = 2.0
        try:
            up = _d._run_upsampler(upsampler, video, vae, scale=scale)
        except Exception as exc:  # noqa: BLE001
            return latent, f"second_pass_op={op} skipped: upsampler failed ({exc})"
        if op == "sharpen":
            tensors[0] = _d._downscale_to(up, h, w)
            note = (f"second_pass_op(sharpen: {scale:g}x upsampler pass, resampled back to "
                    f"{h}x{w}{fixed})")
        else:  # "upscale_2x" — the value is historical; the factor is the knob
            tensors[0] = up
            got_h, got_w = int(up.shape[-2]), int(up.shape[-1])
            note = (f"second_pass_op(upscale: {h}x{w} -> {got_h}x{got_w} — pass 2 runs at "
                    f"{(got_h * got_w) / max(1, h * w):.2g}x the pixels{fixed})")
        if self._is_nested(result.get("samples")):
            result["samples"] = comfy.nested_tensor.NestedTensor(tensors)
        else:
            result["samples"] = tensors[0]
        return result, note

    def _latent_spatial_changed(self, state, chunk):
        """True when `state`'s video stream no longer has `chunk`'s H/W — i.e. a
        second_pass_op resized it. Everything recorded against the old grid (the i2v
        pin, guide keyframe token indices) is invalid past this point. Unreadable
        shapes answer False so a probe failure never drops a guide on its own."""
        try:
            return (self._latent_tensors(state)[0].shape[-2:]
                    != self._latent_tensors(chunk)[0].shape[-2:])
        except Exception:
            return False

    def _rescale_mask_to(self, mask, height, width):
        """A noise mask on `height`x`width`, or None if it cannot be built.

        Nearest-neighbour, and deliberately: this mask is a hard 0/1 pin, and interpolating
        its edge would hand pass 2 half-pinned latent positions that are neither frozen nor
        free. An i2v pin covers whole leading frames, so nearest is exact for it.

        Nested masks carry one entry per stream and only the video stream has spatial axes —
        audio is passed through untouched, as everywhere else.
        """
        if mask is None:
            return None
        try:
            if self._is_nested(mask):
                parts = list(mask.unbind())
                out = [self._rescale_mask_to(parts[0], height, width)] + [
                    self._clone_value(m) for m in parts[1:]]
                if out[0] is None:
                    return None
                return comfy.nested_tensor.NestedTensor(out)
            if not isinstance(mask, torch.Tensor) or mask.dim() < 4:
                return None
            if mask.shape[-2:] == (int(height), int(width)):
                return self._clone_value(mask)
            flat = mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1]).float()
            resized = torch.nn.functional.interpolate(
                flat, size=(int(height), int(width)), mode="nearest")
            return resized.reshape(*mask.shape[:-2], int(height), int(width)).to(mask.dtype)
        except Exception:
            return None

    def _restore_pinned_prefix(self, state, chunk):
        """Put the i2v pin back on a latent that is about to be handed to a second pass.

        _sample_chunk drops noise_mask from what it returns, so without this a second pass
        would run UNPINNED and re-denoise the anchor frame, letting the reference drift. The
        pinned region's values are copied back from the original chunk as well, since pass 2
        re-noises everything it is handed, including frames that are supposed to be frozen.

        When a resolution-changing second_pass_op (upscale_2x) has run, the chunk's anchor and
        mask are the old size. The anchor is NOT re-derived from the source image: the pinned
        frames already in `state` are the upsampler's own rendering of those very frames, so
        they are kept as they are and only the mask is scaled to the new grid. That is the
        same upscale every other frame in the clip just got, which is what makes it the anchor
        and not an invention — but it does mean the pin at 2x holds the upsampled anchor
        rather than the encoded source image.
        """
        mask = chunk.get("noise_mask")
        pinned = self._anchor_pinned_frames(chunk)
        if mask is None or pinned <= 0:
            return state
        try:
            upscaled = self._latent_spatial_changed(state, chunk)
        except Exception as _e:
            _log.failed("FunPackSceneChain", "anchor pin restore", _e,
                        "the scene keeps its UNPINNED latent — the anchor frame is not held")
            return state
        result = self._clone_latent(state)
        if upscaled:
            try:
                h, w = self._latent_tensors(result)[0].shape[-2:]
            except Exception as _e:
                _log.failed("FunPackSceneChain", "anchor pin rescale", _e,
                            "the scene keeps its UNPINNED latent — the anchor frame is not held")
                return state
            scaled = self._rescale_mask_to(mask, int(h), int(w))
            if scaled is None:
                return state
            result["noise_mask"] = scaled
            return result
        try:
            src = self._latent_tensors(chunk)[0]
            dst = self._latent_tensors(result)
            if self._tensor_frames(dst[0]) >= pinned and self._tensor_frames(src) >= pinned:
                dst[0][:, :, :pinned] = src[:, :, :pinned].to(
                    device=dst[0].device, dtype=dst[0].dtype)
                if self._is_nested(result.get("samples")):
                    result["samples"] = comfy.nested_tensor.NestedTensor(dst)
                else:
                    result["samples"] = dst[0]
        except Exception as _e:
            _log.failed("FunPackSceneChain", "anchor pin copy", _e,
                        "the scene keeps its UNPINNED latent — the anchor frame is not held")
            return state
        result["noise_mask"] = self._clone_value(mask)
        return result

    def _crop_stream_head_to(self, tensor, stream, kept, total):
        """Crop a non-video stream's HEAD so it keeps `kept/total` of its own time.

        Each extra stream (audio) has its own rate and its own time AXIS: LTXAV puts time on
        dim 2 for every stream, MiniMax H3 puts the stereo channel there and time last. The
        kept length is derived from the VIDEO proportion rather than from a separately rounded
        drop, because audio timing comes from the audio stream's own index — cropping the two
        by different amounts of TIME desyncs the clip silently, with no error.
        """
        if not isinstance(tensor, torch.Tensor):
            return tensor
        dim = self._stream_dim(stream)
        n = int(tensor.shape[dim])
        keep = max(1, min(n, int(round(n * max(0, kept) / max(1, total)))))
        if keep >= n:
            return tensor
        idx = [slice(None)] * tensor.dim()
        idx[dim] = slice(n - keep, n)
        return tensor[tuple(idx)]

    def _scene_pixel_start(self, latent_start, time_scale):
        """First DECODED pixel index of a scene that begins at `latent_start` in the chain.

        The causal VAE decodes f latent frames to (f-1)*scale+1 pixels: latent frame 0 is
        the temporal origin and covers ONE pixel, every later latent frame covers `scale`.
        So latent frame i starts at pixel (i-1)*scale+1, and only frame 0 starts at 0.
        """
        i = max(0, int(latent_start))
        if i == 0:
            return 0
        return (i - 1) * max(1, int(time_scale)) + 1

    def _cut_opening_pixel_spans(self, images, spans):
        """Remove `spans` — (start, count) pixel ranges — from a decoded image batch.

        Cutting the OPENING has to happen here, on pixels, and never on the latent. The LTX
        VAE is causal: latent frame 0 is the temporal origin, and every later latent frame
        was generated as a CONTINUATION of the frames before it. Slice the front off the
        latent and the survivor promoted to position 0 gets decoded with the origin handling
        it was never generated for, which is what produced the noisy opening frame. Decoding
        the whole latent first and dropping pixels afterwards leaves the origin intact, so
        every surviving frame decodes in the context it was sampled in.

        It is also exact. A latent cut could only remove whole latent frames, and because
        the promoted frame changes from covering 1 pixel to covering `scale`, dropping k
        latent frames actually shortened the clip by k*scale pixels rather than the
        (k-1)*scale+1 it spanned — so the count never matched what was asked for either.
        Here N means N, the way it already does on H3.

        Later spans are removed first so earlier removals cannot shift their indices.
        """
        if not isinstance(images, torch.Tensor) or not spans:
            return images, 0
        total = int(images.shape[0])
        keep = torch.ones(total, dtype=torch.bool)
        for start, count in sorted(spans, key=lambda s: int(s[0]), reverse=True):
            start = max(0, min(int(start), total))
            end = max(start, min(start + max(0, int(count)), total))
            keep[start:end] = False
        if bool(keep.all()):
            return images, 0
        if not bool(keep.any()):
            # Never hand back an empty batch: a cut that would remove the whole clip is a
            # misconfiguration, and returning nothing turns it into an obscure downstream
            # crash instead of a visibly-too-short video.
            keep[-1] = True
        return images[keep], total - int(keep.sum())

    def _remove_latent_time_spans(self, latent, pixel_spans, pixel_total):
        """Remove the same stretches of TIME from every non-video stream of `latent`.

        `pixel_spans` are indices into the decoded VIDEO. Audio runs at its own rate on its
        own axis (LTXAV puts time on dim 2; MiniMax H3 puts it last), so each span is mapped
        by PROPORTION of the clip rather than by a frame count — cropping the two streams by
        different amounts of time desynchronises them with no error, just sound that drifts.

        The video stream is deliberately untouched: it is the thing that must keep its
        temporal origin, and the caller takes video from the decoded IMAGES instead.
        """
        if not pixel_spans or int(pixel_total) <= 0:
            return latent
        result = self._clone_latent(latent)
        tensors = self._latent_tensors(result)
        changed = False
        for idx in range(1, len(tensors)):
            tensor = tensors[idx]
            axis = self._time_dims[idx] if idx < len(self._time_dims) else 2
            try:
                length = int(tensor.shape[axis])
            except Exception:
                continue
            if length <= 1:
                continue
            keep = torch.ones(length, dtype=torch.bool)
            for start, count in pixel_spans:
                a = int(round(int(start) / float(pixel_total) * length))
                b = int(round((int(start) + int(count)) / float(pixel_total) * length))
                a = max(0, min(a, length))
                b = max(a, min(b, length))
                keep[a:b] = False
            if bool(keep.all()):
                continue
            if not bool(keep.any()):
                keep[-1] = True
            tensors[idx] = tensor.index_select(axis, torch.nonzero(keep, as_tuple=True)[0].to(tensor.device))
            changed = True
        if not changed:
            return latent
        if self._is_nested(result.get("samples")):
            result["samples"] = comfy.nested_tensor.NestedTensor(tensors)
        else:
            result["samples"] = tensors[0]
        return result

    def _cut_opening_pixels(self, images, latent, pixel_frames):
        """MiniMax H3's cut: drop the opening off the DECODED frames, not off the latent.

        Same user intent as the LTX path above — let the anchor condition the shot at full
        strength, then remove the frames that are still visibly the reference still — but H3
        cannot express it in latent space. Two reasons, both structural:

        * H3's anchor is not a pinned latent frame at all. It is a keyframe CONDITION row
          packed beside the text (see _apply_h3_anchor), never denoised and never rendered,
          so there is no pinned prefix to slice off and no mask to read a length from.
        * H3's video latent lives on a 5k+2 grid (17k+5 pixel frames). Cutting an arbitrary
          number of latent frames leaves a count the VAE has no defined decode for, where
          LTXAV decodes any f >= 1 as (f-1)*scale+1.

        Cutting the decoded IMAGE batch has neither problem and is exact to the frame: N means
        N. The AUDIO stream still has to move with it — audio decodes from this node's LATENT
        output (VAEDecodeAudio on H3), so leaving it alone would start the sound N frames
        before the picture. Its head is cropped by the same PROPORTION of time, on its own
        axis (H3 puts audio time last).

        The latent's VIDEO stream is deliberately left at full length: it is off-grid the
        moment it is cut. So the latent comes back with a full video stream beside a cropped
        audio one, and the caller says so — video must come from the IMAGES output on a cut
        H3 run.

        Returns (images, latent, dropped).
        """
        drop = int(pixel_frames)
        if not isinstance(images, torch.Tensor) or images.dim() < 1 or drop <= 0:
            return images, latent, 0
        total = int(images.shape[0])
        drop = max(0, min(drop, total - 1))
        if drop <= 0:
            return images, latent, 0
        images = images[drop:]
        kept = total - drop
        result = self._clone_latent(latent)
        tensors = self._latent_tensors(result)
        for _idx in range(1, len(tensors)):
            tensors[_idx] = self._crop_stream_head_to(tensors[_idx], _idx, kept, total)
        if self._is_nested(result.get("samples")):
            result["samples"] = comfy.nested_tensor.NestedTensor(tensors)
        else:
            result["samples"] = tensors[0]
        result.pop("noise_mask", None)
        return images, result, drop

    def _output_connected(self, prompt, unique_id, output_index):
        """Return True if the given output slot index is wired to any downstream node."""
        if not prompt or not unique_id:
            return True  # can't tell — assume connected
        uid = str(unique_id)
        for node_data in prompt.values():
            for v in node_data.get("inputs", {}).values():
                if isinstance(v, list) and len(v) == 2 and str(v[0]) == uid and v[1] == output_index:
                    return True
        return False

    def _blur_pixel_frame(self, frame, sigma):
        """Box-blur approximation on a single [H, W, C] pixel frame."""
        if sigma <= 0:
            return frame
        k = max(3, int(sigma * 3) | 1)
        pad = k // 2
        x = frame.permute(2, 0, 1).unsqueeze(0).float()
        x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
        x = torch.nn.functional.avg_pool2d(x, k, stride=1, padding=0)
        return x.squeeze(0).permute(1, 2, 0).to(dtype=frame.dtype)

    def _apply_effect_on_pixels(self, frames, effect, center, half):
        """Apply one visual transition effect in-place on a [N, H, W, C] float tensor.

        Simple V-ramp centered at the seam: full effect at center, linear ramp
        back to no-effect at ±half frames. transition_duration controls half.
        """
        n = frames.shape[0]
        center = max(half, min(n - half - 1, center))
        if effect == "fade_to_black":
            for i in range(max(0, center - half), min(n, center + half)):
                brightness = abs(i - center) / max(1, half)
                frames[i] = frames[i] * brightness
        elif effect == "crossfade":
            orig = frames.clone()
            for k in range(1, half + 1):
                alpha = k / (half + 1)
                pre_i, post_i = center - k, center + k - 1
                if 0 <= pre_i < n and 0 <= post_i < n:
                    frames[pre_i] = (1 - alpha) * orig[pre_i] + alpha * orig[post_i]
                    frames[post_i] = (1 - alpha) * orig[post_i] + alpha * orig[pre_i]
        elif effect == "blur_out_in":
            for i in range(max(0, center - half), min(n, center + half)):
                ramp = abs(i - center) / max(1, half)
                sigma = 8.0 * (1.0 - ramp)
                if sigma > 0:
                    frames[i] = self._blur_pixel_frame(frames[i], sigma)

    def _apply_transitions_pixel(self, decoded, boundary_entries, transition_duration):
        """Apply visual transitions on a full decoded [N, H, W, C] frame tensor."""
        active = [e for e in boundary_entries if e.get("effect") and e["effect"] != "none"]
        if not active:
            return decoded
        frames = decoded.clone().float()
        half = max(1, transition_duration // 2)
        for entry in active:
            self._apply_effect_on_pixels(frames, entry["effect"], int(entry["pixel_frame"]), half)
        return frames.clamp(0.0, 1.0).to(dtype=decoded.dtype)

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

    def _scene_meta(self, scene_conditioning, index):
        """Extract FunPack scene metadata for overlap / contamination diagnostics."""
        meta = {}
        if (
            isinstance(scene_conditioning, (list, tuple))
            and len(scene_conditioning) >= 2
            and isinstance(scene_conditioning[1], dict)
        ):
            meta = dict(scene_conditioning[1])
        text = str(meta.get("funpack_scene_text") or "").strip() or f"Scene {index + 1}"
        encode = str(meta.get("funpack_encode_text") or "").strip() or text
        return {
            "index": index + 1,
            "text": text,
            "encode_text": encode,
            "seed": meta.get("funpack_scene_seed"),
            "transition_effect": meta.get("funpack_transition_effect"),
            "temporal_mult": meta.get("funpack_temporal_mult"),
        }

    def _latent_to_pixel_frame(self, latent_frame, time_scale):
        latent_frame = max(0, int(latent_frame))
        time_scale = max(1, int(time_scale))
        if time_scale > 1:
            return int((latent_frame - 1) * time_scale + 1) if latent_frame > 0 else 0
        return latent_frame

    def _scene_pixel_ranges(self, scene_count, pixel_frames_per_scene, pixel_overlap):
        """Contiguous pixel spans in the stitched output (before transition post-FX)."""
        if scene_count <= 0:
            return []
        pixel_overlap = max(0, int(pixel_overlap))
        pixel_frames_per_scene = max(1, int(pixel_frames_per_scene))
        stride = max(1, pixel_frames_per_scene - pixel_overlap)
        ranges = []
        for i in range(scene_count):
            start = i * stride
            end = start + pixel_frames_per_scene - 1
            ranges.append({"scene": i + 1, "start": start, "end": end})
        return ranges

    def _boundary_contamination_zones(self, boundary_pixel, pixel_overlap, transition_duration, effect):
        """Pixel ranges where scene N+1 can visibly influence scene N (and vice versa)."""
        boundary_pixel = max(0, int(boundary_pixel))
        pixel_overlap = max(0, int(pixel_overlap))
        zones = {}
        if pixel_overlap > 0:
            zones["latent_blend"] = {
                "scene_prev_tail": [max(0, boundary_pixel - pixel_overlap), max(0, boundary_pixel - 1)],
                "scene_next_head": [boundary_pixel, boundary_pixel + pixel_overlap - 1],
                "note": "Previous scene tail is copied into the next chunk, denoised under next-scene conditioning, then slerp-blended back — motion from scene N+1 can appear in scene N's last overlap frames.",
            }
        if effect and effect != "none" and transition_duration > 0:
            half = max(1, int(transition_duration) // 2)
            lo = max(0, boundary_pixel - half)
            hi = boundary_pixel + half - 1
            zones["transition_effect"] = {
                "effect": effect,
                "pixel_range": [lo, hi],
                "note": "Post-decode pixel effect mixes frames across the seam — scene N+1 content can show in scene N's tail (crossfade is strongest).",
            }
        return zones

    def _build_overlap_diagnostics(
        self,
        *,
        scene_count,
        video_frames,
        num_frames_per_scene,
        pixel_overlap,
        latent_overlap,
        time_scale,
        transition_duration,
        boundaries,
        scene_runs,
        carry_i2v_guides,
        mid_scene_guide,
        embed_guidance,
        embed_guidance_strength,
        embed_guidance_source,
    ):
        """JSON report: where scene conditioning/latent/pixel domains overlap."""
        pixel_ranges = self._scene_pixel_ranges(scene_count, num_frames_per_scene, pixel_overlap)
        boundary_reports = []
        for entry in boundaries or []:
            pixel = int(entry.get("pixel_frame") or 0)
            effect = entry.get("effect")
            between = entry.get("between") or []
            zones = self._boundary_contamination_zones(pixel, pixel_overlap, transition_duration, effect)
            boundary_reports.append({
                "between_scenes": between,
                "pixel_frame": pixel,
                "latent_frame": entry.get("latent_frame") or entry.get("boundary_latent"),
                "effect": effect,
                "contamination_zones": zones,
            })

        global_steering = []
        if embed_guidance:
            global_steering.append({
                "mechanism": "embed_guidance",
                "strength": float(embed_guidance_strength),
                "source": str(embed_guidance_source or "relative"),
                "scope": "all_scenes_every_denoise_step",
                "note": "Steers each scene's conditioning toward the learned liked direction during sampling. "
                        "Absolute mode uses a prompt-agnostic global taste — action from ANY liked scene can bias scene 1.",
            })
        if carry_i2v_guides:
            global_steering.append({
                "mechanism": "carry_i2v_guides",
                "scope": "scene_2_onward",
                "note": "Scene 1 template frames are prepended as hidden guide tokens in continuation chunks only — "
                        "does not inject scene 2 into scene 1.",
            })
        if mid_scene_guide:
            global_steering.append({
                "mechanism": "mid_scene_guide",
                "scope": "scene_2_onward",
                "note": "Middle frame of scene N guides scene N+1 denoising — does not retroactively change scene N.",
            })
        global_steering.append({
            "mechanism": "studio_absolute_steer",
            "scope": "all_scenes_pre_sample",
            "note": "When Studio steer_mode is absolute/both, the same global taste pull is applied to every "
                    "scene conditioning entry before the chain sampler runs — can add scene-2-like motion cues to scene 1.",
        })

        scenes_out = []
        for run in scene_runs:
            idx = int(run.get("index") or 0)
            pr = next((r for r in pixel_ranges if r["scene"] == idx), None)
            item = dict(run)
            if pr:
                item["pixel_range"] = [pr["start"], pr["end"]]
                tail_zones = []
                for br in boundary_reports:
                    between = br.get("between_scenes") or []
                    if len(between) == 2 and between[0] == idx:
                        for zone in (br.get("contamination_zones") or {}).values():
                            if isinstance(zone, dict) and "scene_prev_tail" in zone:
                                tail_zones.append(zone["scene_prev_tail"])
                if tail_zones:
                    item["tail_contamination_pixels"] = tail_zones
                if idx == 1 and embed_guidance:
                    item["whole_scene_steering"] = True
            scenes_out.append(item)

        return {
            "scene_count": scene_count,
            "frames_per_scene": int(video_frames),
            "frames_per_scene_pixel": int(num_frames_per_scene),
            "num_frames_per_scene": int(num_frames_per_scene),
            "pixel_overlap": int(pixel_overlap),
            "latent_overlap": int(latent_overlap),
            "time_scale": int(time_scale),
            "scene_pixel_ranges": pixel_ranges,
            "boundaries": boundary_reports,
            "global_steering": global_steering,
            "scenes": scenes_out,
        }

    def _scene_seed(self, scene_conditioning):
        if (
            isinstance(scene_conditioning, (list, tuple))
            and len(scene_conditioning) >= 2
            and isinstance(scene_conditioning[1], dict)
        ):
            try:
                return int(scene_conditioning[1].get("funpack_scene_seed"))
            except (TypeError, ValueError):
                return None
        return None

    def _scene_transition_effect(self, scene_conditioning):
        if (
            isinstance(scene_conditioning, (list, tuple))
            and len(scene_conditioning) >= 2
            and isinstance(scene_conditioning[1], dict)
        ):
            effect = str(scene_conditioning[1].get("funpack_transition_effect") or "none")
            return None if effect == "none" else effect
        return None

    def _scene_temporal_mult(self, scene_conditioning):
        """Per-scene frame_rate multiplier baked in by Studio's "auto" temporal director.
        Returns a float (1.0 = no change) or None when the scene carries no temporal intent."""
        if (
            isinstance(scene_conditioning, (list, tuple))
            and len(scene_conditioning) >= 2
            and isinstance(scene_conditioning[1], dict)
        ):
            try:
                return float(scene_conditioning[1].get("funpack_temporal_mult"))
            except (TypeError, ValueError):
                return None
        return None

    def _scene_temporal_loop(self, scene_conditioning):
        """Per-scene loop intent baked in by Studio's "auto" temporal director
        (funpack_temporal_loop). True → install the Mobius latent-roll wrapper."""
        if (
            isinstance(scene_conditioning, (list, tuple))
            and len(scene_conditioning) >= 2
            and isinstance(scene_conditioning[1], dict)
        ):
            return bool(scene_conditioning[1].get("funpack_temporal_loop"))
        return False

    def _scene_temporal_mode(self, scene_conditioning):
        """Per-scene temporal mode tag (e.g. pulse). None when unset."""
        if (
            isinstance(scene_conditioning, (list, tuple))
            and len(scene_conditioning) >= 2
            and isinstance(scene_conditioning[1], dict)
        ):
            mode = str(scene_conditioning[1].get("funpack_temporal_mode") or "").strip().lower()
            return mode or None
        return None

    def _decode_last_frame(self, latent, vae):
        try:
            tensors = self._latent_tensors(latent)
            if not tensors:
                return None
            n = self._tensor_frames(tensors[0])
            if n < 1:
                return None
            last = self._time_slice(tensors[0], n - 1, None)
            decoded = vae.decode(last)
            if decoded is None:
                return None
            if decoded.dim() == 5:
                decoded = decoded[:, 0]
            elif decoded.dim() == 3:
                decoded = decoded.unsqueeze(0)
            return decoded.clamp(0.0, 1.0)
        except Exception:
            return None

    def _absolute_key(self):
        """The keyless global-taste key for Absolute embed_guidance (mirrors the Refiner)."""
        try:
            from .conditioning import FUNPACK_ABSOLUTE_KEY
        except ImportError:
            from conditioning import FUNPACK_ABSOLUTE_KEY
        return FUNPACK_ABSOLUTE_KEY

    def _protect_audio(self, steered, original):
        """Confine a conditioning edit to LTXAV's video channel-slice; audio keeps the original
        conditioning. No-op for single-stream LTXV. Mirrors the Refiner's protect_audio_channels."""
        try:
            try:
                from .conditioning import protect_audio_channels
            except ImportError:
                from conditioning import protect_audio_channels
            return protect_audio_channels(steered, original)
        except Exception as _e:
            _log.failed("FunPackSceneChain", "audio channel protection", _e,
                        "conditioning steering reached the AUDIO channels too")
            return steered

    def _load_value_function(self, refinement_key):
        """Load the online value function if trained and ready."""
        try:
            try:
                from .value_function import OnlineValueFunction
                from .conditioning import refinement_state_path
            except ImportError:
                from value_function import OnlineValueFunction
                from conditioning import refinement_state_path
            import os as _os
            path = refinement_state_path(refinement_key, "value_fn", prefix="refine_v2", extension="pt")
            if not _os.path.exists(path):
                return None
            with torch.inference_mode(False):
                vf = OnlineValueFunction.load(path)
            return vf if vf.is_ready() else None
        except Exception:
            return None

    def _load_output_value_function(self, refinement_key):
        """Load the output-space (x0_hat) value function if trained and ready. Sibling of
        _load_value_function, trained on the sampler's own predicted output instead of the
        input conditioning — see value_function.LatentValueFunction."""
        try:
            try:
                from .value_function import LatentValueFunction
                from .conditioning import refinement_state_path
            except ImportError:
                from value_function import LatentValueFunction
                from conditioning import refinement_state_path
            import os as _os
            path = refinement_state_path(refinement_key, "value_fn_x0", prefix="refine_v2", extension="pt")
            if not _os.path.exists(path):
                return None
            with torch.inference_mode(False):
                vf = LatentValueFunction.load(path)
            return vf if vf.is_ready() else None
        except Exception:
            return None

    def _save_output_value_snapshot(self, refinement_key, denoised, video_mask):
        """End-of-run: pool the final x0_hat (video-only, audio excluded — same convention as
        embed_guidance/velocity-bias) down to a small vector and persist it so the NEXT rating
        cycle can pair it with a reward and train the output-space value function. The raw
        latent is never persisted (too large; N varies per run) — only this compressed vector,
        mirroring how the Refiner already persists conditioning into last_run for the
        equivalent conditioning-space training path."""
        try:
            try:
                from .value_function import LatentValueFunction, compress_packed_latent
                from .conditioning import refinement_state_path
            except ImportError:
                from value_function import LatentValueFunction, compress_packed_latent
                from conditioning import refinement_state_path
            import os as _os
            target = denoised if video_mask is None else denoised * video_mask
            with torch.inference_mode(False), torch.no_grad():
                compressed = compress_packed_latent(
                    target.detach().float(), LatentValueFunction.DEFAULT_HIDDEN_DIM
                ).cpu()
            path = refinement_state_path(refinement_key, "x0_snapshot", prefix="refine_v2", extension="pt")
            _os.makedirs(_os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            torch.save(compressed, tmp)
            _os.replace(tmp, path)  # atomic — a rating read never sees a partial write
        except Exception as e:
            print(f"[FunPackSceneChain] output value snapshot save failed: {e}")

    def _build_output_guidance_wrapper(self, model, value_fn, strength, ramp_fn=None):
        """Sibling of _build_embed_guidance_wrapper, but corrects the model's OUTPUT
        (x0_hat) instead of nudging the INPUT conditioning. value_fn.gradient() backprops
        through its own compress() step, so feeding it the denoised prediction returns a
        same-shape gradient — a direct, single-pass correction (no extra forward pass
        through the base model; the only added cost is one backward pass through the value
        function's few-hundred-parameter MLP). Borrows NoiseTilt's core idea (train the
        reward signal on the model's own prediction, not the input) without its SDE
        noise-term mechanism, which has no analogue in this deterministic-ODE sampler —
        see [[research_2026_training_free_candidates]].

        Two invariants the correction depends on:
        - VIDEO SLICE, not post-hoc mask: the gradient is computed on the packed latent's
          video span only, because that is the domain the value function was trained on
          (_save_output_value_snapshot pools the video stream alone). Feeding the full
          packed AV tensor would shift every adaptive-pooling bucket and let audio
          contaminate the score.
        - NORM-CALIBRATED step: the raw MLP gradient reaches each latent element through
          an adaptive-pool window of ~N/512 elements, so its per-element magnitude is
          O(1/window) — numerically inert if applied raw. The delta is rescaled so
          strength means "fraction of the video stream's norm per fully-ramped step"
          (0.02 = 2%), the same relative-calibration convention as the slider's 0.15 and
          the Refiner's NORM_SCALE."""
        old_wrapper = model.model_options.get("model_function_wrapper")
        _ramp = ramp_fn or (lambda sigma: max(0.0, 1.0 - float(sigma) * 2.0))

        def _call(apply_fn, a):
            if old_wrapper is not None:
                return old_wrapper(apply_fn, a)
            return apply_fn(a["input"], a["timestep"], **a.get("c", {}))

        def _output_wrapper(apply_fn, args, _vf=value_fn, _s=strength):
            denoised = _call(apply_fn, args)
            ts = args.get("timestep")
            try:
                sigma = float(ts.max().item()) if ts is not None else 1.0
            except Exception:
                sigma = 1.0
            scale = _ramp(sigma)  # same late-step gate as embed_guidance
            if scale <= 0:
                return denoised
            try:
                span = _video_span(model, denoised)
                if span is not None:
                    off, sz, _shape = span
                    target = denoised[..., off:off + sz]
                else:
                    target = denoised  # single-stream: full tensor IS the training domain
                grad = _vf.gradient(target)
                gn = float(grad.float().norm())
                if gn <= 0.0:
                    return denoised
                k = (_s * scale) * float(target.float().norm()) / gn
                corrected = target + grad * k
                if span is None:
                    return corrected
                return torch.cat(
                    [denoised[..., :off], corrected, denoised[..., off + sz:]], dim=-1
                )
            except Exception as e:
                print(f"[FunPackSceneChain] output_guidance: gradient failed ({e}), passing through")
                return denoised

        model.model_options["model_function_wrapper"] = _tag_scene_wrapper(_output_wrapper, old_wrapper)
        return old_wrapper

    def _build_dynashift_wrapper(self, model, negatives, strength, threshold, raw_cond=None, ramp_fn=None):
        """DynaShift: steer the predicted x0 AWAY from the key's negative latent memory.

        A negative prompt at CFG=1: the repulsor is not a second text pass but the
        stored video latents of generations the user rated as containing something
        unwanted (see negative_memory.py — awful / wrong-appearance ratings feed the
        bank). Per late step:
          1. each current video frame is pooled to a fixed descriptor and compared
             (cosine) against every stored negative frame — alignment-free in time,
             so scene position, chain length and guide tails don't matter; only the
             spatial latent geometry (C, H, W) must match (others are skipped);
          2. each negative is weighted by prompt similarity (stored cond vs current
             c_crossattn mean), so bad gens from unrelated prompts steer less;
          3. frames matching above `threshold` get the matched negative frame's
             component subtracted (positive projection removal, coefficient clamped
             at 0 so anti-aligned frames are never pushed TOWARD the negative),
             scaled by how far above threshold the match is — steering fades to a
             no-op as soon as the unwanted feature is gone ("until it's gone").

        Cost: one pooled similarity matrix + one masked subtraction per step; no
        extra model forward pass, no uncond pass. Audio byte-identical by
        construction (video span only, cat back). Frame-level (not sub-frame) v1:
        the "mask" is temporal + magnitude gating; unmatched frames are untouched."""
        old_wrapper = model.model_options.get("model_function_wrapper")
        _ramp = ramp_fn or (lambda sigma: max(0.0, 1.0 - float(sigma) * 2.0))
        _desc_dim = 512
        _prep = {}  # device -> (desc [Tall,512] fp32, units [Tall,D] fp16, owner [Tall], conds)
        _warned = [False]
        # The bank stores the RAW scene conditioning (negative_memory.save_pending is handed
        # positive[0][0]), so the prompt-similarity weight has to be computed against the raw
        # cond too. On H3 `c_crossattn` is the refined DiT hidden state instead, a different
        # width entirely, and the numel guard below would silently weight every negative 1.0.
        _raw_pooled = None
        if isinstance(raw_cond, torch.Tensor):
            _p = raw_cond.detach().float()
            while _p.dim() > 1:
                _p = _p.mean(dim=0)
            _raw_pooled = _p

        def _call(apply_fn, a):
            if old_wrapper is not None:
                return old_wrapper(apply_fn, a)
            return apply_fn(a["input"], a["timestep"], **a.get("c", {}))

        def _frame_desc(frames_f32):
            d = torch.nn.functional.adaptive_avg_pool1d(frames_f32.unsqueeze(1), _desc_dim).squeeze(1)
            return torch.nn.functional.normalize(d, dim=-1)

        def _prepare(device, c, h, w):
            key = (str(device), c, h, w)
            if key in _prep:
                return _prep[key]
            descs, units, owners, conds = [], [], [], []
            skipped = 0
            for i, entry in enumerate(negatives):
                lat = entry.get("latent")
                if not isinstance(lat, torch.Tensor) or lat.dim() != 4 or \
                        lat.shape[0] != c or lat.shape[2] != h or lat.shape[3] != w:
                    skipped += 1
                    continue
                frames = lat.permute(1, 0, 2, 3).reshape(lat.shape[1], -1).to(device)
                f32 = frames.float()
                descs.append(_frame_desc(f32))
                units.append(torch.nn.functional.normalize(f32, dim=-1).to(torch.float16))
                owners.extend([len(conds)] * frames.shape[0])
                conds.append(entry.get("cond"))
            if skipped and not _warned[0]:
                _warned[0] = True
                print(f"[FunPackSceneChain] dynashift: {skipped} negative(s) skipped "
                      "(different latent resolution than this run)")
            prepared = None
            if descs:
                prepared = (torch.cat(descs, dim=0), torch.cat(units, dim=0),
                            torch.tensor(owners, device=device), conds)
            _prep[key] = prepared
            return prepared

        def _cond_weights(conds, c_dict, device):
            cur = _raw_pooled if _raw_pooled is not None else (c_dict or {}).get("c_crossattn")
            if cur is None or not any(isinstance(cv, torch.Tensor) for cv in conds):
                return torch.ones(len(conds), device=device)
            cm = cur.detach().float().to(device)
            while cm.dim() > 1:
                cm = cm.mean(dim=0)  # -> [D]
            weights = []
            for cv in conds:
                if isinstance(cv, torch.Tensor) and cv.numel() == cm.numel():
                    sim = torch.nn.functional.cosine_similarity(
                        cm, cv.to(device).float(), dim=0)
                    weights.append(sim.clamp(0.0, 1.0))
                else:
                    weights.append(torch.ones((), device=device))
            return torch.stack(weights)

        def _dynashift_wrapper(apply_fn, args, _s=float(strength), _thr=float(threshold)):
            denoised = _call(apply_fn, args)
            ts = args.get("timestep")
            try:
                sigma = float(ts.max().item()) if ts is not None else 1.0
            except Exception:
                sigma = 1.0
            ramp = _ramp(sigma)  # same late-step gate as the other wrappers
            if ramp <= 0.0 or _s <= 0.0:
                return denoised
            try:
                span = _video_span(model, denoised)
                if span is None:
                    return denoised
                off, sz, shape = span
                c, t, h, w = int(shape[-4]), int(shape[-3]), int(shape[-2]), int(shape[-1])
                prepared = _prepare(denoised.device, c, h, w)
                if prepared is None:
                    return denoised
                desc, units, owner, conds = prepared
                cur = denoised[..., off:off + sz].reshape(c, t, h, w)
                cur_f = cur.permute(1, 0, 2, 3).reshape(t, -1).float()
                sims = _frame_desc(cur_f) @ desc.T                     # [T, Tall]
                sims = sims * _cond_weights(conds, args.get("c"), denoised.device)[owner]
                best, idx = sims.max(dim=1)                            # per current frame
                gate = ((best - _thr) / max(1e-6, 1.0 - _thr)).clamp(0.0, 1.0) * (_s * ramp)
                if float(gate.max()) <= 0.0:
                    return denoised
                matched = units[idx].float()                           # [T, D] unit rows
                coef = (cur_f * matched).sum(dim=-1).clamp(min=0.0)    # aligned component only
                new_f = cur_f - (gate * coef).unsqueeze(1) * matched
                new_span = new_f.reshape(t, c, h, w).permute(1, 0, 2, 3).reshape(1, 1, sz)
                return torch.cat([denoised[..., :off],
                                  new_span.to(denoised.dtype),
                                  denoised[..., off + sz:]], dim=-1)
            except Exception as e:
                print(f"[FunPackSceneChain] dynashift failed ({e}), passing through")
                return denoised

        model.model_options["model_function_wrapper"] = _tag_scene_wrapper(_dynashift_wrapper, old_wrapper)
        return old_wrapper

    def _load_taste_direction(self, refinement_key, slot="liked_dir"):
        """Read a learned conditioning direction from the Refiner's global taste store.
        `slot` is "liked_dir" (accumulated from liked gens) or "bad_dir" (accumulated
        from awful / strongly-negative gens). Returns a [dim] tensor, or None when the
        slot lacks the 3+ rated samples that make a direction meaningful."""
        try:
            try:
                from .conditioning import refinement_state_path, serializable_to_tensor
            except ImportError:
                from conditioning import refinement_state_path, serializable_to_tensor
            import json as _json
            path = refinement_state_path(refinement_key, "clip", prefix="refine_v2")
            with open(path, "r", encoding="utf-8") as f:
                state = _json.load(f)
            global_state = state.get("global", state)  # directions live under state["global"]
            dir_slot = global_state.get(slot, {})
            if int(dir_slot.get("direction_count", 0)) < 3:
                return None
            raw = dir_slot.get("direction")
            if raw is None:
                return None
            return serializable_to_tensor(raw)
        except Exception:
            return None

    def _load_liked_direction(self, refinement_key):
        """Read the liked conditioning direction from the Refiner's state file."""
        return self._load_taste_direction(refinement_key, "liked_dir")

    def _load_prompt_dir_index(self, refinement_key):
        """Read the (prompt fingerprint -> liked direction) retrieval index from the taste
        store — the per-prompt sibling of the single global liked_dir. Returns a list of
        (prompt [dim] fp32, direction [dim] fp32, magnitude) tuples, or [] when the key has
        no index yet. Written by conditioning._v2_store_prompt_keyed_direction on every
        liked rating; lives in the same refine_v2 clip-state file as liked_dir."""
        try:
            try:
                from .conditioning import refinement_state_path, serializable_to_tensor
            except ImportError:
                from conditioning import refinement_state_path, serializable_to_tensor
            import json as _json
            path = refinement_state_path(refinement_key, "clip", prefix="refine_v2")
            with open(path, "r", encoding="utf-8") as f:
                state = _json.load(f)
            global_state = state.get("global", state)
            raw = global_state.get("prompt_dir_index", [])
            entries = []
            for e in raw if isinstance(raw, list) else []:
                if not isinstance(e, dict) or e.get("prompt") is None or e.get("direction") is None:
                    continue
                try:
                    entries.append((serializable_to_tensor(e["prompt"]).float(),
                                    serializable_to_tensor(e["direction"]).float(),
                                    float(e.get("direction_magnitude", 0.0))))
                except Exception:
                    continue
            return entries
        except Exception:
            return []

    def _resolve_prompt_keyed_direction(self, index, cond, k=3, min_sim=0.5):
        """Retrieve the liked direction learned on the prompts NEAREST this scene's prompt.

        `cond` is the scene's c_crossattn ([.., seq, D]); it is mean-pooled to [D] and
        cosine-matched against every indexed prompt fingerprint. Returns a unit [D]
        direction = the similarity-weighted mean of the top-k neighbours scoring above
        min_sim, or None when nothing is close enough (the caller then falls back to the
        global liked_dir). This is the whole point of the feature: a prompt about a forest
        pulls the direction that worked on forests, not the average across every prompt
        ever rated. No extra model forward — one pooled cosine sweep + a vector mean."""
        if not index or not isinstance(cond, torch.Tensor):
            return None
        try:
            pooled = cond.detach().float()
            while pooled.dim() > 1:
                pooled = pooled.mean(dim=0)  # [D], same pooling as _v2_pool_conditioning
            pooled_u = torch.nn.functional.normalize(pooled, dim=0)
            scored = []
            for prompt, direction, _mag in index:
                if list(prompt.shape) != list(pooled.shape) or list(direction.shape) != list(pooled.shape):
                    continue
                sim = float(torch.nn.functional.cosine_similarity(
                    pooled_u, torch.nn.functional.normalize(prompt.to(pooled.device), dim=0), dim=0))
                if sim >= min_sim:
                    scored.append((sim, direction.to(pooled.device)))
            if not scored:
                return None
            scored.sort(key=lambda s: s[0], reverse=True)
            acc = None
            for sim, direction in scored[:max(1, int(k))]:
                term = torch.nn.functional.normalize(direction, dim=0) * sim
                acc = term if acc is None else acc + term
            if acc is None or float(acc.norm()) <= 0.0:
                return None
            return torch.nn.functional.normalize(acc, dim=0)
        except Exception:
            return None

    def _direction_in_cond_space(self, model, raw_cond, direction, cond):
        """Express a learned taste direction in the space of the conditioning the DiT
        actually consumes.

        On LTX the two are the same tensor and this is an identity check. On H3 they are
        not: ``MiniMaxH3.extra_conds`` runs ``preprocess_text_embeds`` (condition_proj +
        the token refiner) once per sampling run, so ``c_crossattn`` is refined DiT hidden
        state (5376) while every direction in the taste store was captured from the raw
        Qwen3-VL conditioning (5120). Adding one to the other is a hard shape error in
        embed_guidance and a silently caught one in score_slider.

        condition_proj is linear but the refiner is not (attention + RMSNorm), so the
        direction is carried across by a finite difference through the model's OWN
        preprocessor -- the hidden-space image of nudging the raw conditioning by
        `direction`, which is exactly what embed guidance means on LTX. Two refiner calls
        over ~150 text tokens, once per scene.

        Returns a unit [D] direction in `cond`'s space, or None when there is no
        preprocessor to map through (the caller then skips steering rather than crashing).
        """
        if not isinstance(direction, torch.Tensor) or not isinstance(cond, torch.Tensor):
            return None
        if int(direction.shape[-1]) == int(cond.shape[-1]):
            return direction
        pre = getattr(getattr(getattr(model, "model", None), "diffusion_model", None),
                      "preprocess_text_embeds", None)
        if pre is None or not isinstance(raw_cond, torch.Tensor) or \
                int(raw_cond.shape[-1]) != int(direction.shape[-1]):
            return None
        try:
            with torch.no_grad():
                base = raw_cond.to(device=cond.device, dtype=cond.dtype)
                while base.dim() < 3:
                    base = base.unsqueeze(0)
                unit = torch.nn.functional.normalize(direction.float(), dim=-1).to(
                    device=base.device, dtype=base.dtype)
                # 5% of the conditioning's own RMS: clear of bf16 noise, still inside the
                # refiner's locally-linear regime.
                eps = 0.05 * float(base.float().pow(2).mean().sqrt())
                delta = (pre(base + eps * unit) - pre(base)).float()
            while delta.dim() > 1:
                delta = delta.mean(dim=0)
            if not torch.isfinite(delta).all() or float(delta.norm()) <= 0.0:
                return None
            return torch.nn.functional.normalize(delta, dim=0)
        except Exception:
            return None

    def _taste_direction_resolver(self, model, raw_cond, fixed_dir, label):
        """Lazily map a taste direction into the consumed conditioning space, once.

        Deferred to the first gated step deliberately: at wrapper-build time the model may
        still sit on the offload device, and the finite difference wants the loaded one."""
        state = {}

        def resolve(cond):
            if "d" not in state:
                mapped = self._direction_in_cond_space(model, raw_cond, fixed_dir, cond)
                state["d"] = mapped
                if mapped is None:
                    print(f"[FunPackSceneChain] {label}: the learned taste direction is "
                          f"{int(fixed_dir.shape[-1])}-dim, this model consumes "
                          f"{int(cond.shape[-1])}-dim conditioning, and it exposes no "
                          "preprocessor to map between them \u2014 steering skipped")
                elif int(mapped.shape[-1]) != int(fixed_dir.shape[-1]):
                    print(f"[FunPackSceneChain] {label}: taste direction lifted "
                          f"{int(fixed_dir.shape[-1])} -> {int(mapped.shape[-1])} through "
                          "the model's own text preprocessor")
            return state["d"]

        return resolve

    def _build_embed_guidance_wrapper(self, model, liked_dir, strength, value_fn=None, raw_cond=None, ramp_fn=None):
        """Register a model_function_wrapper that nudges conditioning toward the
        liked quality direction at each denoising step. Uses value function gradient
        when available, falls back to the fixed liked direction otherwise."""
        old_wrapper = model.model_options.get("model_function_wrapper")
        _ramp = ramp_fn or (lambda sigma: max(0.0, 1.0 - float(sigma) * 2.0))
        fixed_dir = torch.nn.functional.normalize(liked_dir.float(), dim=-1)
        resolve_dir = self._taste_direction_resolver(model, raw_cond, fixed_dir, "embed_guidance")
        vf_state = {}

        def resolve_vf(cond):
            """Ask the value function for a gradient in the space it was FITTED in.

            Its first layer is [raw_dim, 256], so handing it the refined conditioning the
            DiT consumes is a matmul error, not a steering signal — the run then silently
            fell back to the fixed direction while the report still claimed the value
            function was driving. Take the gradient at the raw conditioning and carry the
            answer across with the same bridge the fixed direction uses.

            Cached: comfy builds the conds once per run, so within a scene this gradient is
            the same every step — and identical across score_slider's +/- poles, which is
            what makes the embed nudge cancel in (eps_+ - eps_-) as its docstring intends.
            """
            if "d" in vf_state:
                return vf_state["d"]
            vf_state["d"] = None
            try:
                source = raw_cond if isinstance(raw_cond, torch.Tensor) else cond
                grad = value_fn.gradient(source)
                unit = torch.nn.functional.normalize(grad.float(), dim=-1)
                vf_state["d"] = self._direction_in_cond_space(model, raw_cond, unit, cond)
                if vf_state["d"] is not None:
                    # Say so out loud: the fixed-direction fallback announces itself when it
                    # lifts, so silence here reads as "nothing happened" rather than success.
                    lifted = int(vf_state["d"].shape[-1]) != int(unit.shape[-1])
                    print("[FunPackSceneChain] embed_guidance: steering on the value function "
                          "gradient" + (f", lifted {int(unit.shape[-1])} -> "
                                        f"{int(vf_state['d'].shape[-1])} through the model's own "
                                        "text preprocessor" if lifted else ""))
                if vf_state["d"] is None:
                    print("[FunPackSceneChain] embed_guidance: the value function gradient "
                          "cannot be mapped onto this model's conditioning, using fixed direction")
            except Exception as _e:
                print(f"[FunPackSceneChain] embed_guidance: value function gradient failed ({_e}), using fixed direction")
            return vf_state["d"]

        def _embed_wrapper(apply_fn, args, _ew=old_wrapper, _fixed=fixed_dir, _vf=value_fn, _s=strength):
            c = args.get("c") or {}
            cond = c.get("c_crossattn")
            if cond is not None:
                ts = args.get("timestep")
                try:
                    sigma = float(ts.max().item()) if ts is not None else 1.0
                except Exception:
                    sigma = 1.0
                scale = _ramp(sigma)
                if scale > 0:
                    d = None
                    if _vf is not None:
                        _mapped_vf = resolve_vf(cond)
                        if _mapped_vf is not None:
                            d = _mapped_vf.to(cond.device, cond.dtype).expand_as(cond)
                    if d is None:
                        _mapped = resolve_dir(cond)
                        d = (_mapped.to(cond.device, cond.dtype).expand_as(cond)
                             if _mapped is not None else None)
                    if d is not None:
                        new_c = dict(c)
                        # Steer video channels only; audio keeps its original text conditioning.
                        new_c["c_crossattn"] = self._protect_audio(cond + (_s * scale) * d, cond)
                        args = dict(args)
                        args["c"] = new_c
            if _ew is not None:
                return _ew(apply_fn, args)
            return apply_fn(args["input"], args["timestep"], **args.get("c", {}))

        model.model_options["model_function_wrapper"] = _tag_scene_wrapper(_embed_wrapper, old_wrapper)
        return old_wrapper

    def _build_score_slider_wrapper(self, model, liked_dir, eta, bad_dir=None, raw_cond=None, ramp_fn=None):
        """FreeSliders (arxiv 2511.00103) in score space, sourced from the learned taste.

        embed_guidance nudges the conditioning embedding once per step. FreeSliders
        instead works in noise-prediction space: synthesize a +/- concept pair from
        the base conditioning and the learned liked direction (no concept prompts
        needed), then combine predictions
            eps_mod = eps_base + eta * (eps_+ - eps_-)
        where c_+/- = base_cond +/- a calibrated step of the liked direction. Three
        forward passes on low-sigma (quality-phase) steps only; a single pass at high
        sigma reproduces the paper's k-step base-only warmup and keeps the 2x cost off
        the early steps. Video-only: the +/- conds are audio-protected and the eps blend
        is confined to the packed video stream, so audio rides eps_base. Returns the
        previous wrapper so the caller can restore it.

        Contrastive pair: when `bad_dir` is available (3+ disliked/awful gens learned a
        bad_dir in the same store), the MINUS pole is built from the real disliked
        direction instead of mirroring liked. The axis becomes (liked - bad) — a true
        good-vs-bad corrective vector — so the step pushes the noise prediction away from
        the conditioning region that produced rated-bad gens, not just toward "anti-liked".
        Falls back to the symmetric +/-liked mirror when no bad_dir exists.

        Composes with embed_guidance: when both are on, all three passes route through
        the embed wrapper, so its nudge cancels in (eps_+ - eps_-) and only the slider
        axis remains, while eps_base still carries the embed steering."""
        old_wrapper = model.model_options.get("model_function_wrapper")
        _ramp = ramp_fn or (lambda sigma: max(0.0, 1.0 - float(sigma) * 2.0))
        fixed_dir = torch.nn.functional.normalize(liked_dir.float(), dim=-1)
        bad_fixed = None
        if bad_dir is not None:
            bad_fixed = torch.nn.functional.normalize(bad_dir.float(), dim=-1)
        resolve_dir = self._taste_direction_resolver(model, raw_cond, fixed_dir, "score_slider")
        resolve_bad = (self._taste_direction_resolver(model, raw_cond, bad_fixed, "score_slider(bad pole)")
                       if bad_fixed is not None else None)

        def _call(apply_fn, a):
            if old_wrapper is not None:
                return old_wrapper(apply_fn, a)
            return apply_fn(a["input"], a["timestep"], **a.get("c", {}))

        def _slider_wrapper(apply_fn, args, _fixed=fixed_dir, _bad=bad_fixed, _eta=float(eta)):
            c = args.get("c") or {}
            cond = c.get("c_crossattn")
            ts = args.get("timestep")
            try:
                sigma = float(ts.max().item()) if ts is not None else 1.0
            except Exception:
                sigma = 1.0
            ramp = _ramp(sigma)  # base-only warmup early in the schedule
            if cond is None or _eta == 0.0 or ramp <= 0.0:
                return _call(apply_fn, args)
            try:
                _mapped = resolve_dir(cond)
                if _mapped is None:
                    return _call(apply_fn, args)
                d = _mapped.to(cond.device, cond.dtype).expand_as(cond)
                # Calibrated finite-difference step along the taste axis (per-token,
                # mirrors _v2_apply_direction's NORM_SCALE=0.3 calibration but halved
                # so the symmetric +/- spread stays non-destructive).
                scale = 0.15 * cond.norm(dim=-1, keepdim=True)
                step = d * scale
                cond_plus = self._protect_audio(cond + step, cond)
                if _bad is not None:
                    # Contrastive minus pole: step toward the learned BAD direction so the
                    # (eps_+ - eps_-) axis becomes good-vs-bad, steering away from the
                    # conditioning region that produced rated-bad gens. Same step magnitude
                    # keeps the pair symmetric in size, asymmetric in direction.
                    _mapped_bad = resolve_bad(cond) if resolve_bad is not None else None
                    if _mapped_bad is not None:
                        bd = _mapped_bad.to(cond.device, cond.dtype).expand_as(cond)
                        cond_minus = self._protect_audio(cond + bd * scale, cond)
                    else:
                        # Bad pole unmappable: fall back to the symmetric mirror rather
                        # than dropping the slider entirely.
                        cond_minus = self._protect_audio(cond - step, cond)
                else:
                    cond_minus = self._protect_audio(cond - step, cond)
                eps_base = _call(apply_fn, args)
                args_plus = dict(args); cp = dict(c); cp["c_crossattn"] = cond_plus; args_plus["c"] = cp
                args_minus = dict(args); cm = dict(c); cm["c_crossattn"] = cond_minus; args_minus["c"] = cm
                eps_plus = _call(apply_fn, args_plus)
                eps_minus = _call(apply_fn, args_minus)
                delta = (eps_plus - eps_minus) * (_eta * ramp)
                vmask = _packed_video_mask(model, args["input"])
                if vmask is not None and vmask.shape[-1] == delta.shape[-1]:
                    delta = delta * vmask.to(delta.device, delta.dtype)
                return eps_base + delta
            except Exception as _e:
                print(f"[FunPackSceneChain] score_slider failed ({_e}), using base prediction")
                return _call(apply_fn, args)

        model.model_options["model_function_wrapper"] = _tag_scene_wrapper(_slider_wrapper, old_wrapper)
        return old_wrapper

    def _resolve_frame_index(self, total, frame_idx):
        total = max(0, int(total))
        if total <= 0:
            return 0
        idx = int(frame_idx)
        if idx < 0:
            idx = total + idx
        return max(0, min(idx, total - 1))

    def _pixel_frame_idx(self, chunk_frames, apply_at, time_scale):
        """Map apply_at (pixel index, negative from end) to LTX keyframe pixel index."""
        F = max(1, int(chunk_frames))
        time_scale = max(1, int(time_scale))
        at = self._resolve_frame_index(F, int(apply_at))
        causal_fix = (at == 0)
        if causal_fix:
            return 0, True
        return 1 + (at - 1) * time_scale, False

    # Set per run from num_frames_per_scene; H3's keyframe layout needs the scene's PIXEL
    # length to place a last-frame pin, and nothing else in the guide path carries it.
    _h3_frame_count = 0
    _is_h3 = False
    _h3_ref_cache: dict = {}
    _h3_mode_noted = False
    _h3_clock_unreachable_noted = False
    _alg_unreachable_noted = False

    def _h3_add_keyframes(self, conditioning, pins, frame_count, aug=None):
        """Merge keyframe pins into a conditioning, keeping any already attached.

        H3's payload takes a LIST of pins (a first-frame anchor and a last-frame target can
        coexist), so every producer here — the i2v anchor and the guide path — has to add to
        the list rather than overwrite it. Pins are keyed by resolved frame index; writing the
        same index twice replaces it, which is the only sane reading of two anchors on one
        frame. ``minimax_visual_cond_noise_aug`` stays payload-wide as the DiT defines it, so
        the last explicit strength written still wins — stated, not silently averaged.
        """
        if not conditioning or not pins:
            return conditioning
        existing = self._conditioning_value(conditioning, "minimax_keyframes") or []
        merged = {int(p["resolved_frame_index"]): p for p in existing}
        for pin in pins:
            merged[int(pin["resolved_frame_index"])] = pin
        values = {
            "minimax_keyframes": [merged[k] for k in sorted(merged)],
            "minimax_frame_count": int(frame_count),
        }
        if aug is not None:
            values["minimax_visual_cond_noise_aug"] = float(aug)
        return self._condition_with_values(conditioning, values)

    def _apply_h3_anchor(self, positive, chunk, vae, image, strength=1.0):
        """Pin an i2v anchor image as H3's frame-0 keyframe (the fl2va conditioning path).

        LTX anchors an image by writing it into the starting latent (LTXVImgToVideoInplace)
        and masking that frame out of denoising. H3 does not condition that way: the anchor
        is a CONDITION ROW packed beside the text, never denoised and never rendered, and a
        latent written frame-0 is just noise the model is free to overwrite. So on H3 the
        same user intent — "this scene starts from this image" — becomes a keyframe pin.

        Returns (positive, applied). The image is encoded at the scene's own canvas, read
        off the chunk, so it matches whatever the latent template was built for.
        """
        if image is None:
            return positive, False
        try:
            from . import minimax_h3 as h3mod
        except ImportError:
            import minimax_h3 as h3mod
        try:
            video = self._latent_tensors(chunk)[0]
            if video.dim() != 5:
                raise ValueError("video latent is not 5D")
            width = int(video.shape[4]) * h3mod.SPATIAL_DOWNSCALE
            height = int(video.shape[3]) * h3mod.SPATIAL_DOWNSCALE
            pin = h3mod.encode_keyframe(vae, image, width, height, 0, crop="disabled")
        except Exception as error:
            print(f"[FunPackSceneChain] H3: i2v anchor skipped — could not encode it ({error}).")
            return positive, False
        if pin.get(h3mod.REGION_META) is not None:
            kept = int(pin[h3mod.REGION_META].sum())
            total = int(pin[h3mod.REGION_META].numel())
            _log.feature(
                "FunPackSceneChain", "Region lock", True,
                f"{kept} of {total} patches ({100.0 * kept / max(1, total):.0f}%) "
                f"conditioned, the rest is the model's to invent. "
                f"EXPERIMENTAL: the checkpoint was trained on whole condition frames. Feed a "
                f"fully opaque image to pin the whole frame as before.")
        aug = max(0.0, min(1.0, float(strength)))
        positive = self._h3_add_keyframes(positive, [pin], self._h3_frame_count,
                                          aug=None if aug >= 1.0 else aug)
        return positive, True

    def _h3_has_first_frame_pin(self, conditioning):
        pins = self._conditioning_value(conditioning, "minimax_keyframes") or []
        return any(int(p.get("resolved_frame_index", -1)) == 0 for p in pins)

    def _h3_continuation_pin(self, positive, carry_source):
        """Continue an H3 scene from the previous one, the only way H3 can be told to.

        LTX chains through the LATENT: the previous scene's tail is copied into the leading
        frames and masked out of denoising, and the model treats those frames as context.
        H3 has no latent conditioning at all — a pre-filled frame is, in its own terms, just
        noise it is free to overwrite. The mask still holds those pixels in the output, so
        the seam looked right while the rest of the shot was generated knowing nothing about
        what came before: the chain produced a batch of unrelated clips.

        The previous scene's last latent frame becomes this scene's frame-0 keyframe pin,
        which is exactly the conditioning an i2v anchor uses. No VAE round trip: the frame is
        already in latent space, and the pin path takes latents (the guide path already
        relies on that). Skipped when something already pinned frame 0 — an explicit anchor
        or a wired first_frame outranks a carried tail.

        Returns (positive, applied).
        """
        if carry_source is None or self._h3_has_first_frame_pin(positive):
            return positive, False
        try:
            video = self._latent_tensors(carry_source)[0]
            if getattr(video, "ndim", 0) < 5 or int(video.shape[2]) < 1:
                return positive, False
            frame = video[:1, :, -1:].clone()
        except Exception as error:  # noqa: BLE001
            print(f"[FunPackSceneChain] H3: continuation pin skipped — {error}.")
            return positive, False
        positive = self._h3_add_keyframes(
            positive, [{"resolved_frame_index": 0, "latent": frame}], self._h3_frame_count)
        return positive, True

    def _h3_rescale_pins(self, conditioning, height, width):
        """Bring H3 keyframe pins onto a resized latent grid.

        A pin is a latent frame that core packs into the sequence as condition ROWS, so its
        token count is fixed by the grid it was encoded on. After a resolution-changing
        second_pass_op that grid no longer exists, and the model fails placing it:

            shape mismatch: value tensor of shape [168, 96] cannot be broadcast to
            indexing result of shape [672, 96]

        (96 = 24 channels x the 2x2 patch; 672 = 4 x 168, i.e. 2x spatial is 4x the tokens.)

        Dropping them is what the LTX path does with guide keyframes, but on H3 the pin IS
        the anchor, and un-pinning it for pass 2 is precisely what a second pass must not do.
        So they are resampled. Bicubic, not the upsampler: a pin is a condition row, and what
        has to survive is its structure, not invented detail.
        """
        height, width = int(height), int(width)
        out, changed = [], 0
        for entry in conditioning or []:
            if not (isinstance(entry, (list, tuple)) and len(entry) == 2
                    and isinstance(entry[1], dict) and entry[1].get("minimax_keyframes")):
                out.append(entry)
                continue
            pins = []
            for pin in entry[1]["minimax_keyframes"]:
                latent = pin.get("latent")
                if (getattr(latent, "ndim", 0) == 5
                        and tuple(latent.shape[-2:]) != (height, width)):
                    frames = latent.movedim(2, 0).flatten(0, 1)          # [B*T, C, H, W]
                    resized = torch.nn.functional.interpolate(
                        frames.float(), size=(height, width), mode="bicubic",
                        align_corners=False).to(latent.dtype)
                    latent = resized.unflatten(
                        0, (latent.shape[0], latent.shape[2])).movedim(1, 2)
                    changed += 1
                pins.append({**pin, "latent": latent})
            meta = {**entry[1], "minimax_keyframes": pins}
            out.append([entry[0], meta])
        return out, changed

    def _h3_external_pins(self, conditioning):
        """Rescue the keyframe pins from a MiniMax H3 Image to Video node's conditioning.

        That node encodes first_frame/last_frame into `minimax_keyframes` and hands them out
        on its CONDITIONING output — which this pipeline drops, because the sampler's positive
        comes from Studio. Wiring the node's conditioning into `h3_keyframes` lets the pins
        travel anyway. Only the pins are taken; the node's own prompt encode is not used.

        Returns ``{"first": [...], "last": [...], "aug": float|None}``. The node writes a
        first_frame at index 0 and a last_frame at ITS frame_count - 1, which is a per-node
        length that need not match this run's scene length — so pins are classified here and
        re-indexed against the scene they land on, never trusted as absolute positions.
        """
        pins = self._conditioning_value(conditioning, "minimax_keyframes") or []
        if not pins:
            return None
        source_count = self._conditioning_value(conditioning, "minimax_frame_count")
        first, last = [], []
        for pin in pins:
            try:
                index = int(pin["resolved_frame_index"])
            except (KeyError, TypeError, ValueError) as _e:
                _log.failed("FunPackSceneChain", "wired keyframe pin", _e,
                            "that pin is DROPPED — the scene it anchored is unpinned")
                continue
            (first if index == 0 else last).append(pin)
        if not first and not last:
            return None
        aug = self._conditioning_value(conditioning, "minimax_visual_cond_noise_aug")
        return {
            "first": first,
            "last": last,
            "aug": float(aug) if aug is not None else None,
            "source_count": int(source_count) if source_count else None,
        }

    def _apply_h3_external_pins(self, positive, pins, scene_index, scene_count):
        """Place rescued pins on the scene they belong to: first on the opening scene, last
        on the closing one. A one-scene run gets both, which is the plain single-clip case.

        Returns (positive, applied_labels)."""
        frame_count = max(1, int(self._h3_frame_count))
        placed, labels = [], []
        if scene_index == 0 and pins["first"]:
            placed.extend({**p, "resolved_frame_index": 0} for p in pins["first"])
            labels.append("first")
        if scene_index == scene_count - 1 and pins["last"]:
            # re-indexed onto THIS scene's last frame — the source node's frame_count is its
            # own, and a pin at a stale index would land mid-clip, where the layout refuses it
            placed.extend({**p, "resolved_frame_index": frame_count - 1} for p in pins["last"])
            labels.append("last")
        if not placed:
            return positive, []
        return self._h3_add_keyframes(positive, placed, frame_count, aug=pins["aug"]), labels

    def _append_h3_keyframe(self, guide_frame, apply_at, strength, positive, negative):
        """H3's equivalent of an LTX guide: a keyframe pin carried on the conditioning.

        LTX appends the guide as an extra latent frame and masks it out afterwards. H3 packs
        condition rows into the sequence itself — they are never denoised and never rendered,
        so there is no latent to append and no tail to crop (the returned tail is 0).

        Stock ``PackedLayout`` places a pin only at the FIRST or LAST pixel frame and raises
        for anything else. Those two are the endpoints of ONE straight line — the packed
        sequence's time axis advances at a fixed rate per pixel frame — so
        ``install_interior_keyframes`` (attempted by ``keyframe_indices_supported``) extends
        the same rule to the frames between them. When it cannot install, a mid-clip request
        is refused here, loudly, rather than crashing several seconds into the sample.

        An interior pin is EXPERIMENTAL in a way the endpoints are not: fl2va was trained with
        condition rows at the two ends and nowhere between, so the coordinate is representable
        (MM-RoPE is continuous in t) without being something the weights have seen. It costs no
        extra model call either way.
        """
        try:
            from .minimax_h3 import keyframe_indices_supported, keyframe_is_endpoint
        except ImportError:
            from minimax_h3 import keyframe_indices_supported, keyframe_is_endpoint

        frame_count = max(1, int(self._h3_frame_count))
        at = self._resolve_frame_index(frame_count, int(apply_at))
        if not keyframe_indices_supported(at, frame_count):
            print(f"[FunPackSceneChain] H3: guide at pixel frame {at} skipped — this ComfyUI "
                  f"pins only the first (0) or last ({frame_count - 1}) frame. Use a reference "
                  f"image (ref2va) for mid-clip guidance instead.")
            return positive, negative, 0
        if not keyframe_is_endpoint(at, frame_count):
            _log.feature(
                "FunPackSceneChain", "Interior keyframe pin", True,
                f"pinned at pixel frame {at} of {frame_count}. "
                f"EXPERIMENTAL — the checkpoint was trained on first/last pins "
                f"only; if the frame does not land, move the guide to 0 or {frame_count - 1}.")

        pins = [{"resolved_frame_index": int(at), "latent": guide_frame}]
        # strength maps onto the DiT's condition noise augmentation: 1.0 pins the clean latent,
        # lower values mix noise into the condition rows. It is a single payload-wide value, so
        # with several pins in one scene the last one written wins — stated here rather than
        # silently averaged.
        aug = max(0.0, min(1.0, float(strength)))
        aug = None if aug >= 1.0 else aug
        positive = self._h3_add_keyframes(positive, pins, frame_count, aug=aug)
        negative = self._h3_add_keyframes(negative, pins, frame_count, aug=aug) if negative else negative
        return positive, negative, 0

    def _report_h3_checkpoint_mode(self, positive):
        """Say once per run which H3 checkpoint this run's conditioning actually needs.

        H3 ships two DiTs (fl2va for keyframe pins, ref2va for reference blocks) and neither
        rejects the other's conditioning — a mismatch costs quality, silently. Nothing in the
        state dict identifies the variant, so the run reports the mode it is in and leaves
        matching the file to the user.
        """
        if self._h3_mode_noted:
            return
        try:
            from .minimax_h3 import checkpoint_mode_note
        except ImportError:
            from minimax_h3 import checkpoint_mode_note
        note = checkpoint_mode_note(
            bool(self._conditioning_value(positive, "minimax_keyframes")),
            bool(self._conditioning_value(positive, "minimax_refs")),
        )
        self._h3_mode_noted = True
        if note:
            _log.note("FunPackSceneChain", f"H3 conditioning: {note}")

    def _apply_h3_references(self, positive, chunk, vae, audio_vae=None):
        """Turn Studio's resolved ref2va order into the DiT's `minimax_refs` blocks.

        Studio owns the CLIP, so it baked the presentation ("<Picture 1>: <vision block>")
        and recorded WHICH references it presented, in order, as `funpack_h3_refs`. This
        side owns the VAE, so it encodes exactly that list. The order is the contract: drop
        or reorder an entry here and every later "<Picture i>" in the prompt points at a
        different reference than the one the text encoder saw.

        This is the native replacement for the LTX Best-FaceID path, not a port of it —
        H3 packs reference blocks into the sequence itself, so there is nothing to project,
        nothing to append to the text context and nothing to slice back off the output.
        """
        spec = self._conditioning_value(positive, "funpack_h3_refs")
        if not spec:
            return positive, 0
        try:
            from . import minimax_h3 as h3mod
        except ImportError:
            import minimax_h3 as h3mod

        video = self._latent_tensors(chunk)[0]
        if video.dim() != 5:
            print("[FunPackSceneChain] H3 references skipped — couldn't read a 5D video latent.")
            return positive, 0
        width = int(video.shape[4]) * h3mod.SPATIAL_DOWNSCALE
        height = int(video.shape[3]) * h3mod.SPATIAL_DOWNSCALE

        # Every scene in the chain gets the same references, and re-loading + re-VAE-encoding
        # them per scene is pure waste. Cached per RUN only (cleared at the top of sample()),
        # never across requests — the same rule the Studio encode cache follows.
        cache_key = (str(spec), width, height, audio_vae is not None)
        cached = self._h3_ref_cache.get(cache_key)
        if cached is not None:
            blocks = cached
            if not blocks:
                return positive, 0
            return self._condition_with_values(positive, {"minimax_refs": blocks}), len(blocks)

        resolved, load_skips = h3mod.resolve_ref_spec(h3mod.normalize_ref_spec(spec))
        for filename, why in load_skips:
            print(f"[FunPackSceneChain] H3 reference '{filename}' skipped — {why}.")
        blocks, encode_skips = h3mod.ref_blocks_from_spec(
            resolved, vae, width, height, audio_vae=audio_vae)
        for filename, why in encode_skips:
            print(f"[FunPackSceneChain] H3 reference '{filename}' skipped — {why}. The prompt's "
                  f"reference numbering was already baked at encode time, so every later "
                  f"<Audio j> now points one reference earlier than you wrote it.")
        self._h3_ref_cache[cache_key] = blocks
        if not blocks:
            return positive, 0
        print(f"[FunPackSceneChain] H3 ref2va: {len(blocks)} reference block(s) packed "
              f"({', '.join(b['kind'] for b in blocks)}) at {width}x{height}. Reference "
              f"tokens ride every sampling step, so more/longer references cost time.")
        return self._condition_with_values(positive, {"minimax_refs": blocks}), len(blocks)

    def _append_guide_latent(self, chunk, guide_frame, apply_at, strength, positive, negative, vae):
        """Append one guide latent frame with LTX guide attention at apply_at."""
        if self._is_h3:
            positive, negative, tail = self._append_h3_keyframe(
                guide_frame, apply_at, strength, positive, negative)
            return chunk, positive, negative, tail
        try:
            from comfy_extras.nodes_lt import LTXVAddGuide, _append_guide_attention_entry
        except ImportError as _e:
            # A PRIVATE upstream symbol. A ComfyUI rename lands here and every LTX guide
            # stops being applied — until now indistinguishable from having none.
            _log.failed("FunPackSceneChain", "guide append (ComfyUI LTX guide API)", _e,
                        "the scene renders with NO guide applied")
            return chunk, positive, negative, 0

        chunk_tensors = self._latent_tensors(chunk)
        if not chunk_tensors:
            return chunk, positive, negative, 0
        guide_frame = guide_frame.to(device=chunk_tensors[0].device, dtype=chunk_tensors[0].dtype)
        F_chunk = self._tensor_frames(chunk_tensors[0])
        scale_factors = getattr(vae, 'downscale_index_formula', [8, 8, 8])
        time_scale = int(scale_factors[0]) if hasattr(scale_factors, '__getitem__') else 8
        pixel_frame_idx, causal_fix = self._pixel_frame_idx(F_chunk, apply_at, time_scale)

        positive = LTXVAddGuide.add_keyframe_index(
            positive, pixel_frame_idx, guide_frame, scale_factors, causal_fix=causal_fix
        )
        negative = LTXVAddGuide.add_keyframe_index(
            negative, pixel_frame_idx, guide_frame, scale_factors, causal_fix=causal_fix
        )
        guide_latent_shape = [guide_frame.shape[2], guide_frame.shape[3], guide_frame.shape[4]]
        pre_filter_count = guide_frame.shape[2] * guide_frame.shape[3] * guide_frame.shape[4]
        positive, negative = _append_guide_attention_entry(
            positive, negative, pre_filter_count, guide_latent_shape, strength=float(strength)
        )

        result = self._clone_latent(chunk)
        tensors = self._latent_tensors(result)
        masks = self._latent_masks(result, len(tensors))
        if masks[0] is None:
            masks[0] = torch.ones(
                tensors[0].shape[0], 1, tensors[0].shape[2], 1, 1,
                dtype=torch.float32, device=tensors[0].device,
            )
        guide_mask = torch.full_like(masks[0][:, :, :1], max(0.0, 1.0 - float(strength)))
        tensors[0] = torch.cat([tensors[0], guide_frame], dim=2)
        masks[0] = torch.cat([masks[0], guide_mask.to(masks[0].device, masks[0].dtype)], dim=2)
        if self._is_nested(result.get("samples")):
            result["samples"] = comfy.nested_tensor.NestedTensor(tensors)
            result["noise_mask"] = comfy.nested_tensor.NestedTensor(masks)
        else:
            result["samples"] = tensors[0]
            result["noise_mask"] = masks[0]
        return result, positive, negative, 1

    def _append_template_guide_at(self, chunk, template, frame_idx, apply_at, strength, positive, negative, vae):
        """Studio-default path: template protected prefix at apply_at=0; else LTX guide attention."""
        template_tensors = self._latent_tensors(template)
        template_masks = self._latent_masks(template, len(template_tensors))
        if not template_tensors:
            return chunk, positive, negative, 0, 0
        protected = self._protected_prefix_frames(template_masks[0], self._tensor_frames(template_tensors[0]))
        if protected <= 0:
            return chunk, positive, negative, 0, 0
        src_at = self._resolve_frame_index(protected, frame_idx)
        guide_frame = self._time_slice(template_tensors[0], src_at, src_at + 1)
        if int(apply_at) == 0 and src_at == 0:
            chunk_tensors = self._latent_tensors(chunk)
            if not chunk_tensors:
                return chunk, positive, negative, 0, 0
            guide = self._time_slice(template_tensors[0], 0, protected).to(
                device=chunk_tensors[0].device, dtype=chunk_tensors[0].dtype,
            )
            guide_mask = self._time_slice(template_masks[0], 0, protected).to(
                device=chunk_tensors[0].device, dtype=chunk_tensors[0].dtype,
            )
            out_tensors = list(chunk_tensors)
            out_masks = self._latent_masks(chunk, len(out_tensors))
            if out_masks[0] is None:
                out_masks[0] = torch.ones_like(out_tensors[0])
            target_mask = self._time_slice(out_masks[0], 0, protected).to(guide_mask.device, guide_mask.dtype)
            guide_mask = self._expand_mask_like(guide_mask, target_mask)
            out_tensors[0] = torch.cat([guide, out_tensors[0]], dim=2)
            out_masks[0] = torch.cat([guide_mask, out_masks[0].to(guide_mask.device, guide_mask.dtype)], dim=2)
            if self._is_nested(chunk.get("samples")):
                chunk["samples"] = comfy.nested_tensor.NestedTensor(out_tensors)
                chunk["noise_mask"] = comfy.nested_tensor.NestedTensor(out_masks)
            else:
                chunk["samples"] = out_tensors[0]
                chunk["noise_mask"] = out_masks[0]
            return chunk, positive, negative, protected, 0
        chunk, positive, negative, tail = self._append_guide_latent(
            chunk, guide_frame, apply_at, strength, positive, negative, vae,
        )
        return chunk, positive, negative, 0, tail

    def _append_scene_guide_at(self, chunk, scene_output, frame_idx, apply_at, strength, positive, negative, vae):
        tensors = self._latent_tensors(scene_output)
        if not tensors:
            return chunk, positive, negative, 0, 0
        total = self._tensor_frames(tensors[0])
        at = self._resolve_frame_index(total, frame_idx)
        guide_frame = self._time_slice(tensors[0], at, at + 1)
        chunk, positive, negative, tail = self._append_guide_latent(
            chunk, guide_frame, apply_at, strength, positive, negative, vae,
        )
        return chunk, positive, negative, 0, tail

    def _parse_scene_guides(self, raw):
        if not raw or not str(raw).strip():
            return None
        try:
            import json
            data = json.loads(str(raw))
        except Exception:
            return None
        if not isinstance(data, dict) or not data.get("stack_enabled"):
            return None
        return data

    def _parse_scene_anchors(self, raw):
        if not raw or not str(raw).strip():
            return {}
        try:
            import json
            data = json.loads(str(raw))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _load_image_tensor(self, filename, keep_alpha=False):
        """Load an input-directory image as [1, H, W, 3], or [1, H, W, 4] on request.

        Alpha is dropped by default because every consumer but one wants pixels. The H3
        anchor path asks for it: on that path a transparent area means "I have nothing to say
        about this part of the frame", which becomes a region lock rather than a colour.
        """
        import os
        try:
            import folder_paths
            import numpy as np
            from PIL import Image
        except ImportError:
            return None
        path = os.path.join(folder_paths.get_input_directory(), filename)
        if not os.path.isfile(path):
            return None
        try:
            img = Image.open(path)
            has_alpha = keep_alpha and (img.mode in ("RGBA", "LA") or "transparency" in img.info)
            img = img.convert("RGBA" if has_alpha else "RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(arr)[None,]
        except Exception:
            return None

    def _invoke_img2video_inplace(self, vae, image, latent, strength=1.0):
        """Run LTXVImgToVideoInplace — image → starting latent (i2v anchor, not a guide)."""
        try:
            from comfy_extras.nodes_lt import LTXVImgToVideoInplace
        except ImportError:
            return None
        for method in ("execute", "generate"):
            fn = getattr(LTXVImgToVideoInplace, method, None)
            if fn is None:
                continue
            for args in (
                (vae, image, latent, strength, False),
                (vae, image, latent, strength),
            ):
                try:
                    out = fn(*args)
                    if hasattr(out, "args") and out.args:
                        out = out.args[0]
                    elif isinstance(out, tuple) and out:
                        out = out[0]
                    if isinstance(out, dict) and "samples" in out:
                        return out
                except (TypeError, ValueError):
                    continue
        return None

    def _apply_img2video_to_video_latent(self, vae, image, chunk, strength=1.0):
        """Apply Img2Video inplace to the video stream of an (optionally nested AV) latent."""
        if self._is_nested(chunk.get("samples")):
            tensors = self._latent_tensors(chunk)
            masks = self._latent_masks(chunk, len(tensors))
            video_in = {"samples": tensors[0]}
            if masks[0] is not None:
                video_in["noise_mask"] = masks[0]
            video_out = self._invoke_img2video_inplace(vae, image, video_in, strength)
            if video_out is None:
                return chunk
            result = self._clone_latent(chunk)
            out_tensors = list(tensors)
            out_masks = list(masks)
            out_tensors[0] = video_out["samples"]
            if "noise_mask" in video_out:
                out_masks[0] = video_out["noise_mask"]
            result["samples"] = comfy.nested_tensor.NestedTensor(out_tensors)
            if any(m is not None for m in out_masks):
                result["noise_mask"] = comfy.nested_tensor.NestedTensor(out_masks)
            return result
        out = self._invoke_img2video_inplace(vae, image, chunk, strength)
        return out if out is not None else chunk

    def _build_mixed_anchor_chunk(self, vae, anchor_meta, latent_template, previous, video_overlap,
                                  carry_overlap=False):
        """Mixed source: Img2Video anchor latent. By default prior scene overlap is not used
        (hard cut). When carry_overlap is set and a previous scene exists, the chunk is first
        seeded as a normal continuation (previous scene's tail copied into the leading
        video_overlap frames, protected) and the anchor image is then written on top — since
        LTXVImgToVideoInplace only overwrites the single encoded image frame's worth of samples
        and preserves the rest of the incoming noise_mask, the anchor's own leading frame stays a
        hard cut while the remaining carried frames keep the old scene's background/environment.

        On MiniMax H3 the latent is left alone: the anchor is applied as a keyframe pin on the
        conditioning instead (_apply_h3_anchor), which is the only image conditioning that model
        was trained on. The carried-overlap behaviour is identical either way, so a mixed scene
        keeps its prior-scene context on both families."""
        filename = (anchor_meta or {}).get("filename")
        strength = float((anchor_meta or {}).get("strength", 1.0))
        image = self._load_image_tensor(filename) if filename else None
        if image is None:
            if previous is None:
                return self._clone_latent(latent_template)
            return self._build_continuation_chunk(latent_template, previous, 0)
        if carry_overlap and previous is not None and video_overlap > 0:
            base = self._build_continuation_chunk(latent_template, previous, video_overlap)
        else:
            base = self._clone_latent(latent_template)
        if self._is_h3:
            return base
        return self._apply_img2video_to_video_latent(vae, image, base, strength)

    def _identity_pin_filename(self, guide_list, scene_media_by_ref, identity_transfer_enabled):
        """First identity_pin-tagged image guide's filename, or None. Mirrors the matching
        inside _apply_configured_guides, but callable standalone for branches (like the mixed
        i2v anchor path) that don't run the rest of the guide stack against their chunk."""
        if not identity_transfer_enabled:
            return None
        for g in guide_list or []:
            if not g or not g.get("enabled", True):
                continue
            if str(g.get("source", "template")) == "image" and g.get("identity_pin"):
                ref = g.get("media_ref")
                fn = (scene_media_by_ref or {}).get(ref) if ref else None
                if fn:
                    return fn
        return None

    def _encode_image_guide_frame(self, filename, vae, ref_tensor):
        import os
        try:
            import folder_paths
            import numpy as np
            from PIL import Image
        except ImportError as _e:
            _log.failed("FunPackSceneChain", "guide image load (PIL/numpy)", _e,
                        "the scene renders with NO image guide")
            return None
        path = os.path.join(folder_paths.get_input_directory(), filename)
        if not os.path.isfile(path):
            return None
        try:
            img = Image.open(path).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            pixels = torch.from_numpy(arr)[None,]
            if ref_tensor is not None:
                _, _, _, rh, rw = ref_tensor.shape
                if pixels.shape[1] != rh or pixels.shape[2] != rw:
                    pixels = pixels.movedim(-1, 1)
                    pixels = comfy.utils.common_upscale(pixels, rw, rh, "bilinear", "center")
                    pixels = pixels.movedim(1, -1)
            pixels = pixels.movedim(-1, 1)
            if pixels.ndim == 4:
                pixels = pixels.unsqueeze(2)
            encoded = vae.encode(pixels.to(ref_tensor.device if ref_tensor is not None else pixels.device))
            if isinstance(encoded, dict):
                samples = encoded.get("samples")
            else:
                samples = encoded
            if samples is None:
                return None
            if samples.dim() == 5:
                return samples[:, :, :1]
            return samples
        except Exception:
            return None

    def _append_media_guide_at(self, chunk, filename, frame_idx, apply_at, strength,
                               positive, negative, vae):
        chunk_tensors = self._latent_tensors(chunk)
        if not chunk_tensors:
            return chunk, positive, negative, 0, 0
        guide_frame = self._encode_image_guide_frame(filename, vae, chunk_tensors[0])
        if guide_frame is None:
            return chunk, positive, negative, 0, 0
        total = self._tensor_frames(guide_frame)
        at = self._resolve_frame_index(total, frame_idx)
        guide_slice = self._time_slice(guide_frame, at, at + 1)
        return self._append_guide_latent(
            chunk, guide_slice, apply_at, strength, positive, negative, vae,
        ) + (0,)

    def _apply_configured_guides(self, chunk, scene_index, guide_list, latent_template,
                                 scene_outputs, scene_media_by_ref, positive, negative, vae,
                                 identity_transfer_enabled=False):
        head_crop = 0
        tail_crop = 0
        identity_ref_filename = None
        for g in guide_list or []:
            if not g or not g.get("enabled", True):
                continue
            source = str(g.get("source", "template"))
            frame_idx = int(g.get("frame_idx", 0))
            apply_at = int(g.get("apply_at", 0))
            strength = float(g.get("strength", 0.35))
            if source == "template":
                chunk, positive, negative, head, tail = self._append_template_guide_at(
                    chunk, latent_template, frame_idx, apply_at, strength, positive, negative, vae,
                )
            elif source == "scene":
                si = g.get("scene_index")
                if si is None and g.get("scene_id"):
                    si = scene_index - 1
                try:
                    si = int(si)
                except (TypeError, ValueError):
                    si = scene_index - 1
                if si < 0 or si >= len(scene_outputs) or scene_outputs[si] is None:
                    continue
                chunk, positive, negative, head, tail = self._append_scene_guide_at(
                    chunk, scene_outputs[si], frame_idx, apply_at, strength, positive, negative, vae,
                )
            elif source == "image":
                ref = g.get("media_ref")
                fn = (scene_media_by_ref or {}).get(ref) if ref else None
                if not fn:
                    continue
                if identity_transfer_enabled and g.get("identity_pin"):
                    # Best-FaceID full port takes this entry over entirely: the LoRA was
                    # trained on separate overlap tokens + ArcFace projector tokens, not a
                    # blended frame-0 keyframe, so skip the plain guide-attention append and
                    # hand the reference filename back for _install_identity_overlap.
                    identity_ref_filename = fn
                    continue
                chunk, positive, negative, head, tail = self._append_media_guide_at(
                    chunk, fn, frame_idx, apply_at, strength, positive, negative, vae,
                )
            else:
                continue
            head_crop += head
            tail_crop += tail
        return chunk, positive, negative, head_crop, tail_crop, identity_ref_filename

    def _append_mid_scene_guide(self, chunk, previous_output, positive, negative, vae, strength):
        """Append the middle frame of the previous scene as a guide for the current chunk
        using LTX's guide attention mechanism (keyframe_idxs + guide_attention_entries).
        Audio-safe: appends only to the video tensor, guide tokens influence denoising
        through attention weights rather than overwriting hidden states."""
        prev_tensors = self._latent_tensors(previous_output)
        chunk_tensors = self._latent_tensors(chunk)
        if not prev_tensors or not chunk_tensors:
            return chunk, positive, negative, 0

        # Middle frame of previous scene as guide source
        F_prev = self._tensor_frames(prev_tensors[0])
        guide_frame = self._time_slice(prev_tensors[0], F_prev // 2, F_prev // 2 + 1)
        guide_frame = guide_frame.to(device=chunk_tensors[0].device, dtype=chunk_tensors[0].dtype)

        if self._is_h3:
            # A mid-clip pin is not something H3's packed layout can place, and the LTX path
            # below would append a latent frame off its grid rather than refuse. Routed
            # through the keyframe path so it declines by the same rule, and says so.
            positive, negative, tail = self._append_h3_keyframe(
                guide_frame, max(1, int(self._h3_frame_count)) // 2, strength,
                positive, negative)
            return chunk, positive, negative, tail

        # LTX from here down. Imported after the H3 branch so the H3 path does not depend on
        # an LTX-only module importing.
        try:
            from comfy_extras.nodes_lt import LTXVAddGuide, _append_guide_attention_entry
        except ImportError as _e:
            _log.failed("FunPackSceneChain", "mid-scene guide (ComfyUI LTX guide API)", _e,
                        "the scene renders with NO mid-scene guide")
            return chunk, positive, negative, 0

        # Target temporal position: middle of current chunk in pixel space
        F_chunk = self._tensor_frames(chunk_tensors[0])
        mid_chunk = F_chunk // 2
        scale_factors = getattr(vae, 'downscale_index_formula', [8, 8, 8])
        time_scale = int(scale_factors[0]) if hasattr(scale_factors, '__getitem__') else 8
        causal_fix = (mid_chunk == 0)
        pixel_frame_idx = 0 if causal_fix else 1 + (mid_chunk - 1) * time_scale

        # Add keyframe positional indices to conditioning
        positive = LTXVAddGuide.add_keyframe_index(
            positive, pixel_frame_idx, guide_frame, scale_factors, causal_fix=causal_fix
        )
        negative = LTXVAddGuide.add_keyframe_index(
            negative, pixel_frame_idx, guide_frame, scale_factors, causal_fix=causal_fix
        )

        # Register guide attention entry (strength controls how strongly noisy frames
        # attend to the guide tokens vs. ignoring them)
        guide_latent_shape = [guide_frame.shape[2], guide_frame.shape[3], guide_frame.shape[4]]
        pre_filter_count = guide_frame.shape[2] * guide_frame.shape[3] * guide_frame.shape[4]
        positive, negative = _append_guide_attention_entry(
            positive, negative, pre_filter_count, guide_latent_shape, strength=float(strength)
        )

        # Append guide frame to video tensor (mask = 1-strength, partially pinned)
        result = self._clone_latent(chunk)
        tensors = self._latent_tensors(result)
        masks = self._latent_masks(result, len(tensors))
        if masks[0] is None:
            masks[0] = torch.ones(
                tensors[0].shape[0], 1, tensors[0].shape[2], 1, 1,
                dtype=torch.float32, device=tensors[0].device,
            )
        guide_mask = torch.full_like(masks[0][:, :, :1], max(0.0, 1.0 - float(strength)))
        tensors[0] = torch.cat([tensors[0], guide_frame], dim=2)
        masks[0] = torch.cat([masks[0], guide_mask.to(masks[0].device, masks[0].dtype)], dim=2)

        if self._is_nested(result.get("samples")):
            result["samples"] = comfy.nested_tensor.NestedTensor(tensors)
            result["noise_mask"] = comfy.nested_tensor.NestedTensor(masks)
        else:
            result["samples"] = tensors[0]
            result["noise_mask"] = masks[0]

        return result, positive, negative, 1

    def _harvest_joyai_frame(self, sampled, select):
        """Pick one clean latent frame from a finished scene for the memory bank.
        'center' (default) / 'first' / 'random', mirroring JoyAI's frame_selection_mode."""
        tensors = self._latent_tensors(sampled)
        if not tensors:
            return None
        F = self._tensor_frames(tensors[0])
        if F <= 0:
            return None
        if select == "first":
            idx = 0
        elif select == "random":
            idx = int(torch.randint(0, F, (1,)).item())
        else:
            idx = F // 2
        return self._time_slice(tensors[0], idx, idx + 1).detach()

    def _append_joyai_memory_guides(self, chunk, frames, positive, negative, vae, strength):
        """JoyAI-Echo memory injection: attach every banked prior-shot frame to the current chunk
        as a clean LTX guide, so attention carries identity/scene forward across the whole chain.
        The single-anchor mid_scene_guide generalized to N frames — reuses _append_guide_latent
        per entry. Memory frames are placed at distinct early (prefix) positions, matching JoyAI's
        sequence-prefix concat. Audio-safe (guide tokens influence via attention, not overwrite).
        Returns the total appended-frame count so the caller crops them off the output tail."""
        if not frames:
            return chunk, positive, negative, 0
        chunk_tensors = self._latent_tensors(chunk)
        if not chunk_tensors:
            return chunk, positive, negative, 0
        F_chunk = self._tensor_frames(chunk_tensors[0])
        s = max(0.0, float(strength))  # the user's value stands; 0.25-0.35 is the audio-safe band
        tail = 0
        for i, gf in enumerate(frames):
            apply_at = min(max(0, F_chunk - 1), i)  # prefix context: distinct early positions
            chunk, positive, negative, t = self._append_guide_latent(
                chunk, gf, apply_at, s, positive, negative, vae,
            )
            tail += t
        return chunk, positive, negative, tail

    def _harvest_joyai_audio(self, sampled, select):
        """Paired clean AUDIO latent frame from a finished scene, position-matched to the video
        harvest (same center/first/random fraction). None for single-stream LTXV (no audio tensor)."""
        tensors = self._latent_tensors(sampled)
        if len(tensors) < 2 or tensors[1] is None:
            return None
        F = self._tensor_frames(tensors[1], stream=1)
        if F <= 0:
            return None
        if select == "first":
            idx = 0
        elif select == "random":
            idx = int(torch.randint(0, F, (1,)).item())
        else:
            idx = F // 2
        return self._time_slice(tensors[1], idx, idx + 1, stream=1).detach()

    def _append_joyai_audio_memory(self, chunk, audio_frames):
        """Pin every banked prior-shot AUDIO latent frame into the current chunk's audio stream as a
        fully protected (mask=0) tail, the audio analogue of the video memory bank. There is no LTX
        guide-attention API for audio, so memory rides the model's native audio self-attention + the
        video->audio coupling instead. Returns the appended audio-frame count so the caller crops the
        tail back off. No-op on single-stream LTXV (no audio tensor)."""
        frames = [a for a in (audio_frames or []) if a is not None]
        if not frames:
            return chunk, 0
        tensors = self._latent_tensors(chunk)
        if len(tensors) < 2 or tensors[1] is None:
            return chunk, 0
        result = self._clone_latent(chunk)
        rtensors = self._latent_tensors(result)
        masks = self._latent_masks(result, len(rtensors))
        for i in range(len(rtensors)):
            if masks[i] is None:
                masks[i] = torch.ones_like(rtensors[i])
        audio = rtensors[1]
        amask = masks[1]
        adim = self._stream_dim(1) % audio.dim()
        appended = 0
        for af in frames:
            af = af.to(device=audio.device, dtype=audio.dtype)
            # Only append shape-compatible frames: every axis except the stream's own time
            # axis must match (on H3 that axis is the last one, not dim 2).
            if [s for d, s in enumerate(af.shape) if d != adim] != \
                    [s for d, s in enumerate(audio.shape) if d != adim]:
                continue
            clean_shape = list(amask.shape)
            clean_shape[adim] = int(af.shape[adim])
            clean = torch.zeros(clean_shape, device=amask.device, dtype=amask.dtype)
            audio = torch.cat([audio, af], dim=adim)
            amask = torch.cat([amask, clean], dim=adim)
            appended += int(af.shape[adim])
        if appended == 0:
            return chunk, 0
        rtensors[1] = audio
        masks[1] = amask
        if self._is_nested(result.get("samples")):
            result["samples"] = comfy.nested_tensor.NestedTensor(rtensors)
            result["noise_mask"] = comfy.nested_tensor.NestedTensor(masks)
        else:
            return chunk, 0  # single-stream has no audio tensor to pin
        return result, appended

    def _install_v2a_scale(self, model, scale):
        """Scale the model's trained video->audio cross-attention by `scale` (JoyAI's v2a_grad_scale)
        via reversible PyTorch forward hooks on each AV block's `video_to_audio_attn` submodule
        (out *= scale). Returns hook handles to remove afterwards. Empty when scale==1.0 (native, no
        hook installed -> zero overhead / byte-identical) or when the model isn't LTXAV."""
        if scale is None or abs(float(scale) - 1.0) < 1e-6:
            return []
        try:
            blocks = model.model.diffusion_model.transformer_blocks
        except Exception:
            return []
        if not blocks:
            return []
        s = float(scale)

        def _hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return (output[0] * s,) + tuple(output[1:])
            return output * s

        _tag_funpack_hook(_hook)
        handles = []
        for blk in blocks:
            sub = getattr(blk, "video_to_audio_attn", None)
            if sub is not None:
                handles.append(sub.register_forward_hook(_hook))
        return handles

    def _remove_v2a_scale(self, handles):
        for h in handles or []:
            try:
                h.remove()
            except Exception:
                pass

    # Wrapper keys used by context windows; core registers them under these names.
    _CTX_WRAPPER_KEYS = ("ContextWindows_prepare_sampling", "ContextWindows_sampler_sample")

    # Core's schedule identifiers put the words the other way round (comfy.context_windows
    # ContextSchedules: "standard_uniform", "standard_static", "looped_uniform", "batched").
    # This knob shipped with the readable-but-wrong spellings, so every schedule except
    # "batched" raised ValueError out of get_matching_context_schedule and failed the whole
    # render. The knob now offers core's own names; these aliases keep a project saved with
    # the old ones working instead of failing at generation time.
    _CTX_SCHEDULE_ALIASES = {
        "uniform_standard": "standard_uniform",
        "static_standard": "standard_static",
        "uniform_looped": "looped_uniform",
    }

    def _install_context_windows(self, model, length, overlap, schedule, fuse,
                                 freenoise, retain_first):
        """Hand this scene's denoise to ComfyUI core's context-window sampler.

        Nothing is ported: core owns the whole mechanism (comfy.context_windows), we only
        build its handler and hang it on this scene's model_options, the same way the stock
        LTXVContextWindows node does. Core picks it up inside comfy.samplers.calc_cond_batch,
        which every FunPack sampler already goes through (they all call model(x, sigma, ...)
        on the CFGGuider rather than apply_model directly), so no sampler change is needed.

        Returns (remove_fn, latent_len, None), or (None, None, reason) when it cannot run.
        Everything that can go wrong here is a mismatch with the installed core, and every
        one of them used to surface as either the wrong explanation or a failed render, so
        the reason is carried back and reported per scene rather than inferred.

        The capability gate below is the important part: the handler class itself
        has existed for a long time for unpacked video, but the packed AV latent needs
        BaseModel.map_context_window_to_modalities (+ resize_cond_for_context_window) to
        unpack the AV stream, map each video window onto its audio window and re-slice
        keyframe_idxs / guide_attention_entries. Without those, enabling this on LTXAV would
        window the packed tensor blindly and quietly wreck audio and guides — the same class
        of packed-vs-unpacked mistake that killed DynamicConditioning. Refuse instead.
        """
        if self._is_h3:
            # Checked BEFORE the capability gate below, which would otherwise report a
            # missing LTXAV core feature and blame the ComfyUI version for what is really a
            # model difference. Core's windowing unpacks LTXAV's packed AV stream and
            # re-slices its guide entries; H3's sequence is packed by a different layout
            # entirely, and the latent window length here is computed on LTX's 8x temporal
            # ratio besides.
            return None, None, ("MiniMax H3 has no context-window support — core's windowing "
                                "unpacks the LTXAV stream specifically, and the window length "
                                "is measured on LTX's 8x latent ratio")
        try:
            import comfy.context_windows as _cw
            import comfy.patcher_extension as _pe
        except ImportError:
            return None, None, ("this ComfyUI build has no context-window support at all "
                                "(needs ComfyUI >= v0.29.0)")
        base = getattr(model, "model", None)
        if not hasattr(base, "map_context_window_to_modalities"):
            return None, None, ("this ComfyUI build has no LTXAV context-window support "
                                "(needs the core context-windows change, ComfyUI >= v0.29.0)")
        # Real frames -> latent frames, exactly as core's LTXVContextWindows node does it.
        latent_len = max(((int(length) - 1) // 8) + 1, 1)
        latent_overlap = max(int(overlap) // 8, 0)
        retain = "0" if retain_first else ""
        # Resolve the names against CORE's vocabulary, not ours, and say what it accepts —
        # these raise ValueError, which used to escape and fail the render outright.
        _sched_name = self._CTX_SCHEDULE_ALIASES.get(str(schedule), str(schedule))
        try:
            _schedule = _cw.get_matching_context_schedule(_sched_name)
        except ValueError:
            _known = ", ".join(sorted(getattr(_cw, "CONTEXT_MAPPING", {}) or {}))
            return None, None, (f"context_window_schedule={schedule!r} is not a schedule this "
                                f"ComfyUI knows{f' — it accepts {_known}' if _known else ''}")
        try:
            _fuse = _cw.get_matching_fuse_method(str(fuse))
        except ValueError:
            _known = ", ".join(sorted(getattr(_cw, "FUSE_MAPPING", {}) or {}))
            return None, None, (f"context_window_fuse={fuse!r} is not a blend this ComfyUI "
                                f"knows{f' — it accepts {_known}' if _known else ''}")
        kwargs = dict(
            context_schedule=_schedule,
            fuse_method=_fuse,
            context_length=latent_len,
            context_overlap=latent_overlap,
            context_stride=1,
            closed_loop=False,
            dim=2,
            freenoise=bool(freenoise),
            cond_retain_index_list=retain,
            latent_retain_index_list=retain,
            split_conds_to_windows=False,
        )
        # Core's handler has gained and lost keywords over time (latent_retain_index_list is
        # absent on some builds). Passing one it doesn't take raised TypeError and the whole
        # feature was then reported as "core too old", which is a different and misleading
        # thing. Drop the extras instead, and name the one knob that stops working.
        _dropped = []
        try:
            import inspect as _inspect
            _accepted = set(_inspect.signature(_cw.IndexListContextHandler).parameters)
            if "kwargs" not in _accepted:
                _dropped = [k for k in kwargs if k not in _accepted]
                for k in _dropped:
                    kwargs.pop(k)
        except (TypeError, ValueError):
            pass
        try:
            handler = _cw.IndexListContextHandler(**kwargs)
        except TypeError as exc:
            return None, None, (f"this ComfyUI's context-window handler takes different "
                                f"arguments than expected ({exc})")
        prev = model.model_options.get("context_handler")
        had_prev = "context_handler" in model.model_options
        model.model_options["context_handler"] = handler
        _cw.create_prepare_sampling_wrapper(model)
        if freenoise:
            _cw.create_sampler_sample_wrapper(model)

        def _remove():
            if had_prev:
                model.model_options["context_handler"] = prev
            else:
                model.model_options.pop("context_handler", None)
            for wrapper_type, key in (
                (_pe.WrappersMP.PREPARE_SAMPLING, self._CTX_WRAPPER_KEYS[0]),
                (_pe.WrappersMP.SAMPLER_SAMPLE, self._CTX_WRAPPER_KEYS[1]),
            ):
                try:
                    model.remove_wrappers_with_key(wrapper_type, key)
                except Exception:
                    pass

        _note = None
        if retain_first and "latent_retain_index_list" in _dropped:
            _note = ("this ComfyUI's context-window handler has no latent_retain_index_list, "
                     "so 'pin anchor in every window' applies to conditioning only")
        return _remove, latent_len, _note

    def _build_plateau_cache_wrapper(self, model, threshold):
        """Plateau step-cache (MixCache/Chorus-family, adapted to LTX2.3's distilled schedule).

        Installed INNERMOST so it caches the raw base-model forward; every other per-scene
        wrapper (embed/score/dynashift/output guidance, temporal) still layers around it and
        post-processes the (cached-or-fresh) prediction. On the near-pure-noise plateau
        (sigma >= threshold) the transformer output barely changes step-to-step, so we compute
        it once per distinct batch signature and reuse it for the remaining plateau steps,
        skipping those full 48-block forwards. Below the threshold every step is computed for
        real, and the cache is dropped (frees the held tensors) so structure formation is never
        approximated. Deterministic given seed — no effect on batch-variant diversity.

        Cache is keyed by (input.shape, cond_or_uncond) so a CFG>1 split cond/uncond pair each
        gets its own slot instead of thrashing a single slot. Returns a stats dict the caller
        reads after the scene for the run report (never used for control flow)."""
        old_wrapper = model.model_options.get("model_function_wrapper")
        thr = float(threshold)
        stats = {"cache": {}, "reused": 0, "computed": 0}

        def _plateau_wrapper(apply_fn, args, _ew=old_wrapper, _st=stats, _thr=thr):
            def _run():
                if _ew is not None:
                    return _ew(apply_fn, args)
                return apply_fn(args["input"], args["timestep"], **(args.get("c") or {}))

            ts = args.get("timestep")
            x = args.get("input")
            try:
                sigma = float(ts.max().item()) if ts is not None else 0.0
                key = (tuple(x.shape), tuple(args.get("cond_or_uncond") or ()))
            except Exception:
                return _run()  # can't reason about it safely -> always compute

            if sigma < _thr:
                _st["cache"].clear()  # past the plateau: compute every step, release cached tensors
                return _run()
            cached = _st["cache"].get(key)
            if cached is not None:
                _st["reused"] += 1
                return cached
            out = _run()
            _st["cache"][key] = out
            _st["computed"] += 1
            return out

        model.model_options["model_function_wrapper"] = _tag_scene_wrapper(_plateau_wrapper, old_wrapper)
        return stats

    # ---------------------------------------------------------------------------
    # identity_transfer: native port of ComfyUI-BFSNodes' LTX Identity Transfer
    # (LTXIdentityOverlapConditioning) — see identity_transfer.py for the ArcFace/projector/
    # RoPE-rotation pieces. The reference face is injected as SEPARATE tokens appended after
    # the target's video tokens (an "overlap" placement, sharing the frame-0 pixel-coord grid),
    # given a clean (timestep 0) denoise state, tagged with a source-phase RoPE rotation, and
    # trimmed back off before unpatchify — never blended into a real rendered frame, unlike
    # FunPack's guide-attention path. Optional ArcFace IdentityProjector tokens are appended to
    # the text context on top.
    #
    # Four bound methods are wrapped on model.model.diffusion_model, matching LTXBaseModel's own
    # _forward pipeline: _process_input (append ref tokens) -> _prepare_timestep (clean timestep
    # for ref tokens) -> _prepare_positional_embeddings (source-phase RoPE rotation) ->
    # _process_output (trim ref tokens before unpatchify).
    # Idempotent + tag/strip per scene (same discipline as _install_v2a_scale /
    # _strip_funpack_scene_wrappers) so an interrupt mid-scene can never leave a stale patch
    # steering a later, unrelated run — see [[project_hook_leak_bug]].
    # ---------------------------------------------------------------------------
    _IDENTITY_OVERLAP_TAG = "_funpack_identity_overlap_patched"

    def _install_identity_overlap(self, model, ref_latent, seg_value):
        try:
            from .identity_transfer import rotate_overlap_freqs as _rotate_overlap_freqs
        except ImportError:
            from identity_transfer import rotate_overlap_freqs as _rotate_overlap_freqs
        try:
            ltxv = model.model.diffusion_model
        except Exception as _e:
            _log.failed("FunPackSceneChain", "identity transfer install", _e,
                        "identity transfer is INERT for this run — no overlap tokens injected")
            return None
        if getattr(ltxv, self._IDENTITY_OVERLAP_TAG, False):
            return None  # already installed (idempotent across scenes/runs)

        orig_process_input = ltxv._process_input
        orig_prepare_timestep = ltxv._prepare_timestep
        orig_prepare_pe = ltxv._prepare_positional_embeddings
        orig_process_output = ltxv._process_output
        # These wrappers reach into LTXBaseModel internals, so a ComfyUI refactor can break
        # one of them. Falling back silently is how a broken source-phase tag once passed as
        # "Best-FaceID makes the clip open on the reference image" — report each failure once
        # per install (not per step) so the cause is visible in the console immediately.
        _warned: set = set()

        def _warn_once(what, exc):
            if what in _warned:
                return
            _warned.add(what)
            print(f"[FunPackSceneChain] identity_transfer: {what} failed ({type(exc).__name__}: {exc}) "
                  f"— identity conditioning is NOT being applied correctly this run. "
                  f"This usually means a ComfyUI update changed the LTX model internals.")

        def process_input(self_ltxv, x, keyframe_idxs, denoise_mask, **kw):
            out = orig_process_input(x, keyframe_idxs, denoise_mask, **kw)
            self_ltxv._funpack_id_ref_len = 0
            try:
                from comfy.ldm.lightricks.model import latent_to_pixel_coords
                xx, pix, add = out
                is_av = isinstance(xx, (list, tuple))
                vx = xx[0] if is_av else xx
                vco = pix[0] if is_av else pix
                rt, rlc = self_ltxv.patchifier.patchify(ref_latent.to(dtype=vx.dtype, device=vx.device))
                rpc = latent_to_pixel_coords(latent_coords=rlc, scale_factors=self_ltxv.vae_scale_factors,
                                             causal_fix=self_ltxv.causal_temporal_positioning)
                rt = self_ltxv.patchify_proj(rt)
                if rt.shape[0] != vx.shape[0]:
                    rt = rt.expand(vx.shape[0], -1, -1)
                if rpc.shape[0] != vco.shape[0]:
                    rpc = rpc.expand(vco.shape[0], *([-1] * (rpc.dim() - 1)))
                ref_len = rt.shape[1]
                self_ltxv._funpack_id_target_len = vx.shape[1]
                vx = torch.cat([vx, rt], dim=1)
                vco = torch.cat([vco, rpc.to(vco)], dim=2)
                self_ltxv._funpack_id_ref_len = ref_len
                if is_av:
                    xx = [vx, xx[1]]; pix = [vco, pix[1]]
                else:
                    xx, pix = vx, vco
                return xx, pix, add
            except Exception as e:
                _warn_once("reference-token append (_process_input)", e)
                self_ltxv._funpack_id_ref_len = 0
                return out

        def prepare_timestep(self_ltxv, timestep, batch_size, hidden_dtype, **kwargs):
            ref_len = getattr(self_ltxv, "_funpack_id_ref_len", 0)
            if ref_len:
                target_len = getattr(self_ltxv, "_funpack_id_target_len", None)
                if timestep.dim() <= 1 and target_len is not None:
                    timestep = timestep.view(-1, 1).expand(batch_size, target_len).contiguous()
                if timestep.dim() >= 2:
                    ref_ts = torch.zeros(batch_size, ref_len, *timestep.shape[2:],
                                         device=timestep.device, dtype=timestep.dtype)
                    timestep = torch.cat([timestep, ref_ts], dim=1)
            return orig_prepare_timestep(timestep, batch_size, hidden_dtype, **kwargs)

        def prepare_pe(self_ltxv, pixel_coords, frame_rate, x_dtype):
            pe = orig_prepare_pe(pixel_coords, frame_rate, x_dtype)
            ref_len = getattr(self_ltxv, "_funpack_id_ref_len", 0)
            if not ref_len or not seg_value:
                return pe
            try:
                # AV models return [(v_pe, av_cross_video), (a_pe, av_cross_audio)]; plain
                # video models return v_pe directly. Only the video PE carries the ref tokens.
                if isinstance(pe, list) and len(pe) and isinstance(pe[0], (list, tuple)) and isinstance(pe[0][0], (list, tuple)):
                    v_pe, cross_v = pe[0][0], pe[0][1]
                    v_pe = _rotate_overlap_freqs(v_pe, ref_len, seg_value)
                    return [(v_pe, cross_v), pe[1]]
                return _rotate_overlap_freqs(pe, ref_len, seg_value)
            except Exception as e:
                # Untagged reference tokens share the target's frame-0 grid, so the model
                # renders them AS frame 0 — the clip opens on the reference image.
                _warn_once("source-phase RoPE tag (_prepare_positional_embeddings)", e)
                return pe

        def process_output(self_ltxv, x, embedded_timestep, keyframe_idxs, **kw):
            ref_len = getattr(self_ltxv, "_funpack_id_ref_len", 0)
            if ref_len:
                try:
                    from comfy.ldm.lightricks.av_model import CompressedTimestep
                    import copy
                    if isinstance(x, (list, tuple)):
                        x = [x[0][:, :x[0].shape[1] - ref_len], *x[1:]]
                        et_list = list(embedded_timestep) if isinstance(embedded_timestep, (list, tuple)) else [embedded_timestep]
                        v_et = et_list[0]
                        if isinstance(v_et, CompressedTimestep):
                            ppf = max(1, getattr(v_et, "patches_per_frame", 1) or 1)
                            n_ref_frames = max(1, ref_len // ppf)
                            v_et2 = copy.copy(v_et)
                            v_et2.data = v_et.data[:, : v_et.num_frames - n_ref_frames].contiguous()
                            v_et2.num_frames = v_et.num_frames - n_ref_frames
                            et_list[0] = v_et2
                        elif hasattr(v_et, "shape") and v_et.dim() >= 2 and v_et.shape[1] > 1:
                            et_list[0] = v_et[:, : v_et.shape[1] - ref_len]
                        embedded_timestep = et_list
                    else:
                        x = x[:, :x.shape[1] - ref_len]
                        if hasattr(embedded_timestep, "shape") and embedded_timestep.dim() >= 2 and embedded_timestep.shape[1] > 1:
                            embedded_timestep = embedded_timestep[:, : embedded_timestep.shape[1] - ref_len]
                except Exception as e:
                    _warn_once("reference-token trim (_process_output)", e)
            return orig_process_output(x, embedded_timestep, keyframe_idxs, **kw)

        ltxv._process_input = types.MethodType(process_input, ltxv)
        ltxv._prepare_timestep = types.MethodType(prepare_timestep, ltxv)
        ltxv._prepare_positional_embeddings = types.MethodType(prepare_pe, ltxv)
        ltxv._process_output = types.MethodType(process_output, ltxv)
        setattr(ltxv, self._IDENTITY_OVERLAP_TAG, True)
        return (ltxv, orig_process_input, orig_prepare_timestep, orig_prepare_pe, orig_process_output)

    def _strip_identity_overlap(self, handle):
        if not handle:
            return
        ltxv, orig_process_input, orig_prepare_timestep, orig_prepare_pe, orig_process_output = handle
        try:
            ltxv._process_input = orig_process_input
            ltxv._prepare_timestep = orig_prepare_timestep
            ltxv._prepare_positional_embeddings = orig_prepare_pe
            ltxv._process_output = orig_process_output
            setattr(ltxv, self._IDENTITY_OVERLAP_TAG, False)
            for attr in ("_funpack_id_ref_len", "_funpack_id_target_len"):
                if hasattr(ltxv, attr):
                    delattr(ltxv, attr)
        except Exception as _e:
            # A cleanup that fails silently is how a patch survives the run that installed it
            # and steers every later generation — the "each gen looks dirtier" failure mode.
            _log.failed("FunPackSceneChain", "identity transfer removal", _e,
                        "the patch may still be installed — RESTART ComfyUI before trusting "
                        "later generations")

    def _resolve_identity_overlap(self, state, filename, vae, chunk, identity_projector, source_id,
                                   phase_scale, id_strength, arcface_mode, debug_log):
        """Lazily encode the reference face + (optional) ArcFace projector tokens once per
        run() call and cache them in `state` (a dict owned by the caller). Cheap on every
        later scene — just returns the cached tuple. Returns
        (ref_latent_or_None, seg_value, pos_tokens_or_None, neg_tokens_or_None)."""
        if state.get("ready"):
            return state["ref_latent"], state["seg_value"], state["pos_tokens"], state["neg_tokens"]
        state.update(ready=True, ref_latent=None, seg_value=float(source_id) * float(phase_scale),
                     pos_tokens=None, neg_tokens=None)
        image = self._load_image_tensor(filename)
        if image is None:
            print(f"[FunPackSceneChain] identity_transfer: couldn't load reference image '{filename}' — skipped.")
            return None, 0.0, None, None
        chunk_tensors = self._latent_tensors(chunk)
        if not chunk_tensors:
            return None, 0.0, None, None
        try:
            import comfy.utils
            scale_factors = getattr(vae, "downscale_index_formula", [8, 8, 8])
            _, w_sf, h_sf = scale_factors
            _, _, _, lat_h, lat_w = chunk_tensors[0].shape
            ref_px = comfy.utils.common_upscale(
                image.movedim(-1, 1), lat_w * w_sf, lat_h * h_sf, "bilinear", "center",
            ).movedim(1, -1)[:1, :, :, :3]
            state["ref_latent"] = vae.encode(ref_px)
            if debug_log:
                print(f"[FunPackSceneChain] identity_transfer: overlap ref latent "
                      f"{list(state['ref_latent'].shape)}, seg={state['seg_value']}")
        except Exception as e:
            print(f"[FunPackSceneChain] identity_transfer: overlap ref-latent encode failed ({e}) — skipped.")
            return None, 0.0, None, None

        if identity_projector in (None, "", "None"):
            return state["ref_latent"], state["seg_value"], None, None
        try:
            try:
                from .identity_transfer import arcface_embed, load_identity_projector
            except ImportError:
                from identity_transfer import arcface_embed, load_identity_projector
            emb = arcface_embed(image, mode=arcface_mode)
            if emb is None:
                print(f"[FunPackSceneChain] identity_transfer: ArcFace mode={arcface_mode} found no "
                      "face — projector tokens skipped (overlap only).")
                return state["ref_latent"], state["seg_value"], None, None
            import folder_paths
            path = folder_paths.get_full_path("loras", identity_projector) or identity_projector
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            projector = load_identity_projector(path, device)
            emb = emb.to(device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                state["pos_tokens"] = projector(emb) * float(id_strength)
                state["neg_tokens"] = projector(torch.zeros(1, projector.in_dim, device=device))
            if debug_log:
                print(f"[FunPackSceneChain] identity_transfer: ArcFace projector '{identity_projector}' "
                      f"OK, id_strength={id_strength}")
        except Exception as e:
            print(f"[FunPackSceneChain] identity_transfer: ArcFace/projector failed ({e}) — overlap tokens only "
                  "(insightface installed? `pip install insightface`).")
            return state["ref_latent"], state["seg_value"], None, None
        return state["ref_latent"], state["seg_value"], state["pos_tokens"], state["neg_tokens"]

    def _append_identity_context_tokens(self, positive, negative, pos_tokens, neg_tokens):
        try:
            from .identity_transfer import append_context_tokens
        except ImportError:
            from identity_transfer import append_context_tokens
        return append_context_tokens(positive, pos_tokens), append_context_tokens(negative, neg_tokens)

    def _bounded_attention_region_mask(self, t, h, w, device):
        """[T*H*W] region id per video token: 0 = left half (by width), 1 = right half.
        Assumes (t, h, w) row-major flattening — matches the packed-latent layout
        _alg_blur_frames already relies on; LTX's patchify is 1:1 so the transformer's
        token order should match. Best-effort for an experimental feature, not guaranteed
        if that assumption ever changes upstream."""
        idx = torch.arange(t * h * w, device=device)
        w_idx = idx % w
        return (w_idx >= (w // 2)).long()

    #: what Studio tags onto entry 0 when it has learned gains for this key
    H3_GAINS_META = "funpack_h3_gains"
    #: and the learned taste direction that rides with them
    H3_TASTE_DIR_META = "funpack_h3_taste_dir"
    #: the value of each render gain that means "untouched". Not all of them are 1.0:
    #: refiner_bias is a signed push along a learned direction, so its neutral is 0.0.
    H3_GAIN_NEUTRAL = {"video": 1.0, "prompt": 1.0, "audio": 1.0,
                       "prompt_scale": 1.0, "refiner_bias": 0.0,
                       "prompt_time": 0.0, "video_detail": 1.0}

    def _h3_render_gains(self, positive):
        """The four render strengths for this run: learned from ratings, or the widgets.

        Learned is the default because rating is the only input the user wants to give — four
        scalars is a smaller search than the sigma profile already runs on ratings alone, so a
        hand-tuned value here is the user doing work the loop can do. `manual` is the explicit
        override, and it still works with no Refiner in the graph at all.

        Reading the CONDITIONING META rather than a refinement key keeps the boundary intact:
        Studio owns the key and the learning, the sampler owns the application — the same
        bridge H3 token weighting already uses.
        """
        manual = {"video": float(getattr(self, "_h3_gain_video", 1.0)),
                  "prompt": float(getattr(self, "_h3_gain_prompt", 1.0)),
                  "audio": float(getattr(self, "_h3_gain_audio", 1.0)),
                  "prompt_scale": float(getattr(self, "_h3_prompt_scale", 1.0)),
                  "refiner_bias": float(getattr(self, "_h3_taste_bias", 0.0)),
                  "prompt_time": float(getattr(self, "_h3_prompt_time", 0.0)),
                  "video_detail": float(getattr(self, "_h3_video_detail", 1.0))}
        if str(getattr(self, "_h3_gain_mode", "learned")).lower() == "manual":
            return manual
        learned = None
        try:
            entry = positive[0] if isinstance(positive, list) and positive else None
            if isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
                candidate = entry[1].get(self.H3_GAINS_META)
                if isinstance(candidate, dict):
                    learned = {k: float(v) for k, v in candidate.items()}
        except Exception:  # noqa: BLE001
            learned = None
        if learned is None:
            # No key, or nothing rated yet. Trained strengths — NOT the widgets, which in
            # this mode are not what the user is steering with.
            if any(value != self.H3_GAIN_NEUTRAL[key] for key, value in manual.items()):
                _log.feature(
                    "FunPackSceneChain", "Rating-learned render gains", False,
                    "nothing rated on this key yet, so the run uses the model's trained "
                    "strengths. The h3_gain_* widgets are IGNORED in 'learned' mode — set "
                    "h3_gain_mode to 'manual' to use them as typed.")
            return dict(self.H3_GAIN_NEUTRAL)
        return {k: learned.get(k, self.H3_GAIN_NEUTRAL[k]) for k in manual}

    def _h3_prompt_rows(self, positive):
        """Where the PROMPT starts inside the text span, from the conditioning's own tags.

        Returns (start, note). The `<Picture N>` label and a reference's vision block sit at
        the head of the span and are not ours to edit, so everything the refiner edit does is
        confined to the rows after them.
        """
        try:
            from . import h3_token_weights as _tw
        except ImportError:
            import h3_token_weights as _tw
        try:
            entry = positive[0] if isinstance(positive, list) and positive else None
            meta = entry[1] if isinstance(entry, (list, tuple)) and len(entry) >= 2 else {}
            cond = entry[0] if entry is not None else None
            cond_len = int(cond.shape[1]) if hasattr(cond, "shape") and cond.dim() >= 2 else 0
            region = _tw.prompt_region(meta.get("minimax_token_tags"), cond_len)
            if region:
                return int(region[0]), None
            if cond_len:
                return 0, ("the prompt/reference boundary could not be read from the token "
                           "tags, so the WHOLE text span is edited — including a reference's "
                           "vision block, if one is wired")
        except Exception as error:  # noqa: BLE001
            _log.failed("FunPackSceneChain", "h3 token-refiner row range", error,
                        "the whole text span is edited instead of just the prompt")
        return 0, None

    def _h3_taste_bias_vector(self, model, positive, strength):
        """The learned taste direction, projected into the refiner's own space.

        Studio learns `liked_dir` in Qwen space and already steers the conditioning along it.
        That edit then goes through condition_proj, two refiner blocks and a final RMSNorm —
        which mixes it across tokens and normalizes its magnitude away, so the direction that
        reaches the 50 DiT blocks is not the one that was learned. Adding it AFTER the
        refiner lands it unchanged, in the space the blocks actually read.

        Added to every prompt row equally, which is what makes it transferable: Qwen does not
        pad, so the row COUNT changes with every prompt edit and a per-position bias would
        mean something different on each run.

        Returns a unit vector for `TokenRefinerEdit(bias_relative=True)` to rescale, or None.
        """
        if not strength:
            return None
        try:
            from . import minimax_h3 as h3mod
        except ImportError:
            import minimax_h3 as h3mod
        direction = None
        try:
            entry = positive[0] if isinstance(positive, list) and positive else None
            if isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
                direction = entry[1].get(self.H3_TASTE_DIR_META)
        except Exception:  # noqa: BLE001
            direction = None
        if direction is None:
            _log.feature(
                "FunPackSceneChain", "Taste push", False,
                "the conditioning carries no learned direction. "
                "It needs a refinement key with at least 3 liked runs on it.")
            return None
        projected = h3mod.project_into_refiner_space(model, direction)
        if projected is None:
            _log.feature(
                "FunPackSceneChain", "Taste push", False,
                "the learned direction does not fit the token refiner's space. "
                "Its width does not match condition_proj.")
            return None
        norm = float(projected.float().norm().item())
        if not norm or norm != norm:
            return None
        return (projected.float() / norm) * float(strength)

    def _install_h3_token_refiner(self, model, positive):
        """Edit the token refiner's OUTPUT: prompt loudness, and the learned taste push.

        One patch for both. They act on the same rows through the same wrapper, so applying
        them separately would nest two wrappers and make the second one's `bias_relative`
        rescale read a span the first had already scaled.
        """
        gains = self._h3_render_gains(positive)
        scale = float(gains.get("prompt_scale", 1.0))
        strength = float(gains.get("refiner_bias", 0.0))
        if scale == 1.0 and not strength:
            return model
        try:
            from . import minimax_h3 as h3mod
        except ImportError:
            import minimax_h3 as h3mod
        if not h3mod.is_h3_model(model):
            _log.feature(
                "FunPackSceneChain", "Token-refiner edit", False,
                "not a MiniMax H3 model. There is no token refiner to edit.")
            return model
        start, note = self._h3_prompt_rows(positive)
        if note:
            _log.feature("FunPackSceneChain", "Token-refiner edit", True, note)
        bias = self._h3_taste_bias_vector(model, positive, strength)
        if scale == 1.0 and bias is None:
            return model
        try:
            patched, applied = h3mod.apply_token_refiner_edit(
                model, scale=scale, bias=bias, row_start=start, bias_relative=bias is not None)
        except Exception as error:  # noqa: BLE001
            _log.failed("FunPackSceneChain", "h3 token-refiner edit", error,
                        "the prompt is read at its trained strength and unbiased")
            return model
        if applied:
            print(f"[FunPackSceneChain] {applied}")
        return patched

    def _install_h3_adaln_gains(self, model, positive):
        """Scale every DiT block's AdaLN gates per modality, from the three sampler widgets.

        Deliberately NOT part of the refinement path. This is a visual-behaviour op, so it
        lives on the sampler (Studio produces conditioning; the sampler decides how the model
        renders it) and it reads only its own widgets. Nothing here consults a refinement key,
        a rating, or a learned direction — turning conditioning steering off does not turn
        this off, and turning this on does not require the Refiner to be wired at all.

        Returns `model` untouched when all three gains are 1.0, which is every default run.
        """
        gains = self._h3_render_gains(positive)
        prompt_time = float(gains.get("prompt_time", 0.0))
        if all(gains.get(k, 1.0) == 1.0 for k in ("video", "prompt", "audio")) \
                and prompt_time <= 0.0:
            return model
        try:
            from . import minimax_h3 as h3mod
        except ImportError:
            import minimax_h3 as h3mod
        if not h3mod.is_h3_model(model):
            _log.feature(
                "FunPackSceneChain", "AdaLN render gains", False,
                "not a MiniMax H3 model. The per-modality gates only exist there.")
            return model
        # MODALITY_TAGS names the text modality "text"; the widget says "prompt" because that
        # is what it is to the person setting it.
        tagged = {"video": gains["video"], "text": gains["prompt"], "audio": gains["audio"]}
        try:
            patched, note = h3mod.apply_adaln_edits(model, tagged, prompt_timestep=prompt_time)
        except Exception as error:  # noqa: BLE001
            _log.failed("FunPackSceneChain", "AdaLN modality gain", error,
                        "the blocks write at their trained strength for every modality")
            return model
        if note:
            print(f"[FunPackSceneChain] {note}")
        return patched

    def _install_h3_final_layer(self, model, positive):
        """The video-only detail scale, applied past the model's last attention pass.

        Everything else that moves the picture also reaches the soundtrack, because all 50
        blocks share one attention sequence. This one cannot: there is no attention after
        the final layer, so the audio branch is untouched by construction.
        """
        gains = self._h3_render_gains(positive)
        detail = float(gains.get("video_detail", 1.0))
        if detail == 1.0:
            return model
        try:
            from . import minimax_h3 as h3mod
        except ImportError:
            import minimax_h3 as h3mod
        if not h3mod.is_h3_model(model):
            _log.feature(
                "FunPackSceneChain", "Video detail", False,
                "not a MiniMax H3 model. The final-layer edit is an H3-only lane.")
            return model
        try:
            patched, note = h3mod.apply_final_video_scale(model, detail)
        except Exception as error:  # noqa: BLE001
            _log.failed("FunPackSceneChain", "Video detail", error,
                        "the picture is rendered at the model's trained strength")
            return model
        if note:
            print(f"[FunPackSceneChain] {note}")
        return patched

    def _install_h3_token_weights(self, model, positive):
        """Apply the rating-derived phrase emphasis Studio tagged onto the conditioning.

        Returns `model` untouched when there is no tag, which is every non-H3 run and every
        H3 run before the first rating. Never raises: this is a refinement, not a
        prerequisite.

        The bias is an attention MASK, and SLA routes masked calls to dense
        (sla_attention.py) because a block-sparse kernel cannot carry a per-key bias. So a
        weighted run is a dense run — said out loud, once, rather than discovered as a
        slowdown.
        """
        try:
            meta = None
            if isinstance(positive, list) and positive and isinstance(positive[0], (list, tuple)) \
                    and len(positive[0]) >= 2 and isinstance(positive[0][1], dict):
                meta = positive[0][1].get("funpack_h3_token_weights")
            if not isinstance(meta, dict):
                return model
            spans = meta.get("spans") or []
            prompt_tokens = int(meta.get("prompt_tokens") or 0)
            if not spans or prompt_tokens <= 0:
                return model
            cond = positive[0][0]
            cond_len = int(cond.shape[1]) if hasattr(cond, "shape") and cond.dim() >= 2 else 0
            if cond_len <= 0:
                return model
            try:
                from . import h3_token_weights as _tw
            except ImportError:
                import h3_token_weights as _tw

            patched = model.clone()
            to = patched.model_options.get("transformer_options", {}).copy()
            inner = to.get("optimized_attention_override")
            base = meta.get("base")
            to["optimized_attention_override"] = _tw.make_override(
                spans, prompt_tokens, cond_len, inner=inner,
                base=int(base) if base is not None else None)
            patched.model_options["transformer_options"] = to
            strongest = max((w for _, _, w in spans), default=1.0)
            placed = "modality tags" if base is not None else "conditioning tail"
            print(f"[FunPackStudio] H3 phrase emphasis: {len(spans)} token span(s) biased in "
                  f"the packed attention stream (strongest x{strongest:.2f}, placed from the "
                  f"{placed}). This is an attention mask, so SLA runs DENSE for this "
                  f"generation.")
            return patched
        except Exception as _e:  # noqa: BLE001
            _log.failed("FunPackStudio", "H3 phrase emphasis", _e,
                        "the rating's per-phrase weighting is NOT being applied")
            return model

    def _install_bounded_attention(self, model, latent, positive):
        """EXPERIMENTAL (arXiv:2403.16990-inspired, see [[project_bounded_attention]]): mask the
        text cross-attention (attn2 only — never audio_attn2, never self-attention) so the LEFT
        half of the video frame (by width) can only attend to subject-1's prompt tokens and the
        RIGHT half only to subject-2's, per the sentence-count split Studio recorded as
        funpack_bound_split_tokens on the scene's conditioning metadata. Pure attention masking,
        no extra forward pass. Returns [] (no-op) without that metadata, a 5D video tensor, an
        existing padding mask already in play, or recognizable transformer blocks."""
        try:
            n1 = None
            if isinstance(positive, list) and positive and isinstance(positive[0], (list, tuple)) and len(positive[0]) >= 2:
                meta = positive[0][1]
                if isinstance(meta, dict):
                    n1 = meta.get("funpack_bound_split_tokens")
            if n1 is None:
                print("[FunPack AV] Bounded Attention enabled but skipped this scene — prompt "
                      "didn't split into 2+ sentences (needs subject-1/subject-2 in separate sentences).")
                return []
            n1 = int(n1)
            tensors = self._latent_tensors(latent)
            video = max(tensors, key=lambda v: v.dim())
            if video.dim() != 5:
                print("[FunPack AV] Bounded Attention enabled but skipped — couldn't read a 5D video latent.")
                return []
            _, _, t, h, w = video.shape
            if w < 2:
                print("[FunPack AV] Bounded Attention enabled but skipped — frame too narrow to split.")
                return []
            blocks = model.model.diffusion_model.transformer_blocks
        except Exception:
            print("[FunPack AV] Bounded Attention enabled but skipped — couldn't reach the model's transformer blocks.")
            return []
        if not blocks:
            print("[FunPack AV] Bounded Attention enabled but skipped — no transformer blocks found.")
            return []
        region = self._bounded_attention_region_mask(t, h, w, video.device)  # [Q], 0=left, 1=right

        def _hook(_module, args, kwargs):
            try:
                if kwargs.get("mask") is not None:
                    return args, kwargs  # real padding mask already present -> skip, don't fight it
                x = args[0] if args else None
                context = kwargs.get("context")
                if x is None or context is None:
                    return args, kwargs
                q = int(x.shape[1])
                if q != region.shape[0]:
                    return args, kwargs  # resolution mismatch -> no-op safely
                k = int(context.shape[1])
                n1c = max(0, min(k, n1))
                key_idx = torch.arange(k, device=x.device)
                allow0 = key_idx < n1c
                mask = torch.where(region.unsqueeze(1) == 0, allow0.unsqueeze(0), (~allow0).unsqueeze(0))
                kwargs = dict(kwargs)
                kwargs["mask"] = mask
                return args, kwargs
            except Exception:
                return args, kwargs

        _tag_funpack_hook(_hook)
        handles = []
        for blk in blocks:
            sub = getattr(blk, "attn2", None)
            if sub is not None:
                handles.append(sub.register_forward_pre_hook(_hook, with_kwargs=True))
        if handles:
            print(f"[FunPack AV] Bounded Attention on ({len(handles)} blocks hooked, "
                  f"split at token {n1}, frame {w}x{h}x{t}) — left half sees tokens <{n1}, "
                  f"right half sees tokens >={n1}")
        return handles

    def _remove_bounded_attention(self, handles):
        for h in handles or []:
            try:
                h.remove()
            except Exception as _e:
                # Same leak class as identity-transfer removal: a hook that outlives its run.
                _log.failed("FunPackSceneChain", "bounded attention removal", _e,
                            "an attention hook may still be installed — RESTART ComfyUI before "
                            "trusting later generations")

    @staticmethod
    def _mem_report(label):
        """RSS, system RAM and what ComfyUI is holding — one line, at a phase boundary.

        This exists because of a failure with no traceback: the box stops responding after
        sampling and has to be power-cycled. The suspect is where a video model's memory GOES
        rather than how much of it there is. ComfyUI frees VRAM by moving weights back to HOST
        RAM, and the VAE decode is the call that triggers that eviction — so the moment the
        picture is finished is exactly when tens of GB can land in system memory. On H3 the
        text encoder alone (Qwen3-VL-32B) is ~64 GB in bf16, and it is DONE by then: its
        conditioning was built before sampling started. If the host has no room for what gets
        evicted, Linux swaps, and a swapping box looks hung long before the OOM killer fires.

        Everything is best-effort. This must never be the thing that fails a run.
        """
        parts = []
        try:
            import psutil
            proc = psutil.Process()
            vm = psutil.virtual_memory()
            parts.append(f"RSS {proc.memory_info().rss / 1024 ** 3:.1f} GB")
            parts.append(f"RAM {vm.available / 1024 ** 3:.1f} GB free of "
                         f"{vm.total / 1024 ** 3:.0f}")
            swap = psutil.swap_memory()
            if swap.total:
                parts.append(f"swap {swap.used / 1024 ** 3:.1f}/{swap.total / 1024 ** 3:.0f} GB")
        except Exception:  # noqa: BLE001
            pass
        try:
            parts.append(f"VRAM {comfy.model_management.get_free_memory() / 1024 ** 3:.1f} GB free")
        except Exception:  # noqa: BLE001
            pass
        # Pinned host memory is the number that actually kills a Linux box. Page-locked pages
        # cannot be swapped or reclaimed, so once the budget is committed the kernel has
        # nothing left to give and goes into unrecoverable direct reclaim — the machine stops
        # responding rather than the process being killed. ComfyUI's default budget is up to
        # 90% of RAM (comfy/model_management.py), which leaves very little for the OS, the
        # page cache, ffmpeg and the decode itself. Launch with --disable-pinned-memory when
        # this line shows the budget close to total RAM.
        try:
            mm = comfy.model_management
            cap = float(getattr(mm, "MAX_PINNED_MEMORY", -1) or -1)
            if cap > 0:
                used = float(getattr(mm, "TOTAL_PINNED_MEMORY", 0) or 0)
                note = f"pinned {used / 1024 ** 3:.0f}/{cap / 1024 ** 3:.0f} GB budget"
                try:
                    import psutil
                    if cap > psutil.virtual_memory().total * 0.80:
                        note += " — budget is >80% of RAM; pinned pages cannot be swapped, " \
                                "so this is the usual cause of a host that stops responding " \
                                "(launch with --disable-pinned-memory)"
                except Exception:  # noqa: BLE001
                    pass
                parts.append(note)
        except Exception:  # noqa: BLE001
            pass
        try:
            held = []
            for lm in comfy.model_management.current_loaded_models:
                name = type(getattr(lm.model, "model", lm.model)).__name__
                size = float(lm.model_memory()) / 1024 ** 3
                off = float(lm.model_offloaded_memory()) / 1024 ** 3
                held.append(f"{name} {size:.0f}G" + (f" ({off:.0f}G on CPU)" if off > 0.5 else ""))
            if held:
                parts.append("holding " + ", ".join(held))
        except Exception:  # noqa: BLE001
            pass
        if parts:
            print(f"[FunPackSceneChain] mem @ {label}: " + " | ".join(parts))

    def _log_decode_plan(self, vae, video_tensor):
        """One line naming what the decode needs and what is free, before it needs it.

        Every number here is best-effort: this runs at the most fragile moment of the run and
        must never be the thing that fails it.
        """
        try:
            shape = tuple(video_tensor.shape)
            frames = int(shape[2]) if len(shape) >= 5 else None
            px = None
            if frames is not None:
                # LTX/H3 latents are 8x spatial, 8x temporal (plus the origin frame).
                h, w = int(shape[-2]) * 8, int(shape[-1]) * 8
                n = (frames - 1) * 8 + 1
                px = n * h * w * 3 * 4 / (1024 ** 3)      # float32 pixels, on the CPU
            free_vram = free_ram = None
            try:
                free_vram = comfy.model_management.get_free_memory() / (1024 ** 3)
            except Exception:  # noqa: BLE001
                pass
            try:
                import psutil
                free_ram = psutil.virtual_memory().available / (1024 ** 3)
            except Exception:  # noqa: BLE001
                pass
            print(f"[FunPackSceneChain] decoding {shape} -> "
                  f"{f'~{px:.1f} GB of pixels' if px else 'pixels'} on the CPU"
                  f"{f' | {free_vram:.1f} GB VRAM free' if free_vram else ''}"
                  f"{f' | {free_ram:.1f} GB RAM free' if free_ram else ''}"
                  f"{'' if px is None or free_ram is None or px < free_ram * 0.6 else ' — this is close to the RAM available; set decode_tile_size or lower the frame count if the run dies here'}")
        except Exception:  # noqa: BLE001
            pass

    def _decode_tile_latent(self, vae, decode_tile_size):
        """`decode_tile_size` is in PIXELS; decode_tiled wants latent units.

        The divisor is the VAE's own spatial downscale (LTX 32, H3 16), not the 8 the old
        hardcode assumed — on H3 that made every tile a quarter of the requested area, which
        is slower and can trip the model's internal tiling. Falls back to 8 so a VAE that
        doesn't report a ratio behaves exactly as before.
        """
        ratio = 8
        try:
            r = getattr(vae, "downscale_ratio", None)
            if isinstance(r, (list, tuple)) and len(r) >= 2:
                ratio = int(r[1])
            elif isinstance(r, (int, float)) and r:
                ratio = int(r)
        except Exception:
            ratio = 8
        return max(1, int(decode_tile_size) // max(1, ratio))

    def _vae_with_decode_noise(self, vae, timestep, scale, seed):
        """Return a shallow copy of the VAE stamped with LTX decode-time noise settings so its
        internal decoder restores fine detail/grain. Never mutates the shared input VAE. Mirrors
        LTXV's 'Set VAE Decoder Noise', but owned by the Chain Sampler (it does the decode).

        Only the conv decoder honours this. LTX 2.5's diffusion decoder (CausalDiffusionVAE)
        takes no timestep and hard-codes its own generator seed, and because it is an nn.Module
        the two assignments below would SUCCEED and then be ignored — a knob that reads as live
        and does nothing. So the capability is tested first and reported when it is missing."""
        fsm = getattr(vae, "first_stage_model", None)
        # The consumer is VideoVAE.decode, which reads self.decode_timestep / decode_noise_scale.
        # A decoder that never had the attribute is one that never reads it — this tests the
        # actual contract rather than the class name, so it survives an upstream rename.
        if fsm is not None and not hasattr(fsm, "decode_timestep"):
            print(f"[FunPackSceneChain] decode_noise_scale={scale} / decode_timestep={timestep} "
                  f"IGNORED: this VAE's decoder ({type(fsm).__name__}) does not take decode-time "
                  f"noise — LTX 2.5's diffusion decoder generates its own detail from a fixed "
                  f"internal seed. Decoding without it. Load the conv VAE "
                  f"(ltx-2.5-video-vae-conv-*.safetensors) if you want this knob back.")
            return vae
        try:
            result = copy.copy(vae)
        except Exception:
            return vae
        if fsm is not None:
            try:
                result.first_stage_model.decode_timestep = timestep
                result.first_stage_model.decode_noise_scale = scale
            except Exception:
                pass
        result._decode_timestep = timestep
        result.decode_noise_scale = scale
        result.seed = seed
        return result

    def sample(self, model, vae, positive, negative, sampler, sigmas, seed, latent_template,
               num_frames_per_scene, frame_overlap, cfg, max_scenes, use_same_seed=False,
               carry_i2v_guides=False,
               mid_scene_guide=False, mid_scene_guide_strength=0.4,
               joyai_memory=False, joyai_memory_size=7, joyai_fix_frames=3,
               joyai_frame_select="center", joyai_memory_strength=0.3,
               joyai_audio_memory=False, v2a_grad_scale=1.0,
               embed_guidance=False, embed_guidance_strength=0.02,
               embed_guidance_source="relative",
               score_slider=False, score_slider_strength=1.0,
               transition_duration=16, decode_tile_size=0,
               decode_noise_scale=0.0, decode_timestep=0.05,
               refinement_key_input="", funpack_scene_guides="",
               funpack_scene_anchors="",
               funpack_scene_media_refs="",
               alg_blur_guides=False,
               bounded_attention_enabled=False,
               output_guidance=False, output_guidance_strength=0.02,
               dynashift=False, dynashift_strength=0.3, dynashift_threshold=0.6,
               alg_guide_blur_strength=2.0, alg_guide_blur_sigma_threshold=0.975,
               alg_anchor=False, alg_anchor_strength=2.0, alg_anchor_sigma_threshold=0.975,
               identity_transfer_enabled=False, identity_projector="None", source_id=2.0,
               phase_scale=1.0, id_strength=1.0, arcface_mode="auto_adjust", debug_log=False,
               carry_overlap_through_anchor=False,
               plateau_cache=False, plateau_cache_threshold=0.975,
               taste_nearest_prompt=False,
               segmented_detailing=False, detail_targets="hands",
               second_pass_upscale=2.0,
               detail_upsampler="None", detail_strength=1.0, detail_threshold=0.35,
               detail_max_area=0.35, detail_denoise=0.85, detail_mode="repair",
               context_windows=False, context_window_length=145, context_window_overlap=40,
               context_window_schedule="standard_uniform", context_window_fuse="pyramid",
               context_window_freenoise=True, context_window_retain_first=False,
               cut_opening_frames=0,
               second_pass=False, second_pass_op="none", second_pass_sigmas=None,
               second_pass_sampler=None,
               h3_audio_clock=False,
               h3_gain_video=1.0, h3_gain_prompt=1.0, h3_gain_audio=1.0,
               h3_prompt_scale=1.0, h3_taste_bias=0.0, h3_gain_mode="learned",
               h3_prompt_time=0.0, h3_video_detail=1.0,
               audio_vae=None, h3_keyframes=None,
               unique_id=None, prompt=None):
        if not isinstance(positive, list) or not positive:
            raise ValueError("positive conditioning must contain at least one scene entry.")
        if negative is None:
            negative = []

        # Which model family is this? Everything downstream that slices a latent stream on
        # its time axis needs to know before it touches a tensor, because LTXAV and MiniMax
        # H3 disagree about which axis that is on the AUDIO stream and both are 4-D.
        # New generation: per-run log suppression starts over, so a failure that also
        # happened last run is reported again rather than deduped away forever.
        _log.begin_run()
        self._is_h3 = self._set_stream_axes(model)
        # Read once per run, consumed by _install_h3_adaln_gains at each scene's sample call.
        self._h3_gain_video = max(0.0, min(2.0, float(h3_gain_video)))
        self._h3_gain_prompt = max(0.0, min(2.0, float(h3_gain_prompt)))
        self._h3_gain_audio = max(0.0, min(2.0, float(h3_gain_audio)))
        self._h3_prompt_scale = max(0.0, min(2.0, float(h3_prompt_scale)))
        self._h3_taste_bias = max(-0.30, min(0.30, float(h3_taste_bias)))
        self._h3_prompt_time = max(0.0, min(1.0, float(h3_prompt_time)))
        self._h3_video_detail = max(0.0, min(2.0, float(h3_video_detail)))
        self._h3_gain_mode = str(h3_gain_mode or "learned").strip().lower()
        # The gate every rating-driven wrapper shares. On H3 it is read off the schedule's
        # own base grid; on LTX it stays the absolute-sigma gate it was validated with.
        _steer_ramp = _make_steer_ramp(sigmas, self._is_h3)
        if self._is_h3:
            # H3 only generates on a 17k+5 pixel-frame grid. Empty MiniMax H3 AV Latent snaps
            # its own `length` up silently, so an off-grid scene length produces a template
            # that is LONGER than the count asked for here and the length check below fails
            # with an arithmetic complaint that never mentions the grid. Snap to the same
            # number the latent node would, so the two always agree. LTX is untouched: its
            # grid is 8k+1 and the count comes from the VAE either way.
            try:
                from .minimax_h3 import align_frame_count
            except ImportError:
                from minimax_h3 import align_frame_count
            _aligned = align_frame_count(num_frames_per_scene)
            if _aligned != int(num_frames_per_scene):
                print(f"[FunPackSceneChain] H3: num_frames_per_scene {int(num_frames_per_scene)} "
                      f"-> {_aligned} (the model's 17k+5 frame grid at 24 fps). Every scene is "
                      f"that much longer than requested.")
            num_frames_per_scene = _aligned
        self._h3_frame_count = int(num_frames_per_scene)
        # Fresh per run: the encoded references must always trace back to the media that is
        # live in THIS request (see [[feedback_no_persistent_state_caches]]).
        self._h3_ref_cache = {}
        self._h3_mode_noted = False
        self._h3_clock_unreachable_noted = False
        self._alg_unreachable_noted = False
        if self._is_h3:
            # A standing property of the model, identical on every run: stated when it
            # becomes true and again only if the family changes under you.
            _log.note_on_change(
                "chain:family", "FunPackSceneChain",
                "MiniMax H3 detected — audio stream time axis is the last dim, frame grid is "
                "17k+5, conditioning is a single packed self-attention stream "
                "(no cross-attention).")
            # Say what cannot run BEFORE sampling starts. Each of these depends on an LTX
            # transformer structure H3 does not have, so left alone they would install
            # cleanly, never fire, and be indistinguishable from "on but not helping".
            _dead = []
            if bounded_attention_enabled:
                _dead.append("bounded_attention (needs text cross-attention; H3 packs text into "
                             "the same self-attention stream, and an S x S mask over that "
                             "sequence is not affordable)")
            if identity_transfer_enabled:
                _dead.append("identity_transfer / Best-FaceID (the ArcFace projector is trained "
                             "against LTX's 4096-wide cross-attention context, and the overlap "
                             "tokens need LTX's patchifier; H3's native ref2va reference blocks "
                             "are the equivalent and are not wired yet)")
            if context_windows:
                _dead.append("context_windows (core's windowing unpacks the LTXAV stream and "
                             "re-slices its guide entries; H3 packs its sequence differently, "
                             "and the window length is measured on LTX's 8x latent ratio)")
            if alg_blur_guides:
                _dead.append("alg_blur_guides (it blurs the trailing GUIDE frames appended to "
                             "the latent; on H3 a guide is a condition row, so the appended "
                             "tail is always 0 and there is nothing to blur. alg_anchor still "
                             "works — a continuation scene does carry real latent frames)")
            if joyai_memory:
                _dead.append("joyai_memory / JoyAI-Echo (a LoRA-driven technique — the base "
                             "weights were never trained to read the injected memory frames "
                             "as memory, the LoRA teaches that, and no JoyAI-Echo LoRA exists "
                             "for H3. Inaccessible by design, not merely unwired. It would "
                             "also actively hurt output if forced on: the bank places memory "
                             "frame i at sequence position i, and writing index 0 twice "
                             "REPLACES whatever was there, evicting the scene's i2v anchor)")
            # Gated on joyai_audio_memory, not on the value alone: v2a_grad_scale is that
            # feature's coupling knob and is never installed without it (see the call site),
            # so reporting a left-over value as an H3 limitation blames the model for a knob
            # that would be equally inert on LTX.
            if (joyai_audio_memory and v2a_grad_scale is not None
                    and abs(float(v2a_grad_scale) - 1.0) > 1e-6):
                _dead.append("v2a_grad_scale (hooks LTXAV's video_to_audio_attn submodule; H3 "
                             "has no separate video->audio cross-attention to scale)")
            # Also standing: the same toggles are inert on every run of an H3 project. Keyed
            # per feature so turning one off (or switching family) is reported when it happens.
            for _line in _dead:
                _log.note_on_change(f"chain:h3dead:{_line.split(' ')[0]}", "FunPackSceneChain",
                                    f"H3: {_line} — SKIPPED.")
            # Turn them off for real rather than letting each one discover its own missing
            # LTX attribute mid-scene: several would raise rather than no-op, and a scene
            # that dies three minutes in is worse than a knob that says why it is inert.
            bounded_attention_enabled = False
            identity_transfer_enabled = False
            # Off for real, not merely hidden in the UI: unlike the two above, this one does
            # not fail to fire — it fires and overwrites the anchor pin. A stored value from
            # an LTX project must not keep doing that here.
            joyai_memory = False
            joyai_audio_memory = False
            # These two only ever no-op on H3 — off here so the run report does not list a
            # mechanism that did nothing.
            context_windows = False
            alg_blur_guides = False
            v2a_grad_scale = 1.0
            # second_pass_op is NOT switched off here. It only needs a latent upsampler that
            # matches this model's latent width; which file that is depends on what is
            # installed, not on the family, so it is checked where the upsampler is loaded.

        # Defensively strip any enhancement block hooks left on the shared diffusion
        # model by a previous run (build_enhancements only removes them on scene
        # transitions, not at end-of-sampling). This covers runs that don't go through
        # build_enhancements, so stale hooks can't fire on an unenhanced generation.
        try:
            try:
                from .ltx_enhancements import strip_funpack_block_hooks, count_module_hooks
            except ImportError:
                from ltx_enhancements import strip_funpack_block_hooks, count_module_hooks
            strip_funpack_block_hooks(model)
            # Nothing of ours is installed yet at this point, so whatever is left is either a
            # third-party hook or a leak this sweep cannot prove is ours. Printing the count
            # every run makes the difference visible: a number that climbs run after run is
            # the progressive-degradation bug, not a hunch.
            _left, _mods = count_module_hooks(model)
            if _left:
                print(f"[FunPackSceneChain] hook census before sampling: {_left} hook(s) on "
                      f"{_mods} module(s) — FunPack installs none until it samples, so a count "
                      f"that grows every run means hooks are leaking")
        except Exception as _e:
            print(f"[FunPackLTXAVSceneChainSampler] hook strip failed: {_e}")
        # Same defense for per-scene model_function_wrappers: normally unwound in the
        # per-scene finally, but a hard kill between install and restore (or a crash in a
        # third-party wrapper we chained onto) could still leave one behind in-process.
        _strip_funpack_scene_wrappers(model)

        # Blackwell (sm_120) GPUs can't run xformers attention with a tensor mask; the LTX
        # guide path uses one, so anchor scenes generate but guide scenes crash. Route masked
        # attention to SDPA when that exact combo is detected (no-op otherwise). Threaded via
        # transformer_options so it reaches every scene's model forward.
        _funpack_install_mask_safe_attention(model)

        # H3's DiT refuses a batched forward (`MiniMax H3 supports batch size 1`), and comfy
        # batches the positive and negative conds together whenever cfg != 1.0 — so without
        # this every guided H3 generation dies on step 1. Splitting the batch around the model
        # call costs exactly what CFG already costs and keeps the cfg knob live. Installed as
        # the run's BASE wrapper so every per-scene wrapper chains on top of it.
        if self._is_h3:
            try:
                from .minimax_h3 import install_batch_split
            except ImportError:
                from minimax_h3 import install_batch_split
            _h3_prev, _h3_wrapper = install_batch_split(model)
            _tag_scene_wrapper(_h3_wrapper, _h3_prev)

        # Decode-time noise (folded in from LTXV's Set VAE Decoder Noise per the boundary law:
        # the Chain Sampler owns IMAGES decode, so this lives here, not on a separate node).
        # Stamp settings onto a shallow copy of the VAE so we never mutate the shared input.
        if decode_noise_scale and decode_noise_scale > 0:
            vae = self._vae_with_decode_noise(vae, decode_timestep, decode_noise_scale, seed)

        # Batch Training: Studio (the hub) packs N conditionings into positive, each scene entry
        # tagged 'funpack_batch_variant'. That marker is the only trigger — the sampler has no
        # batch-count input. Sample one chain per packed entry, persist each for rating in Studio.
        if self._split_batch_variants(positive) is not None:
            return self._run_batch_training(
                model, vae, positive, negative, sampler, sigmas, seed, latent_template,
                num_frames_per_scene, frame_overlap, cfg, max_scenes, use_same_seed,
                carry_i2v_guides, mid_scene_guide, mid_scene_guide_strength,
                embed_guidance, embed_guidance_strength, transition_duration,
                decode_tile_size, refinement_key_input, embed_guidance_source,
                score_slider, score_slider_strength,
                joyai_memory=joyai_memory, joyai_memory_size=joyai_memory_size,
                joyai_fix_frames=joyai_fix_frames, joyai_frame_select=joyai_frame_select,
                joyai_memory_strength=joyai_memory_strength,
                joyai_audio_memory=joyai_audio_memory, v2a_grad_scale=v2a_grad_scale,
                alg_blur_guides=alg_blur_guides,
                alg_guide_blur_strength=alg_guide_blur_strength,
                alg_guide_blur_sigma_threshold=alg_guide_blur_sigma_threshold,
                alg_anchor=alg_anchor, alg_anchor_strength=alg_anchor_strength,
                alg_anchor_sigma_threshold=alg_anchor_sigma_threshold,
                bounded_attention_enabled=bounded_attention_enabled,
                output_guidance=output_guidance, output_guidance_strength=output_guidance_strength,
                dynashift=dynashift, dynashift_strength=dynashift_strength,
                dynashift_threshold=dynashift_threshold,
                plateau_cache=plateau_cache, plateau_cache_threshold=plateau_cache_threshold,
                context_windows=context_windows, context_window_length=context_window_length,
                context_window_overlap=context_window_overlap,
                context_window_schedule=context_window_schedule,
                context_window_fuse=context_window_fuse,
                context_window_freenoise=context_window_freenoise,
                context_window_retain_first=context_window_retain_first,
                cut_opening_frames=cut_opening_frames,
                second_pass=second_pass, second_pass_op=second_pass_op,
                second_pass_sigmas=second_pass_sigmas,
                second_pass_sampler=second_pass_sampler,
                h3_audio_clock=h3_audio_clock,
            )

        max_scene_count = max(1, int(max_scenes))
        # An entry here is a SCENE — except a companion. A wired r2v conditioning arrives as
        # several entries that all describe ONE generation (the reference block and the
        # encoded prompt); Studio tags everything after the first so they are not counted as
        # scenes. They ride with every scene instead, below.
        _companions = [c for c in positive if self._is_companion_conditioning(c)]
        scene_conditionings = [c for c in positive
                               if not self._is_companion_conditioning(c)][:max_scene_count]
        if not scene_conditionings and positive:
            scene_conditionings = positive[:1]     # all companions: still sample something
        scene_count = len(scene_conditionings)
        if _companions:
            print(f"[FunPackSceneChain] {len(_companions)} companion conditioning entr"
                  f"{'y' if len(_companions) == 1 else 'ies'} ride with every scene "
                  f"(a wired reference conditioning); {scene_count} scene(s) to sample.")
        # Keyframe pins wired in from a MiniMax H3 Image to Video node, read once for the run.
        _h3_wired_pins = self._h3_external_pins(h3_keyframes) if self._is_h3 else None
        # Did the opening scene get an anchor pin? cut_opening_frames' H3 path needs it.
        _h3_opening_anchored = False
        if h3_keyframes and not self._is_h3:
            print("[FunPackSceneChain] h3_keyframes is wired but this model is not MiniMax H3 — ignored.")
        elif h3_keyframes and _h3_wired_pins is None:
            print("[FunPackSceneChain] h3_keyframes is wired but carries no keyframe pins — the "
                  "source node had no first_frame/last_frame image.")
        time_scale = self._time_scale(vae)
        video_frames = self._validate_template_length(latent_template, num_frames_per_scene, time_scale, vae=vae)
        video_overlap = self._overlap_frames(latent_template, frame_overlap, time_scale, vae=vae)

        output = None
        report_lines = []
        carried_guide_frames = 0
        boundary_entries = []
        cumulative_latent_frames = 0
        # (start_pixel, count) ranges cut_opening_frames will remove from the DECODED video,
        # collected per eligible scene and applied once after decode. Latents stay whole so
        # every frame decodes with the temporal origin it was sampled against.
        _cut_spans = []
        # Read before the scene loop, not just before the decode: cut_opening_frames is a
        # crop on the decoded frames now, so scene eligibility depends on knowing whether
        # anything is being decoded at all.
        want_image = self._output_connected(prompt, unique_id, 1)

        # Load liked direction once for embed_guidance. Source selects which learned direction:
        # 'relative' = this prompt's key; 'absolute' = the global, prompt-agnostic taste store
        # (keyless, so it works even without refinement_key_input).
        _liked_dir = None
        _bad_dir = None
        _value_fn = None
        _prompt_dir_index = None  # per-prompt taste retrieval index (taste_nearest_prompt)
        _eg_source = str(embed_guidance_source or "relative").lower()
        _eg_key = self._absolute_key() if _eg_source == "absolute" else refinement_key_input
        if (embed_guidance or score_slider) and _eg_key:
            _liked_dir = self._load_liked_direction(_eg_key)
            if _liked_dir is None:
                print(f"[FunPackSceneChain] taste steering ({_eg_source}): no liked direction found (need 3+ liked generations)")
            else:
                if taste_nearest_prompt:
                    _prompt_dir_index = self._load_prompt_dir_index(_eg_key) or None
                    if _prompt_dir_index:
                        print(f"[FunPackSceneChain] taste_nearest_prompt: on — retrieving per-scene "
                              f"direction from {len(_prompt_dir_index)} rated prompt(s), global liked_dir as fallback")
                    else:
                        print("[FunPackSceneChain] taste_nearest_prompt: on but index empty — "
                              "using global liked_dir (rate a few liked gens to build the index)")
                _value_fn = self._load_value_function(_eg_key)
                if embed_guidance:
                    if _value_fn:
                        ready = _value_fn.is_ready()
                        mode = f"value function ({_value_fn.n_trained} samples, ascent {'on' if ready else 'pending'})"
                    else:
                        mode = "fixed direction"
                    print(f"[FunPackSceneChain] embed_guidance ({_eg_source}): active via {mode}, strength={embed_guidance_strength}")
                if score_slider:
                    # Contrastive pole for the slider: the learned bad direction from the same
                    # store, when 3+ disliked/awful gens have populated it.
                    _bad_dir = self._load_taste_direction(_eg_key, "bad_dir")
                    pole = "contrastive (liked-vs-bad)" if _bad_dir is not None else "symmetric (+/-liked)"
                    print(f"[FunPackSceneChain] score_slider ({_eg_source}): active (score-space), eta={score_slider_strength}, pole={pole}")

        # DynaShift negative latent memory — loaded once per run. Empty bank = feature
        # stays silent (nothing rated bad yet for this key); it fills as awful /
        # wrong-appearance ratings land.
        _dynashift_negatives = None
        if dynashift and refinement_key_input:
            try:
                try:
                    from .negative_memory import load_negatives as _load_negs
                except ImportError:
                    from negative_memory import load_negatives as _load_negs
                _dynashift_negatives = _load_negs(refinement_key_input) or None
            except Exception as _e:
                print(f"[FunPackSceneChain] dynashift: bank load failed ({_e})")
            if _dynashift_negatives is None:
                print("[FunPackSceneChain] dynashift: negative bank empty — rate a bad "
                      "generation (Awful / Wrong appearance) to populate it")
            else:
                print(f"[FunPackSceneChain] dynashift: active with {len(_dynashift_negatives)} "
                      f"negative(s), strength={dynashift_strength}, threshold={dynashift_threshold}")
        elif dynashift:
            print("[FunPackSceneChain] dynashift: requires refinement_key_input — disabled")

        # Output-space value function for output_guidance — sibling of the block above, but
        # keyed on refinement_key_input directly (relative-only for now, no absolute mode yet)
        # since it's trained on the sampler's own predicted output, not on prompt conditioning.
        _output_value_fn = None
        if output_guidance and refinement_key_input:
            _output_value_fn = self._load_output_value_function(refinement_key_input)
            if _output_value_fn is None:
                print("[FunPackSceneChain] output_guidance: value function not ready yet "
                      "(needs 10+ rated generations to reach MIN_SAMPLES)")
            else:
                print(f"[FunPackSceneChain] output_guidance: active ({_output_value_fn.n_trained} samples), "
                      f"strength={output_guidance_strength}")

        # How much of the schedule the steering actually reaches. "active" alone hid the
        # fact that a shift-heavy H3 schedule left these wrappers with almost no window.
        if embed_guidance or score_slider or dynashift or output_guidance:
            _cov = _steer_ramp_coverage(_steer_ramp, sigmas)
            if _cov is not None:
                _gated, _total, _peak = _cov
                print(f"[FunPackSceneChain] steering window: {_gated} of {_total} steps "
                      f"(peak gate {_peak:.2f})"
                      + (" — read off the schedule's base grid" if self._is_h3 else ""))
                if _gated == 0:
                    print("[FunPackSceneChain] steering window: NOTHING will steer on this "
                          "schedule — every rating-driven mechanism is gated off")

        # Phase timing + quantized-path census: "the matmuls got faster" and "the video
        # arrives sooner" are different claims — the report separates sampling, decode,
        # and everything else so an optimization's real ceiling is visible. The census
        # calls out quantized layers that weight patches (LoRA / refiner attn deltas) or
        # emulation force OFF the fast path — those dequantize on every call.
        _t_run0 = _time.perf_counter()
        _phase_sampling = 0.0
        _phase_decode = 0.0
        try:
            _q_total = _q_off = 0
            for _qm in model.model.diffusion_model.modules():
                if getattr(_qm, "quant_format", None):
                    _q_total += 1
                    if len(getattr(_qm, "weight_function", []) or []) > 0 or getattr(_qm, "_full_precision_mm", False):
                        _q_off += 1
            if _q_total:
                print(f"[FunPackSceneChain] quantized layers: {_q_total} total, {_q_off} forced OFF "
                      "the quantized matmul path (weight patches / emulation)"
                      + (" — those dequantize every call, slower than bf16" if _q_off else ""))
        except Exception:
            pass

        first_scene_seed = self._scene_seed(scene_conditionings[0])
        if first_scene_seed is None:
            first_scene_seed = int(seed)

        # A second pass samples every scene twice, so its steps have to be in the total or
        # the bar overflows on scene 1 and jumps backwards on scene 2 (each scene's offset
        # is a multiple of steps_per_scene). The pass-2 schedule is one node input shared by
        # every scene, so it can be measured once here — and validated the same way the loop
        # will, so an unusable schedule doesn't inflate the total for a pass that gets skipped.
        _pass2_steps = 0
        if second_pass:
            _sp_probe, _ = self._second_pass_schedule(second_pass_sigmas)
            if _sp_probe is not None:
                _pass2_steps = max(0, int(_sp_probe.numel()) - 1)
        steps_per_scene = max(1, (int(len(sigmas)) - 1) + _pass2_steps)
        total_sampling_steps = scene_count * steps_per_scene
        pbar = None
        try:
            pbar = comfy.utils.ProgressBar(total_sampling_steps)
            pbar.update_absolute(0)
        except Exception:
            pass

        scene_guides_cfg = self._parse_scene_guides(funpack_scene_guides)
        per_scene_guides = (scene_guides_cfg or {}).get("scenes") if scene_guides_cfg else None
        scene_anchors = self._parse_scene_anchors(funpack_scene_anchors)
        scene_outputs: list = []
        scene_media_by_ref = self._parse_scene_anchors(funpack_scene_media_refs)
        scene_runs: list = []
        joyai_bank = _JoyAIMemoryBank(joyai_memory_size, joyai_fix_frames) if joyai_memory else None
        _identity_overlap_state: dict = {}
        _detail_upsampler_model = None  # lazy: resolved+loaded at the first detailed scene
        _detail_disabled_reason = None  # set on resolve/load failure: don't retry per scene
        _ctx_unsupported_reported = False  # print the "core too old" line once, not per scene

        for scene_index, scene_cond in enumerate(scene_conditionings):
            scene_positive = [scene_cond] + _companions
            scene_negative = negative

            h3_ref_count = 0
            provided_seed = self._scene_seed(scene_cond)
            if use_same_seed:
                scene_seed = first_scene_seed
            else:
                scene_seed = provided_seed if provided_seed is not None else int(seed) + scene_index
            carried = 0
            soft_carried = 0
            guide_tail = 0
            audio_tail = 0
            identity_ref_filename = None
            run_mechanisms: list = []
            anchor_meta = (scene_anchors or {}).get(str(scene_index))
            # Everything that crosses a scene boundary (carried overlap, the anchor's
            # continuation, the soft join) comes off the finished chain, which a
            # resolution-changing second_pass_op leaves on a different grid than the
            # template this scene is built from. Bring it back once, here, rather than at
            # each splice — identical object when no op resized anything.
            _carry_source = output if output is None else self._match_template_resolution(
                output, latent_template)
            if _carry_source is not output:
                run_mechanisms.append(
                    "second_pass_op resized the previous scene — its carried frames were "
                    "brought back to the template grid for this scene's continuity")
            if output is None:
                chunk = self._clone_latent(latent_template)
                custom_guides = None
                if per_scene_guides and scene_index < len(per_scene_guides):
                    custom_guides = per_scene_guides[scene_index]
                if custom_guides:
                    run_mechanisms.append("custom_guide_stack")
                    chunk, scene_positive, scene_negative, carried, guide_tail, identity_ref_filename = self._apply_configured_guides(
                        chunk, scene_index, custom_guides, latent_template, scene_outputs, scene_media_by_ref,
                        scene_positive, scene_negative, vae,
                        identity_transfer_enabled=identity_transfer_enabled,
                    )
                    carried_guide_frames = max(carried_guide_frames, carried)
            elif anchor_meta:
                run_mechanisms.append("mixed_i2v_anchor")
                effect = self._scene_transition_effect(scene_cond)
                boundary_latent = cumulative_latent_frames
                boundary_pixel = self._latent_to_pixel_frame(boundary_latent, time_scale)
                boundary_entries.append({
                    "between": [scene_index, scene_index + 1],
                    "boundary_latent": boundary_latent,
                    "pixel_frame": max(0, boundary_pixel),
                    "effect": effect if effect and transition_duration > 0 else None,
                })
                chunk = self._build_mixed_anchor_chunk(
                    vae, anchor_meta, latent_template, _carry_source, video_overlap,
                    carry_overlap=carry_overlap_through_anchor,
                )
                if carry_overlap_through_anchor and video_overlap > 0:
                    run_mechanisms.append(f"latent_overlap_through_anchor({frame_overlap}px)")
                # The anchor branch skips _apply_configured_guides entirely, so an identity_pin
                # guide configured for this scene would otherwise never resolve — Best-FaceID
                # identity_transfer needs this to fire on the exact scenes that swap anchors.
                if per_scene_guides and scene_index < len(per_scene_guides):
                    identity_ref_filename = self._identity_pin_filename(
                        per_scene_guides[scene_index], scene_media_by_ref, identity_transfer_enabled,
                    )
                    if identity_ref_filename:
                        run_mechanisms.append("identity_pin_on_anchor_scene")
            else:
                # Record boundary before blending
                effect = self._scene_transition_effect(scene_cond)
                boundary_latent = cumulative_latent_frames
                boundary_pixel = self._latent_to_pixel_frame(boundary_latent, time_scale)
                boundary_entries.append({
                    "between": [scene_index, scene_index + 1],
                    "boundary_latent": boundary_latent,
                    "pixel_frame": max(0, boundary_pixel),
                    "effect": effect if effect and transition_duration > 0 else None,
                })
                chunk = self._build_continuation_chunk(latent_template, _carry_source, video_overlap)
                if video_overlap == 0:
                    chunk, soft_carried = self._prepend_soft_continuation(chunk, _carry_source)
                    if soft_carried > 0:
                        run_mechanisms.append(f"soft_continuation({soft_carried})")
                elif video_overlap > 0:
                    run_mechanisms.append(f"latent_overlap({frame_overlap}px)")
                custom_guides = None
                if per_scene_guides and scene_index < len(per_scene_guides):
                    custom_guides = per_scene_guides[scene_index]
                if custom_guides:
                    run_mechanisms.append("custom_guide_stack")
                    chunk, scene_positive, scene_negative, carried, guide_tail, identity_ref_filename = self._apply_configured_guides(
                        chunk, scene_index, custom_guides, latent_template, scene_outputs, scene_media_by_ref,
                        scene_positive, scene_negative, vae,
                        identity_transfer_enabled=identity_transfer_enabled,
                    )
                    carried_guide_frames = max(carried_guide_frames, carried)
                elif carry_i2v_guides:
                    run_mechanisms.append("carry_i2v_guides")
                    chunk, scene_positive, scene_negative, carried = self._append_i2v_guides(
                        chunk, latent_template, scene_positive, scene_negative,
                    )
                    carried_guide_frames = max(carried_guide_frames, carried)
                if joyai_memory and joyai_bank is not None and not custom_guides:
                    mem_frames = joyai_bank.frames()
                    run_mechanisms.append(f"joyai_memory({len(mem_frames)}/{joyai_memory_size},fix={joyai_fix_frames})")
                    chunk, scene_positive, scene_negative, guide_tail = self._append_joyai_memory_guides(
                        chunk, mem_frames, scene_positive, scene_negative, vae, joyai_memory_strength,
                    )
                    if joyai_audio_memory:
                        chunk, audio_tail = self._append_joyai_audio_memory(chunk, joyai_bank.audio())
                        if audio_tail > 0:
                            run_mechanisms.append(f"joyai_audio_memory({audio_tail})")
                elif mid_scene_guide and not custom_guides:
                    run_mechanisms.append("mid_scene_guide")
                    chunk, scene_positive, scene_negative, guide_tail = self._append_mid_scene_guide(
                        chunk, output, scene_positive, scene_negative, vae, mid_scene_guide_strength,
                    )
                elif not custom_guides:
                    guide_tail = 0

            # H3 i2v anchor (fl2va). On LTX the anchor is already IN the chunk — either written
            # by LTXVImgToVideoInplace above, or, for the opening scene, baked into the latent
            # template by whatever produced it. H3 has no latent i2v path at all, so the anchor
            # image becomes a frame-0 keyframe pin here, after the chunk is final (the pin is
            # encoded at this scene's canvas) and for every branch that carries an anchor.
            if self._is_h3:
                _anchor_image, _anchor_strength = None, 1.0
                _anchor_file = (anchor_meta or {}).get("filename")
                if _anchor_file:
                    _anchor_image = self._load_image_tensor(_anchor_file, keep_alpha=True)
                    _anchor_strength = float(anchor_meta.get("strength", 1.0))
                elif output is None:
                    # Studio owns source_image (it presents it to Qwen); it hands the pixels on
                    # so this side can encode them. Opening scene only — later scenes continue
                    # from the previous one unless they carry their own anchor.
                    _studio_anchor = self._conditioning_value(scene_positive, "funpack_h3_anchor")
                    if isinstance(_studio_anchor, dict):
                        _anchor_image = _studio_anchor.get("image")
                if _anchor_image is not None:
                    scene_positive, _pinned = self._apply_h3_anchor(
                        scene_positive, chunk, vae, _anchor_image, _anchor_strength)
                    if _pinned:
                        run_mechanisms.append(f"h3_keyframe_anchor(strength={_anchor_strength:g})")
                        # cut_opening_frames cuts the front of the FINISHED clip, so what it
                        # needs to know is whether the clip OPENS on an anchor — i.e. whether
                        # the first scene got one. A later scene's anchor sits in the middle
                        # of the batch, where a head crop would never reach it.
                        if scene_index == 0:
                            _h3_opening_anchored = True
                # Pins rescued from a wired MiniMax H3 Image to Video node. Applied AFTER the
                # anchor so an explicitly wired first_frame wins the frame-0 slot — the user
                # drew that wire, the anchor is inferred from the timeline.
                if _h3_wired_pins:
                    scene_positive, _pin_labels = self._apply_h3_external_pins(
                        scene_positive, _h3_wired_pins, scene_index, scene_count)
                    if _pin_labels:
                        run_mechanisms.append(f"h3_wired_keyframes({'+'.join(_pin_labels)})")
                    if "first" in _pin_labels:
                        _h3_opening_anchored = True
                # Nothing claimed frame 0 and a previous scene exists: continue from its last
                # frame. Without this the carried latent tail is not conditioning on H3 — the
                # seam matched and the rest of the shot knew nothing about the scene before it.
                if _carry_source is not None:
                    scene_positive, _continued = self._h3_continuation_pin(
                        scene_positive, _carry_source)
                    if _continued:
                        run_mechanisms.append("h3_continuation_pin")

            # ref2va reference blocks. Applied after the chunk is final because the blocks are
            # sized against this scene's canvas, and to every scene branch (fresh / anchored /
            # continuation) because a reference identity is meant to hold across the whole
            # chain, not just the opening shot.
            if self._is_h3:
                scene_positive, h3_ref_count = self._apply_h3_references(
                    scene_positive, chunk, vae, audio_vae=audio_vae)
                if h3_ref_count:
                    run_mechanisms.append(f"h3_ref2va({h3_ref_count})")
                self._report_h3_checkpoint_mode(scene_positive)

            # Everything from here through sampling installs per-scene state on the SHARED
            # model (function wrappers, forward hooks). One snapshot + one finally guarantees
            # the model leaves this scene exactly as it entered it — even on interrupt/OOM
            # mid-sampling, where the old per-feature unwind blocks were never reached and
            # the wrappers leaked in-process, double-steering every later run (same failure
            # mode as the block-hook leak; see _strip_funpack_scene_wrappers).
            _scene_base_wrapper = model.model_options.get("model_function_wrapper")
            _v2a_handles = []
            _identity_overlap_handle = None
            _plateau_stats = None
            _ctx_remove = None
            try:
                # Context windows: core-owned, installed first so every FunPack wrapper below
                # sits inside the per-window forward rather than around the whole clip.
                if context_windows:
                    _ctx_remove, _ctx_latent_len, _ctx_reason = self._install_context_windows(
                        model, context_window_length, context_window_overlap,
                        context_window_schedule, context_window_fuse,
                        context_window_freenoise, context_window_retain_first)
                    if _ctx_remove is None:
                        # Every reason here is a mismatch with the installed core, so it is
                        # the same for every scene — print it once, report it on each.
                        if not _ctx_unsupported_reported:
                            print(f"[FunPackSceneChain] context_windows requested but "
                                  f"{_ctx_reason} — skipped.")
                            _ctx_unsupported_reported = True
                        run_mechanisms.append(f"context_windows(SKIPPED: {_ctx_reason})")
                    else:
                        if _ctx_reason:
                            run_mechanisms.append(f"context_windows NOTE: {_ctx_reason}")
                        # Core engages windowing only when the scene is longer than the window,
                        # and logs that decision itself; say so here too so a scene that silently
                        # sampled whole doesn't read as the feature being broken.
                        _scene_latent_len = self._context_scene_latent_frames(chunk)
                        if _scene_latent_len is not None and _scene_latent_len <= _ctx_latent_len:
                            run_mechanisms.append(
                                f"context_windows(inactive: scene {_scene_latent_len} <= window "
                                f"{_ctx_latent_len} latent frames — raise num_frames_per_scene or "
                                f"lower context_window_length)")
                        else:
                            run_mechanisms.append(
                                f"context_windows({context_window_schedule},{context_window_fuse},"
                                f"len={_ctx_latent_len},ovl={max(int(context_window_overlap) // 8, 0)}"
                                f"{',freenoise' if context_window_freenoise else ''}"
                                f"{',retain_first' if context_window_retain_first else ''})")
                # Innermost wrapper (installed first): caches the raw base-model forward on the
                # near-noise plateau so later plateau steps reuse it. All guidance wrappers below
                # layer around it and still post-process each step's (cached-or-fresh) prediction.
                #
                # MUTUALLY EXCLUSIVE with context windows: the cache is keyed by
                # (input.shape, cond_or_uncond), and every window within one step calls the model
                # with the SAME shape. Windows 2..N would therefore be handed window 1's cached
                # prediction on every plateau step — one window's content bleeding across the whole
                # clip. Context windows wins (it's a capability, the cache is a speed experiment);
                # say so in the report rather than silently dropping one of two ticked boxes.
                if plateau_cache and _ctx_remove is not None:
                    run_mechanisms.append(
                        "plateau_cache(SKIPPED: incompatible with context_windows — the per-step "
                        "cache cannot tell two windows apart)")
                elif plateau_cache:
                    _plateau_stats = self._build_plateau_cache_wrapper(model, plateau_cache_threshold)
                # Loop temporal style (auto director's funpack_temporal_loop): Mobius latent
                # roll. Installed BELOW the guidance wrappers (embed guidance / slider /
                # dynashift / output guidance) so they see canonical-orientation inputs and
                # predictions — the roll exists only around the base forward. Appended guide
                # frames stay pinned: the wrapper rolls only the content region in front of
                # the guide tail (counted from keyframe_idxs / the audio mask), so guides
                # keep informing the whole cycle without ever being dragged into it.
                if self._scene_temporal_loop(scene_cond):
                    try:
                        from .ltx_enhancements import make_loop_temporal_wrapper
                    except ImportError:
                        from ltx_enhancements import make_loop_temporal_wrapper
                    _loop_base = model.model_options.get("model_function_wrapper")
                    model.model_options["model_function_wrapper"] = _tag_scene_wrapper(
                        make_loop_temporal_wrapper(_loop_base), _loop_base)
                    _loop_pinned = guide_tail > 0 or carried > 0 or carried_guide_frames > 0
                    run_mechanisms.append(
                        "temporal_loop_roll(content-only: guide tail pinned)" if _loop_pinned
                        else "temporal_loop_roll")
                # taste_nearest_prompt: swap the single global liked direction for the one
                # learned on this scene's closest rated prompts (resolved from the ORIGINAL
                # scene conditioning, before any value-fn ascent mutates it). Falls back to
                # the global _liked_dir when nothing rated is close enough.
                _scene_liked_dir = _liked_dir
                if _prompt_dir_index and _liked_dir is not None and scene_positive:
                    _retrieved = self._resolve_prompt_keyed_direction(_prompt_dir_index, scene_positive[0][0])
                    if _retrieved is not None:
                        _scene_liked_dir = _retrieved
                        run_mechanisms.append("taste_nearest_prompt")
                if embed_guidance and _value_fn is not None and _value_fn.is_ready():
                    run_mechanisms.append("embed_guidance_vf_ascend")
                    orig_cond, orig_extra = scene_positive[0][0], scene_positive[0][1]
                    ascended = self._protect_audio(_value_fn.ascend(orig_cond), orig_cond)
                    scene_positive = [[ascended, orig_extra]] + list(scene_positive[1:])
                # The taste store, the value function and the negative bank all live in the
                # RAW conditioning space, which is not what every model consumes -- hand the
                # raw scene cond to whoever needs to bridge the two.
                _raw_scene_cond = (scene_positive[0][0]
                                   if scene_positive and isinstance(scene_positive[0], (list, tuple))
                                   else None)
                if embed_guidance and _scene_liked_dir is not None:
                    run_mechanisms.append(f"embed_guidance({_eg_source},{embed_guidance_strength})")
                    self._build_embed_guidance_wrapper(model, _scene_liked_dir, embed_guidance_strength,
                                                       value_fn=_value_fn, raw_cond=_raw_scene_cond,
                                                       ramp_fn=_steer_ramp)
                if score_slider and _scene_liked_dir is not None:
                    _pole = "contrastive" if _bad_dir is not None else "symmetric"
                    run_mechanisms.append(f"score_slider({_eg_source},{score_slider_strength},{_pole})")
                    self._build_score_slider_wrapper(model, _scene_liked_dir, score_slider_strength,
                                                     bad_dir=_bad_dir, raw_cond=_raw_scene_cond,
                                                     ramp_fn=_steer_ramp)
                if dynashift and _dynashift_negatives:
                    run_mechanisms.append(
                        f"dynashift({len(_dynashift_negatives)}neg,{dynashift_strength},thr={dynashift_threshold})")
                    self._build_dynashift_wrapper(
                        model, _dynashift_negatives, dynashift_strength, dynashift_threshold,
                        raw_cond=_raw_scene_cond, ramp_fn=_steer_ramp)
                if output_guidance and _output_value_fn is not None:
                    # Installed outermost (after embed_guidance/score_slider/dynashift) so it
                    # corrects whatever prediction those already produced, not the raw base one.
                    run_mechanisms.append(f"output_guidance({output_guidance_strength})")
                    self._build_output_guidance_wrapper(model, _output_value_fn, output_guidance_strength,
                                                        ramp_fn=_steer_ramp)
                # Per-scene temporal style (auto / pulse / rapid_start / rapid_end /
                # rapid_start_end): layer a frame_rate wrapper on top of whatever is
                # installed (e.g. embed guidance).
                _scene_mode = self._scene_temporal_mode(scene_cond)
                _scene_mult = self._scene_temporal_mult(scene_cond)
                _cur_wrapper = model.model_options.get("model_function_wrapper")
                if _scene_mode == "pulse":
                    try:
                        from .ltx_enhancements import make_pulse_temporal_wrapper
                    except ImportError:
                        from ltx_enhancements import make_pulse_temporal_wrapper
                    _tw = make_pulse_temporal_wrapper(_cur_wrapper)
                    if _tw is not None:
                        model.model_options["model_function_wrapper"] = _tag_scene_wrapper(_tw, _cur_wrapper)
                elif _scene_mode in ("rapid_start", "rapid_end", "rapid_start_end"):
                    try:
                        from .ltx_enhancements import make_rapid_temporal_wrapper
                    except ImportError:
                        from ltx_enhancements import make_rapid_temporal_wrapper
                    _tw = make_rapid_temporal_wrapper(_cur_wrapper, _scene_mode)
                    if _tw is not None:
                        model.model_options["model_function_wrapper"] = _tag_scene_wrapper(_tw, _cur_wrapper)
                elif _scene_mult is not None and abs(_scene_mult - 1.0) >= 1e-3:
                    try:
                        from .ltx_enhancements import make_temporal_wrapper
                    except ImportError:
                        from ltx_enhancements import make_temporal_wrapper
                    _tw = make_temporal_wrapper(_cur_wrapper, _scene_mult)
                    if _tw is not None:
                        model.model_options["model_function_wrapper"] = _tag_scene_wrapper(_tw, _cur_wrapper)
                # JoyAI v2a coupling: amplify (or mute) the trained video->audio cross-attention
                # for this scene's denoise. Only when audio memory is on and the scale differs
                # from native (1.0).
                if joyai_audio_memory and audio_tail > 0:
                    _v2a_handles = self._install_v2a_scale(model, v2a_grad_scale)
                    if _v2a_handles:
                        run_mechanisms.append(f"v2a_grad_scale({v2a_grad_scale})")
                if identity_transfer_enabled and identity_ref_filename:
                    _id_ref_latent, _id_seg_value, _id_pos_tokens, _id_neg_tokens = self._resolve_identity_overlap(
                        _identity_overlap_state, identity_ref_filename, vae, chunk,
                        identity_projector, source_id, phase_scale, id_strength, arcface_mode, debug_log,
                    )
                    if _id_ref_latent is not None:
                        _identity_overlap_handle = self._install_identity_overlap(model, _id_ref_latent, _id_seg_value)
                        if _identity_overlap_handle:
                            run_mechanisms.append(f"identity_transfer_overlap(seg={_id_seg_value})")
                        if _id_pos_tokens is not None:
                            scene_positive, scene_negative = self._append_identity_context_tokens(
                                scene_positive, scene_negative, _id_pos_tokens, _id_neg_tokens,
                            )
                            run_mechanisms.append(f"identity_transfer_arcface(id_strength={id_strength})")
                _t_sample0 = _time.perf_counter()
                _sample_kwargs = dict(
                    pbar=pbar, step_offset=scene_index * steps_per_scene,
                    alg_guide_tail_frames=(guide_tail if (alg_blur_guides and guide_tail > 0) else 0),
                    alg_guide_blur_strength=alg_guide_blur_strength,
                    alg_guide_blur_sigma_threshold=alg_guide_blur_sigma_threshold,
                    alg_anchor=alg_anchor,
                    alg_anchor_strength=alg_anchor_strength,
                    alg_anchor_sigma_threshold=alg_anchor_sigma_threshold,
                    bounded_attention_enabled=bounded_attention_enabled,
                    h3_audio_clock=h3_audio_clock,
                )
                # cut_opening_frames: let the real, untouched i2v anchor condition the scene
                # at full strength, then cut it out of the finished clip instead of weakening
                # it on the way in (ALG blurs the anchor and loses character detail; overlap
                # tokens approximate it and lose some too). Eligibility is decided here, while
                # the pre-sampling chunk is still around; the cut itself is a post-process on
                # the finished latent, below, and costs no sampling at all.
                # H3 takes the other path entirely (_cut_opening_pixels, after decode): its
                # anchor is a keyframe condition row, not a pinned latent prefix, so none of
                # the eligibility below can even be evaluated.
                _cut_drop = 0
                if int(cut_opening_frames) > 0 and not self._is_h3:
                    # The cut is measured in REAL frames and applied to decoded pixels, so
                    # there is no rounding to a latent grid any more and no minimum: 1 means
                    # 1. This is only a "does this scene qualify" flag now.
                    _cut_drop = 1
                    if carried + soft_carried > 0:
                        _reason = ("continuation scene — the opening is carried frames from the "
                                   "previous scene, not an anchor, and cutting them would break "
                                   "the join with it")
                    elif guide_tail > 0 or audio_tail > 0:
                        _reason = ("scene is carrying appended frames (mid_scene_guide / carried "
                                   "i2v guides / JoyAI audio memory) that are stripped after "
                                   "sampling — the audio crop is derived from the video length, "
                                   "which those extra frames would make wrong")
                    elif self._context_scene_latent_frames(chunk) is None:
                        _reason = "scene latent could not be read"
                    elif not want_image:
                        # The cut is a crop on the decoded frames now, so with nothing
                        # decoding there is nothing to crop — the same rule H3 already has.
                        _reason = ("the IMAGES output is not connected — the cut is a crop on "
                                   "the decoded frames, so there is nothing to crop")
                    elif self._anchor_pinned_frames(chunk) <= 0:
                        # No pinned prefix = a genuine t2v scene (no anchor image attached).
                        # Cutting here would throw away real generated frames for nothing —
                        # this is the "fake t2v" trick, it needs a real anchor to fake it WITH.
                        _reason = ("scene has no pinned i2v anchor (t2v scene) — nothing to cut "
                                   "out; attach an anchor image, or set cut_opening_frames to 0")
                    else:
                        _reason = None
                    if _reason is not None:
                        run_mechanisms.append(f"cut_opening_frames(SKIPPED: {_reason})")
                        _cut_drop = 0
                # Optional second pass. The shape is deliberately plain: pass 1 runs the
                # main schedule IN FULL, exactly as written, then pass 2 runs its own
                # schedule IN FULL, exactly as written. Nothing is cut short and nothing is
                # derived — to make pass 1 shorter, shorten the main schedule. Pass 2 starts
                # from a FINISHED clip, so it re-enters through comfy's ordinary CONST noise
                # scaling, valid precisely because the input is clean. Total steps are simply
                # the two schedules added up.
                _sp_b = None
                if second_pass:
                    _sp_b, _sp_reason = self._second_pass_schedule(second_pass_sigmas)
                    if _sp_b is None:
                        run_mechanisms.append(f"second_pass(SKIPPED: {_sp_reason})")
                _scene_label = (f"scene {scene_index + 1}/{scene_count}"
                                if scene_count > 1 else "")
                self._set_phase(f"{_scene_label}{' · ' if _scene_label else ''}"
                                f"{'pass 1 of 2' if _sp_b is not None else 'sampling'}")
                _full = self._sample_chunk(
                    model, sampler, sigmas, scene_seed, cfg, scene_positive,
                    scene_negative, chunk, **_sample_kwargs)
                if _sp_b is None:
                    sampled = _full
                else:
                    _sp_state = _full
                    if second_pass_op not in (None, "none", ""):
                        # The op runs on the FINISHED pass-1 clip, which is the only state
                        # it makes sense on: sharpening a half-denoised latent sharpens the
                        # noise in it. Shares segmented detailing's lazily-loaded upsampler.
                        if _detail_upsampler_model is None and _detail_disabled_reason is None:
                            try:
                                try:
                                    from . import detailing as _det
                                except ImportError:
                                    import detailing as _det
                                _chans = None
                                try:
                                    _v = self._latent_tensors(_sp_state)[0]
                                    _chans = int(_v.shape[1]) if _v.ndim >= 5 else None
                                except Exception:  # noqa: BLE001
                                    pass
                                _r = _det.resolve_upsampler_name(detail_upsampler, _chans)
                                _detail_upsampler_model = _det.load_latent_upsampler(_r)
                            except Exception as _exc:  # noqa: BLE001
                                _detail_disabled_reason = str(_exc)
                                # resolve_upsampler_name prints when it DOWNLOADS; nothing
                                # printed when it can't, so a silent skip looked like the op
                                # having run. Say it on the console too, once.
                                print(f"[FunPackSceneChain] second_pass_op={second_pass_op} "
                                      f"needs the latent upsampler and it could not be "
                                      f"loaded: {_detail_disabled_reason} — the second pass "
                                      f"still runs, without the operation.")
                        _sp_state, _op_note = self._second_pass_operate(
                            _sp_state, second_pass_op, _detail_upsampler_model, vae,
                            scale=second_pass_upscale)
                        if _op_note:
                            # "no upsampler could be loaded" on its own doesn't say WHY —
                            # missing huggingface_hub, a failed download, an unreadable file
                            # are all different fixes. Carry the real reason through.
                            if _detail_upsampler_model is None and _detail_disabled_reason:
                                _op_note = f"{_op_note} ({_detail_disabled_reason})"
                            run_mechanisms.append(_op_note)
                    # A second pass must KEEP the i2v anchor pinned — otherwise pass 2
                    # re-denoises the reference frame and the scene drifts away from the
                    # image it was supposed to start from.
                    _sp_before = _sp_state
                    _sp_pinned = self._anchor_pinned_frames(chunk) > 0
                    _sp_resized = _sp_pinned and self._latent_spatial_changed(_sp_state, chunk)
                    _sp_state = self._restore_pinned_prefix(_sp_state, chunk)
                    if _sp_state is _sp_before and _sp_pinned:
                        # The mask could not be rebuilt at the new size — an unpinned pass 2
                        # can drift off the reference image, so say so.
                        run_mechanisms.append(
                            "second_pass NOTE: the i2v anchor could not stay pinned for pass 2 "
                            "because second_pass_op changed the latent resolution — pass 2 runs "
                            "unpinned and may drift from the reference image.")
                    elif _sp_resized:
                        run_mechanisms.append(
                            "second_pass NOTE: second_pass_op changed the resolution, so the i2v "
                            "anchor stays pinned as the UPSCALED anchor — the same upsampler pass "
                            "the rest of the clip got, not the encoded source image.")
                    # Same resolution change, second casualty: guide keyframes are recorded as
                    # TOKEN indices into the pass-1 grid, and 2x spatial means 4x the tokens per
                    # latent frame, so those indices now address the wrong tokens entirely.
                    # Since the LTX-2.5 commit core rejects it outright ("keyframe_idxs holds N
                    # tokens, which is not a whole number of M-token latent frames") and tells
                    # you to crop the guides before upscaling; older cores silently mis-placed
                    # them, which is worse. Drop them for pass 2, exactly as segmented detailing
                    # does for its crop — and only for pass 2, so pass 1 keeps every guide.
                    _sp_positive, _sp_negative = scene_positive, scene_negative
                    if self._latent_spatial_changed(_sp_state, chunk):
                        try:
                            from .detailing import has_layout_conds, strip_layout_conds
                        except ImportError:
                            from detailing import has_layout_conds, strip_layout_conds
                        if has_layout_conds(scene_positive) or has_layout_conds(scene_negative):
                            _sp_positive = strip_layout_conds(scene_positive)
                            _sp_negative = strip_layout_conds(scene_negative)
                            run_mechanisms.append(
                                "second_pass NOTE: mid-scene guides / guide keyframes were "
                                "dropped for pass 2 because second_pass_op changed the latent "
                                "resolution — their token positions describe the pass-1 grid. "
                                "Pass 1 used them in full.")
                    # H3's own casualty of the same resolution change: a keyframe pin is
                    # packed as condition ROWS, so its token count belongs to the grid it was
                    # encoded on and the model refuses it outright on any other. The LTX keys
                    # above are dropped; a pin is resampled instead, because on H3 the pin IS
                    # the anchor and pass 2 has to keep holding it.
                    if self._is_h3 and self._latent_spatial_changed(_sp_state, chunk):
                        try:
                            _h, _w = self._latent_tensors(_sp_state)[0].shape[-2:]
                            _sp_positive, _n_pins = self._h3_rescale_pins(
                                _sp_positive, int(_h), int(_w))
                        except Exception as _pin_exc:  # noqa: BLE001
                            _n_pins = 0
                            print(f"[FunPackSceneChain] H3: could not resample the keyframe "
                                  f"pins for pass 2 ({_pin_exc}).")
                        if _n_pins:
                            run_mechanisms.append(
                                f"second_pass NOTE: {_n_pins} H3 keyframe pin(s) resampled to "
                                f"the pass-2 grid ({int(_h)}x{int(_w)}) — a pin is packed as "
                                f"condition rows, so it only fits the grid it was encoded on")
                    _sp_kw = dict(_sample_kwargs)
                    _sp_kw["step_offset"] = _sample_kwargs["step_offset"] + (int(sigmas.numel()) - 1)
                    # Announced BEFORE it runs: the after-the-fact run_mechanisms line says
                    # a second pass happened, which is no help while you are waiting on one.
                    self._set_phase(f"{_scene_label}{' · ' if _scene_label else ''}pass 2 of 2")
                    # A second sampler is an algorithm change mid-scene, so it is named on
                    # the console: "why does pass 2 look different" needs a visible answer,
                    # and an unwired socket reusing pass 1's sampler looks identical from
                    # the outside.
                    _sp_sampler = sampler if second_pass_sampler is None else second_pass_sampler
                    _sp_which = "" if second_pass_sampler is None else (
                        " using its own sampler ("
                        + getattr(getattr(_sp_sampler, "sampler_function", None), "__name__", "?")
                        + ")")
                    print(f"[FunPack AV] second pass starting"
                          f"{' on ' + _scene_label if _scene_label else ''}: "
                          f"{int(_sp_b.numel()) - 1} steps from sigma {float(_sp_b[0].item()):g}"
                          f"{_sp_which}")
                    sampled = self._sample_chunk(
                        model, _sp_sampler, _sp_b, scene_seed + 4242, cfg, _sp_positive,
                        _sp_negative, _sp_state, **_sp_kw)
                    run_mechanisms.append(
                        f"second_pass({int(sigmas.numel()) - 1} steps + "
                        f"{int(_sp_b.numel()) - 1} steps = "
                        f"{int(sigmas.numel()) + int(_sp_b.numel()) - 2} total, "
                        f"pass 2 from {float(_sp_b[0].item()):g}"
                        f"{', own sampler' if second_pass_sampler is not None else ''})")
                _scene_sample_s = _time.perf_counter() - _t_sample0
                _phase_sampling += _scene_sample_s
                if _plateau_stats is not None:
                    _reused, _computed = _plateau_stats["reused"], _plateau_stats["computed"]
                    _total = _reused + _computed
                    if _total > 0:
                        run_mechanisms.append(
                            f"plateau_cache(skipped {_reused}/{_total} plateau fwd, thr={plateau_cache_threshold})")
            finally:
                # Nothing is sampling once this scene is out of the sampler — including on an
                # interrupt, which is exactly when a stale "pass 2 of 2" would linger.
                self._set_phase("")
                if _ctx_remove is not None:
                    _ctx_remove()
                self._remove_v2a_scale(_v2a_handles)
                self._strip_identity_overlap(_identity_overlap_handle)
                if model.model_options.get("model_function_wrapper") is not _scene_base_wrapper:
                    if _scene_base_wrapper is not None:
                        model.model_options["model_function_wrapper"] = _scene_base_wrapper
                    else:
                        model.model_options.pop("model_function_wrapper", None)
            if _cut_drop > 0:
                # Recorded now, applied after decode. The latent itself is deliberately left
                # whole: cutting it here is what made the opening frame noisy, because the
                # frame promoted to position 0 gets decoded as a temporal origin it was never
                # generated as (see _cut_opening_pixel_spans). The scene's position in the
                # chain is read BEFORE it is appended, so the span is an index into the final
                # decoded video, and anchor scenes append with zero blend overlap, which is
                # what makes that index exact.
                _cut_latent_start = (0 if output is None
                                     else self._tensor_frames(self._latent_tensors(output)[0]))
                _cut_spans.append((self._scene_pixel_start(_cut_latent_start, time_scale),
                                   int(cut_opening_frames)))
                _pinned_n = self._anchor_pinned_frames(chunk)
                # What the anchor actually spans in PIXELS, which is what the cut is now
                # measured in: the pinned prefix is at the scene's origin, so it covers
                # (pinned - 1) * scale + 1 real frames.
                _anchor_px = (_pinned_n - 1) * max(1, int(time_scale)) + 1 if _pinned_n > 0 else 0
                run_mechanisms.append(
                    f"cut_opening_frames(scheduled: {int(cut_opening_frames)} real frames off "
                    f"the front of this scene, cut after decode so the opening frame keeps a "
                    f"clean origin — the scene is that much SHORTER)")
                if int(cut_opening_frames) < _anchor_px:
                    # Part of the anchor is still visible — the one outcome this exists to
                    # prevent, so name the fix.
                    run_mechanisms.append(
                        f"cut_opening_frames WARNING: the pinned anchor spans {_anchor_px} real "
                        f"frames and only {int(cut_opening_frames)} are being cut — raise "
                        f"cut_opening_frames to at least {_anchor_px}")
            if carried + soft_carried > 0:
                sampled = self._crop_video_head(sampled, carried + soft_carried)
            if guide_tail > 0:
                sampled = self._crop_video_tail(sampled, guide_tail)
            if audio_tail > 0:
                sampled = self._crop_audio_tail(sampled, audio_tail)
            # Segmented detailing runs on the clean, fully-cropped scene (guide/audio
            # tails gone, overlap still present — the tube is spatial, so carried head
            # frames detail together with the rest) and BEFORE the JoyAI harvest, so
            # cross-shot memory banks the improved frames. Every skip is LOUD (console +
            # scene report): with the toggle on, silence must mean "ran and found nothing".
            if segmented_detailing and detail_strength > 0 and not _detail_disabled_reason:
                try:
                    try:
                        from . import detailing as _detailing
                    except ImportError:
                        import detailing as _detailing
                    if _detail_upsampler_model is None:
                        _chans = None
                        try:
                            _v = self._latent_tensors(sampled)[0]
                            _chans = int(_v.shape[1]) if _v.ndim >= 5 else None
                        except Exception:  # noqa: BLE001
                            pass
                        _resolved = _detailing.resolve_upsampler_name(detail_upsampler, _chans)
                        _detail_upsampler_model = _detailing.load_latent_upsampler(_resolved)
                        print(f"[FunPackSceneChain] latent upsampler loaded: {_resolved}")
                    _t_detail0 = _time.perf_counter()
                    sampled, _detail_note = _detailing.detail_refine_scene(
                        self, model, vae, sampler, scene_positive, scene_negative, sampled,
                        detail_targets, _detail_upsampler_model, scene_seed, cfg,
                        threshold=detail_threshold, strength=detail_strength,
                        area_cap=detail_max_area, renoise_sigma=detail_denoise,
                        mode=detail_mode, debug=debug_log)
                    # detail_refine_scene always returns a diagnostic note when it actually
                    # ran detection (miss or hit) — None only means it never attempted
                    # detection at all (which can't happen on this branch).
                    if _detail_note:
                        run_mechanisms.append(_detail_note)
                        _phase_sampling += _time.perf_counter() - _t_detail0
                except Exception as _detail_exc:
                    # A failed detail pass must never cost the scene itself — but it must
                    # never fail silently either. Model resolution/load failures disable
                    # the pass for the rest of the run (no per-scene download retries).
                    if _detail_upsampler_model is None:
                        _detail_disabled_reason = str(_detail_exc)
                    print(f"[FunPackSceneChain] SEGMENTED DETAILING SKIPPED (scene kept as-is): {_detail_exc}")
                    run_mechanisms.append(f"segmented_detail(SKIPPED: {_detail_exc})")
            elif segmented_detailing and _detail_disabled_reason:
                run_mechanisms.append("segmented_detail(SKIPPED: upsampler unavailable, see first scene)")
            if joyai_bank is not None:
                # Harvest from the clean, fully-cropped scene so injected memory tails never re-enter
                # the bank. Scene 0 seeds the pinned anchor (num_fix); later scenes roll in. The audio
                # half is harvested only when audio memory is on, else stored as None (video-only).
                # The bank is injected into LATER scenes, which are built from the template,
                # so a scene left upscaled by second_pass_op must be brought back first —
                # otherwise the memory frame is the one thing in the chunk on the wrong grid.
                _joyai_src = self._match_template_resolution(sampled, latent_template)
                v_frame = self._harvest_joyai_frame(_joyai_src, joyai_frame_select)
                a_frame = self._harvest_joyai_audio(_joyai_src, joyai_frame_select) if joyai_audio_memory else None
                joyai_bank.add(v_frame, a_frame)
            # Kept on the TEMPLATE's grid: this list exists to be spliced back in as guide
            # frames for later scenes, and those chunks are built from the template.
            scene_outputs.append(self._clone_latent(
                self._match_template_resolution(sampled, latent_template)))
            # Stays 0 for anchor scenes even with carry_overlap_through_anchor on: the post-sample
            # slerp blend would reach into the anchor image's own leading frame (position 0 of
            # `sampled`) and fade it against the previous scene's tail, undermining the hard cut.
            # The carried frames beyond it are already seeded pre-sample (see
            # _build_mixed_anchor_chunk), so no post-hoc smoothing is needed there.
            blend_overlap = 0 if anchor_meta else video_overlap
            # No grid guard here on purpose: second_pass and second_pass_op are run-level
            # settings applied to every scene, so the chain cannot end up holding two
            # resolutions. Every scene is upscaled or none is.
            output = sampled if output is None else self._blend_latents(output, sampled, blend_overlap)
            cumulative_latent_frames = self._tensor_frames(self._latent_tensors(output)[0])
            scene_meta = self._scene_meta(scene_cond, scene_index)
            scene_runs.append({
                **scene_meta,
                "seed_used": scene_seed,
                "mechanisms": run_mechanisms,
            })
            # Per-scene sampling time + appended-token context: guide/carry scenes sample a
            # LONGER sequence (carried head + guide/audio tails) and take the masked-attention
            # path when guide strengths != 1.0 — this line makes any per-scene overhead
            # attributable at a glance instead of hiding inside the run total.
            _scene_extras = (
                (f", carried={carried + soft_carried}f" if (carried + soft_carried) > 0 else "")
                + (f", guide_tail={guide_tail}f" if guide_tail > 0 else "")
                + (f", audio_tail={audio_tail}f" if audio_tail > 0 else "")
                + (f" [{', '.join(run_mechanisms)}]" if run_mechanisms else "")
            )
            print(f"[FunPackSceneChain] Scene {scene_index + 1}: sampling {_scene_sample_s:.1f}s{_scene_extras}")
            report_lines.append(
                f"Scene {scene_index + 1}: seed={scene_seed}, sampling {_scene_sample_s:.1f}s{_scene_extras}, "
                f"text={scene_meta['text']}"
                + (f" | encode≠text" if scene_meta["encode_text"] != scene_meta["text"] else "")
            )

        del scene_cond, scene_positive, scene_negative, scene_conditionings, chunk, sampled

        # RETURN_TYPES slot indices: 0=latent, 1=images, 2=status...
        # Sampling is fully complete. Latent is untouched and returned as-is.
        # IMAGES: decode the whole latent in one pass, then apply transition effects.
        # (want_image was read before the scene loop — cut_opening_frames eligibility needs it.)

        images = None
        if want_image:
            _t_dec0 = _time.perf_counter()
            video_tensor = self._latent_tensors(output)[0]
            # Say what the decode is about to ask for, BEFORE asking. A decode that takes the
            # process (or the box) down leaves no traceback, so without this line a crash here
            # is indistinguishable from any other sudden death. comfy's own OOM recovery is
            # spatial tiling; it does not shrink the full-size CPU output buffer, which is the
            # host-RAM allocation that a machine actually dies on.
            # The decode is one uninterruptible call with no progress of its own, so without
            # this the UI shows the last sampling phase and a frozen timer — and a decode that
            # is merely SLOW (an fp32 VAE, a CPU fallback, a box that started swapping) is
            # indistinguishable from one that is hung.
            self._set_phase("decoding")
            self._mem_report("sampling done, before decode")
            self._log_decode_plan(vae, video_tensor)
            # Hand back everything sampling was holding first. The banks (dynashift negatives,
            # guide frames, per-scene wrappers) are dead by now but their blocks are still
            # reserved by torch's allocator, and the decode is the peak of the whole run.
            try:
                comfy.model_management.soft_empty_cache()
            except Exception:  # noqa: BLE001
                pass
            if decode_tile_size > 0:
                try:
                    _tile = self._decode_tile_latent(vae, decode_tile_size)
                    decoded = vae.decode_tiled(video_tensor, tile_x=_tile, tile_y=_tile)
                except Exception:
                    decoded = vae.decode(video_tensor)
            else:
                decoded = vae.decode(video_tensor)
            if decoded.dim() == 5:
                b, t, h, w, c = decoded.shape
                decoded = decoded.reshape(b * t, h, w, c)
            images = self._apply_transitions_pixel(decoded, boundary_entries, transition_duration)
            _phase_decode = _time.perf_counter() - _t_dec0
            self._set_phase("")
            print(f"[FunPackSceneChain] decoded in {_phase_decode:.1f}s")
            # After, not just before: the eviction happens DURING the decode, so the two
            # lines together show what moved and where it went.
            self._mem_report("after decode")

        # cut_opening_frames, LTX path. Applied HERE, after a decode that saw every latent
        # frame in its sampled context, rather than on the latent inside the loop — a latent
        # cut promoted a continuation frame to position 0 and the causal VAE decoded it as an
        # origin, which is what made the first frame noisy. Transitions run first so
        # boundary_entries still index the uncut video.
        if _cut_spans and not self._is_h3:
            _px_total = int(images.shape[0]) if isinstance(images, torch.Tensor) else 0
            if _px_total > 0:
                images, _px_dropped = self._cut_opening_pixel_spans(images, _cut_spans)
                if _px_dropped > 0:
                    # The audio stream has to lose the same stretches of TIME or the sound
                    # runs ahead of the picture — silently, with no error, exactly the trap
                    # _crop_stream_head_to exists for on the carried-frame path.
                    output = self._remove_latent_time_spans(output, _cut_spans, _px_total)
                    report_lines.append(
                        f"cut_opening_frames(dropped {_px_dropped} of {_px_total} decoded "
                        f"frames across {len(_cut_spans)} scene opening(s) — cut after decode, "
                        f"so the first surviving frame keeps a clean origin; the audio stream "
                        f"was cropped by the same amount of time)")
                    report_lines.append(
                        "cut_opening_frames NOTE: the LATENT output keeps its FULL video "
                        "stream (cutting it is what produced a noisy opening frame) — take "
                        "video from the IMAGES output on a cut run, audio from the latent.")
                    if transition_duration and boundary_entries:
                        report_lines.append(
                            "cut_opening_frames NOTE: transitions were rendered before this "
                            "cut, so a transition running INTO a cut scene loses the frames "
                            "that were removed with the anchor.")

        # cut_opening_frames, H3 path. The LTX path cut each scene's finished LATENT inside
        # the loop above; H3 has no pinned latent prefix to cut and an off-grid video latent
        # cannot be decoded, so the cut lands here instead — exactly N frames off the head of
        # the decoded batch, with the audio stream moved to match. Every skip is stated: the
        # toggle is on, so silence would read as "it ran".
        if self._is_h3 and int(cut_opening_frames) > 0:
            _h3_cut_reason = None
            if not _h3_opening_anchored:
                _h3_cut_reason = ("the opening scene has no anchor image (t2v) — the cut exists "
                                  "to hide an anchor, and here it would just throw away real "
                                  "generated frames")
            elif not want_image:
                _h3_cut_reason = ("the IMAGES output is not connected — on H3 the cut is a crop "
                                  "on the decoded frames, so there is nothing to crop")
            if _h3_cut_reason is not None:
                report_lines.append(f"cut_opening_frames(SKIPPED: {_h3_cut_reason})")
            else:
                _before = int(images.shape[0])
                images, output, _dropped = self._cut_opening_pixels(
                    images, output, int(cut_opening_frames))
                report_lines.append(
                    f"cut_opening_frames(H3: dropped {_dropped} of {_before} decoded frames "
                    f"from the front — the clip is {_dropped} frames SHORTER and the audio "
                    f"stream was cropped to match)")
                # Said plainly because the two outputs genuinely disagree after this, and the
                # disagreement is silent in a graph that decodes video from the latent.
                report_lines.append(
                    "cut_opening_frames NOTE: the LATENT output keeps its FULL video stream "
                    "(H3's 5k+2 latent grid cannot express the cut) — take video from the "
                    "IMAGES output on a cut run, audio from the latent as usual.")
                if scene_count > 1:
                    report_lines.append(
                        "cut_opening_frames NOTE: only the chain's opening is cut. A later "
                        "scene's own anchor sits mid-batch, where a head crop cannot reach it.")

        if images is None:
            images = torch.zeros(1, 8, 8, 3)

        _t_total = _time.perf_counter() - _t_run0
        _phase_other = max(0.0, _t_total - _phase_sampling - _phase_decode)
        _timing = (f"Timing: total {_t_total:.1f}s = sampling {_phase_sampling:.1f}s "
                   f"({100.0 * _phase_sampling / max(_t_total, 1e-9):.0f}%) + VAE decode "
                   f"{_phase_decode:.1f}s ({100.0 * _phase_decode / max(_t_total, 1e-9):.0f}%) + "
                   f"other {_phase_other:.1f}s")
        report_lines.append(_timing)
        print(f"[FunPackSceneChain] {_timing}")

        import json as _json
        final_frames = self._tensor_frames(self._latent_tensors(output)[0])
        status = (
            f"Scene chain complete: {scene_count} scene(s), "
            f"template={video_frames} latent frames, overlap={video_overlap}, output={final_frames}"
        )
        if carry_i2v_guides and carried_guide_frames > 0:
            status += f", i2v guide tokens={carried_guide_frames} latent frame(s)"
        overlap_diag = self._build_overlap_diagnostics(
            scene_count=scene_count,
            video_frames=video_frames,
            num_frames_per_scene=int(num_frames_per_scene),
            pixel_overlap=int(frame_overlap),
            latent_overlap=int(video_overlap),
            time_scale=time_scale,
            transition_duration=int(transition_duration),
            boundaries=boundary_entries,
            scene_runs=scene_runs,
            carry_i2v_guides=bool(carry_i2v_guides),
            mid_scene_guide=bool(mid_scene_guide),
            embed_guidance=bool(embed_guidance),
            embed_guidance_strength=float(embed_guidance_strength),
            embed_guidance_source=str(embed_guidance_source or "relative"),
        )
        try:
            from movie_editor.backend.chain_layout import scene_playback_layout
        except ImportError:
            try:
                from .movie_editor.backend.chain_layout import scene_playback_layout  # type: ignore
            except ImportError:
                scene_playback_layout = None
        if scene_playback_layout is not None:
            overlap_diag["scene_playback"] = scene_playback_layout(
                scene_count,
                fps=24.0,
                num_frames_per_scene=int(num_frames_per_scene),
                frame_overlap=int(frame_overlap),
                time_scale=time_scale,
                boundaries=boundary_entries,
            )
        if scene_count > 1 and int(frame_overlap) > 0:
            status += (
                f", overlap_blend={int(frame_overlap)}px"
                f" (scene N tail may show scene N+1 motion in last {int(frame_overlap)} frames)"
            )
        if embed_guidance:
            status += (
                f", embed_guidance={_eg_source}@{embed_guidance_strength}"
                f" (steers ALL scenes — not boundary-local)"
            )
        boundaries_out = overlap_diag
        if refinement_key_input:
            try:
                try:
                    from .conditioning import update_refinement_sampler_context
                except ImportError:
                    from conditioning import update_refinement_sampler_context
                update_refinement_sampler_context(refinement_key_input, {
                    "carry_i2v_guides": bool(carry_i2v_guides),
                    "frame_overlap": int(frame_overlap),
                    "transitions_enabled": scene_count > 1,
                    "joyai_memory": bool(joyai_memory),
                    "joyai_audio_memory": bool(joyai_audio_memory),
                    "carry_overlap_through_anchor": bool(carry_overlap_through_anchor),
                })
            except Exception as e:
                print(f"[FunPackLTXAVSceneChainSampler] Failed to write sampler context: {e}")

        # Snapshot the final output for output_guidance's NEXT training cycle (the rating that
        # scores THIS run pairs with this snapshot, not with denoised — see
        # _save_output_value_snapshot). Independent of the output_guidance toggle itself: a
        # snapshot from an unguided run is still valid training data. video_mask=None because
        # `output["samples"]` is the final unpacked latent (standard per-node layout), not the
        # sampler-internal packed AV tensor _packed_video_mask expects.
        # On LTX-AV the output is a NESTED tensor (video+audio) — the old plain-Tensor gate
        # silently skipped it, so the output value function NEVER received a sample on AV
        # ("not ready yet (needs 10+)" forever). Snapshot the video stream (largest tensor),
        # matching the video-only convention of the in-flight guidance path.
        if refinement_key_input and isinstance(output, dict):
            _snap = output.get("samples")
            if self._is_nested(_snap):
                _parts = [t for t in _snap.unbind() if isinstance(t, torch.Tensor) and t.numel() > 0]
                _snap = max(_parts, key=lambda t: t.numel()) if _parts else None
            if isinstance(_snap, torch.Tensor):
                self._save_output_value_snapshot(refinement_key_input, _snap, None)
                # DynaShift pending candidate: same run/rating pairing as the snapshot, but
                # the RAW video latent (fp16) — the rating decides whether it becomes a
                # negative-bank entry or is discarded (negative_memory.consume_pending).
                # Saved regardless of the dynashift toggle, same rationale as the snapshot:
                # a bad rating on an unguided run is still valid negative memory.
                try:
                    try:
                        from .negative_memory import save_pending as _save_neg_pending
                    except ImportError:
                        from negative_memory import save_pending as _save_neg_pending
                    _cond0 = positive[0][0] if isinstance(positive[0], (list, tuple)) else None
                    _save_neg_pending(refinement_key_input, _snap,
                                      _cond0 if isinstance(_cond0, torch.Tensor) else None)
                except Exception as _e:
                    print(f"[FunPackSceneChain] dynashift pending save failed: {_e}")

        return (output, images, status, scene_count, "\n".join(report_lines), _json.dumps(boundaries_out))

    # --- Batch Training -----------------------------------------------------
    def _batch_dir(self, refinement_key, stamp):
        # Store under ComfyUI's temp dir so batch artifacts are wiped on restart (ephemeral
        # rating material — useless once rated/learned) and are servable via /view?type=temp.
        import re as _re
        try:
            import folder_paths
            base = folder_paths.get_temp_directory()
        except Exception:
            base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refinements", "tmp")
        safe = _re.sub(r"[^A-Za-z0-9_.-]", "_", str(refinement_key or "default").strip() or "default")
        rel = os.path.join("funpack_batches", safe, stamp)
        d = os.path.join(base, rel)
        os.makedirs(d, exist_ok=True)
        return d, safe, rel

    def _split_batch_variants(self, positive):
        """If Studio packed shortcut variants into positive (each scene entry tagged with
        'funpack_batch_variant' = variant index), return a list of per-variant positive lists in
        variant order. None when there are no markers or only one variant (= normal seed-only)."""
        groups = {}
        for entry in positive:
            meta = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 and isinstance(entry[1], dict) else None
            vi = meta.get("funpack_batch_variant") if isinstance(meta, dict) else None
            if vi is None:
                return None
            try:
                groups.setdefault(int(vi), []).append(entry)
            except (TypeError, ValueError):
                return None
        if len(groups) <= 1:
            return None
        return [groups[k] for k in sorted(groups)]

    def _save_batch_preview(self, decoded, path, max_frames=16, width=256):
        """Save a decoded video tensor [T,H,W,C] in 0..1 as a downscaled animated webp."""
        try:
            from PIL import Image
            import numpy as _np
        except Exception:
            return False
        try:
            t = decoded.detach().float().clamp(0, 1).cpu()
            if t.dim() == 4 and t.shape[-1] not in (1, 3):
                t = t.permute(0, 2, 3, 1)  # [T,C,H,W] -> [T,H,W,C]
            frames = t.numpy()
            n = int(frames.shape[0])
            stride = max(1, n // max_frames)
            imgs = []
            for f in frames[::stride][:max_frames]:
                arr = (f * 255).astype(_np.uint8)
                if arr.shape[-1] == 1:
                    arr = arr.repeat(3, axis=-1)
                im = Image.fromarray(arr)
                if im.width > width:
                    im = im.resize((width, max(1, int(im.height * width / im.width))))
                imgs.append(im)
            if not imgs:
                return False
            imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=120, loop=0, format="WEBP")
            return True
        except Exception as e:
            print(f"[FunPackSceneChain] batch preview failed: {e}")
            return False

    def _decode_for_preview(self, vae, latent, decode_tile_size=0):
        video_tensor = self._latent_tensors(latent)[0]
        try:
            if decode_tile_size > 0:
                _tile = self._decode_tile_latent(vae, decode_tile_size)
                decoded = vae.decode_tiled(video_tensor, tile_x=_tile, tile_y=_tile)
            else:
                decoded = vae.decode(video_tensor)
        except Exception:
            decoded = vae.decode(video_tensor)
        if decoded.dim() == 5:
            b, tt, h, w, c = decoded.shape
            decoded = decoded.reshape(b * tt, h, w, c)
        return decoded

    def _run_batch_training(self, model, vae, positive, negative, sampler, sigmas, seed,
                            latent_template, num_frames_per_scene, frame_overlap, cfg, max_scenes,
                            use_same_seed, carry_i2v_guides, mid_scene_guide, mid_scene_guide_strength,
                            embed_guidance, embed_guidance_strength, transition_duration,
                            decode_tile_size, refinement_key_input, embed_guidance_source="relative",
                            score_slider=False, score_slider_strength=1.0,
                            joyai_memory=False, joyai_memory_size=7, joyai_fix_frames=3,
                            joyai_frame_select="center", joyai_memory_strength=0.3,
                            joyai_audio_memory=False, v2a_grad_scale=1.0,
                            alg_blur_guides=False, bounded_attention_enabled=False,
                            output_guidance=False, output_guidance_strength=0.02,
                            dynashift=False, dynashift_strength=0.3, dynashift_threshold=0.6,
                            alg_guide_blur_strength=2.0, alg_guide_blur_sigma_threshold=0.975,
                            alg_anchor=False, alg_anchor_strength=2.0,
                            alg_anchor_sigma_threshold=0.975,
                            plateau_cache=False, plateau_cache_threshold=0.975,
                            context_windows=False, context_window_length=145,
                            context_window_overlap=40, context_window_schedule="standard_uniform",
                            context_window_fuse="pyramid", context_window_freenoise=True,
                            context_window_retain_first=False,
                            cut_opening_frames=0,
                            second_pass=False, second_pass_op="none",
                            second_pass_sigmas=None, second_pass_sampler=None,
                            h3_audio_clock=False):
        """Sample one chain per Studio-packed variant entry (seed + index), persisting each result
        (latent + preview + per-entry cond + manifest) under ComfyUI temp for rating in Studio.
        Reuses sample() per entry with only the seed changed, so each entry is a clean generation."""
        import json as _json, time as _time
        key = str(refinement_key_input or "").strip()
        # persist=True -> save each variant to temp + manifest for rating in Studio (needs a key).
        # No key -> generate-only: run every variant and concat to IMAGES, but don't save/rate.
        # (NEVER recurse into self.sample() with the marked positive — that loops forever.)
        persist = bool(key)
        stamp = safe = rel = batch_dir = None
        if persist:
            stamp = _time.strftime("%Y-%m-%d_%H-%M-%S")
            batch_dir, safe, rel = self._batch_dir(key, stamp)
            scene_prompts = [self._scene_text(c, i) for i, c in enumerate(positive[:max(1, int(max_scenes))])]
            manifest = {"key": key, "created": stamp,
                        "subfolder": rel.replace(os.sep, "/"), "scene_prompts": scene_prompts, "items": []}
        else:
            print("[FunPackSceneChain] Batch (no refinement_key): generating all variants to IMAGES, "
                  "not saved for rating.")
            manifest = {"items": []}
        # Studio packed N entries into positive (each scene entry tagged 'funpack_batch_variant').
        # Sample one chain per entry. Entries may be genuinely different (shortcut variants) or
        # identical (seed-only batch — Studio packed N copies); either way each saves its own cond.
        variants = self._split_batch_variants(positive) or [positive]
        entries = list(enumerate(variants))   # [(idx, variant_positive_list), ...]
        manifest["iterations"] = len(entries)
        last = None
        batch_images = []  # collect each variant's decoded frames for the IMAGES output
        base_seed = int(seed)
        for idx, pos_i in entries:
            iter_seed = base_seed + idx
            print(f"[FunPackSceneChain] Batch Training {idx + 1}/{len(entries)}")
            out = self.sample(
                model, vae, pos_i, negative, sampler, sigmas, iter_seed, latent_template,
                num_frames_per_scene, frame_overlap, cfg, max_scenes, use_same_seed=use_same_seed,
                carry_i2v_guides=carry_i2v_guides, mid_scene_guide=mid_scene_guide,
                mid_scene_guide_strength=mid_scene_guide_strength, embed_guidance=embed_guidance,
                embed_guidance_strength=embed_guidance_strength, embed_guidance_source=embed_guidance_source,
                score_slider=score_slider, score_slider_strength=score_slider_strength,
                joyai_memory=joyai_memory, joyai_memory_size=joyai_memory_size,
                joyai_fix_frames=joyai_fix_frames, joyai_frame_select=joyai_frame_select,
                joyai_memory_strength=joyai_memory_strength,
                joyai_audio_memory=joyai_audio_memory, v2a_grad_scale=v2a_grad_scale,
                transition_duration=transition_duration,
                decode_tile_size=decode_tile_size, refinement_key_input=key,
                alg_blur_guides=alg_blur_guides,
                alg_guide_blur_strength=alg_guide_blur_strength,
                alg_guide_blur_sigma_threshold=alg_guide_blur_sigma_threshold,
                alg_anchor=alg_anchor, alg_anchor_strength=alg_anchor_strength,
                alg_anchor_sigma_threshold=alg_anchor_sigma_threshold,
                bounded_attention_enabled=bounded_attention_enabled,
                output_guidance=output_guidance, output_guidance_strength=output_guidance_strength,
                dynashift=dynashift, dynashift_strength=dynashift_strength,
                dynashift_threshold=dynashift_threshold,
                plateau_cache=plateau_cache, plateau_cache_threshold=plateau_cache_threshold,
                context_windows=context_windows, context_window_length=context_window_length,
                context_window_overlap=context_window_overlap,
                context_window_schedule=context_window_schedule,
                context_window_fuse=context_window_fuse,
                context_window_freenoise=context_window_freenoise,
                context_window_retain_first=context_window_retain_first,
                cut_opening_frames=cut_opening_frames,
                second_pass=second_pass, second_pass_op=second_pass_op,
                second_pass_sigmas=second_pass_sigmas,
                second_pass_sampler=second_pass_sampler,
                h3_audio_clock=h3_audio_clock,
                unique_id=None, prompt=None,
            )
            last = out
            # Collect this variant's decoded frames (CPU) so the IMAGES output shows ALL
            # videos back-to-back, not just the last one. The sub-call already decoded them.
            if isinstance(out[1], torch.Tensor) and out[1].dim() == 4 and out[1].shape[1] > 8:
                batch_images.append(out[1].detach().cpu())
            if not persist:
                continue  # generate-only: no temp save / manifest
            out_latent = out[0]
            iid = f"{safe}_{stamp}_{idx:02d}"
            latent_path = os.path.join(batch_dir, f"{iid}.latent.pt")
            preview_path = os.path.join(batch_dir, f"{iid}.webp")
            cond_name = None
            try:
                cond_i = pos_i[0][0] if pos_i and isinstance(pos_i[0], (list, tuple)) else None
                if isinstance(cond_i, torch.Tensor):
                    cond_name = f"{iid}.cond.pt"
                    torch.save(cond_i.detach().cpu(), os.path.join(batch_dir, cond_name))
            except Exception as e:
                print(f"[FunPackSceneChain] batch cond save failed: {e}")
            try:
                torch.save({"samples": self._latent_tensors(out_latent)[0].detach().cpu()}, latent_path)
            except Exception as e:
                print(f"[FunPackSceneChain] batch latent save failed: {e}")
                latent_path = None
            has_preview = False
            try:
                decoded = self._decode_for_preview(vae, out_latent, decode_tile_size)
                has_preview = self._save_batch_preview(decoded, preview_path)
                del decoded
            except Exception as e:
                print(f"[FunPackSceneChain] batch decode failed: {e}")
            manifest["items"].append({
                "index": idx, "id": iid, "seed": iter_seed,
                "variant": idx,
                "prompt": self._scene_text(pos_i[0], 0) if pos_i else None,
                "latent": os.path.basename(latent_path) if latent_path else None,
                "preview": os.path.basename(preview_path) if has_preview else None,
                "cond": cond_name,
                "guess_factor": (pos_i[0][1].get("funpack_guess_factor")
                                 if pos_i and isinstance(pos_i[0], (list, tuple))
                                 and len(pos_i[0]) > 1 and isinstance(pos_i[0][1], dict) else None),
                "rating": None,
            })
        if persist:
            manifest["iterations"] = len(manifest["items"])
            try:
                with open(os.path.join(batch_dir, "batch.json"), "w") as f:
                    _json.dump(manifest, f, indent=2)
            except Exception as e:
                print(f"[FunPackSceneChain] batch manifest save failed: {e}")
            status = (f"Batch Training complete: {len(manifest['items'])} generation(s) in "
                      f"temp/{rel.replace(os.sep, '/')} (cleared on restart) — rate them in Studio.")
        else:
            status = f"Batch (generate-only): {len(entries)} variant(s) → IMAGES; no key, not saved for rating."
        print(f"[FunPackSceneChain] {status}")
        if last is None:
            raise RuntimeError("Batch Training produced no output.")
        # IMAGES output = all variants' frames concatenated (so you can view the whole batch
        # without rating). Falls back to the last variant if shapes are incompatible.
        all_images = last[1]
        compatible = [im for im in batch_images if im.shape[1:] == batch_images[0].shape[1:]] if batch_images else []
        if len(compatible) > 1:
            try:
                all_images = torch.cat(compatible, dim=0)
                print(f"[FunPackSceneChain] Batch IMAGES: {len(compatible)} videos concatenated "
                      f"({all_images.shape[0]} frames total).")
            except Exception as e:
                print(f"[FunPackSceneChain] Batch IMAGES concat failed ({e}); returning last only.")
        elif compatible:
            all_images = compatible[0]
        return (last[0], all_images, status, last[3], last[4], last[5])
