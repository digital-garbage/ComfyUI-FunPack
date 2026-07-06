import copy
import hashlib
import math
import os
import time as _time

import torch

import comfy.k_diffusion.sampling as k_diffusion_sampling
import comfy.model_sampling
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils


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
    except Exception:
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
    except Exception:
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
    except Exception:
        return x_new


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


def _tag_scene_wrapper(wrapper, prev):
    """Mark a per-scene wrapper (and remember what it wrapped) so a later run can
    identify and unwind leaked ones."""
    setattr(wrapper, _FUNPACK_SCENE_WRAPPER_TAG, True)
    setattr(wrapper, "_funpack_prev_wrapper", prev)
    return wrapper


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
    except Exception:
        return None


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
                          normalize_strength=0.0, normalize_start_sigma=0.9):
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

    if not motion_pulse_steps:
        _, _, motion_pulse_steps, motion_pulse_noise = _prepare_dynamic_sigmas(
            sigmas, high_quality_pct, motion_pulse_mode, motion_pulse_start_pct,
            motion_pulse_count, motion_pulse_spacing_pct, motion_pulse_strength)
    motion_pulse_noise = max(0.0, float(motion_pulse_noise or 0.0))
    motion_step_noise = {
        int(item.get("step_index", -1)): max(0.0, float(item.get("noise", motion_pulse_noise)))
        for item in (motion_pulse_steps or []) if isinstance(item, dict)
    }

    late_start = _get_late_start_index(total_steps, high_quality_pct)
    quality_sigma_start = float(sigmas[late_start].item()) if late_start < sigmas.shape[0] else None
    num_quality_steps = total_steps - late_start

    s_in = x.new_ones([x.shape[0]])
    prev_denoised = None
    prev_h = None
    quality_step_index = 0
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
            x = denoised_eff
            continue

        if in_quality:
            # Deterministic Heun corrector (RF-correct 2nd order). Progressive blend:
            # first half of quality steps = euler, second half = full Heun, matching
            # the hybrid sampler's correction schedule. Note: to_d-based euler is
            # identical to the RF flow update, so this is consistent with the early phase.
            if num_quality_steps <= 1:
                effective_blend = correction_blend
            else:
                effective_blend = 0.0 if quality_step_index < (num_quality_steps // 2) else correction_blend
            dt = sigma_next - sigma
            d1 = k_diffusion_sampling.to_d(x, sigma, denoised_eff)
            if effective_blend > 0.0:
                x_pred = x + d1 * dt
                denoised_pred = model(x_pred, sigma_next * s_in, **extra_args)
                d2 = k_diffusion_sampling.to_d(x_pred, sigma_next, denoised_pred)
                d_use = d1 + effective_blend * ((d1 + d2) * 0.5 - d1)
            else:
                d_use = d1
            # Audio rides the plain euler direction (d1); only video gets the Heun correction.
            d_use = _video_only(d_use, d1, video_mask)
            x = x + d_use * dt
            # Heun changed x with a corrected direction; invalidate AB2 history.
            prev_denoised = None
            prev_h = None
            quality_step_index += 1
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
                x = _video_only(x_anc, x_det, video_mask)
            else:
                x = x_det

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
                                   normalize_start_sigma=0.9):
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


def sample_funpack_distilled_flow(model, x, sigmas, extra_args=None, callback=None,
                                   disable=None, order=2, s_noise=0.0,
                                   final_correction_steps=1, ab2_ramp=False,
                                   normalize_strength=0.0, normalize_start_sigma=0.9,
                                   velocity_bias_mode="off", velocity_bias_strength=0.0,
                                   velocity_bias_source="mean", velocity_refinement_key="default",
                                   rescue_mode=False, rescue_threshold=0.15, rescue_strength=0.2,
                                   rescue_prompt_sig=None,
                                   alg_enabled=False, alg_strength=2.0, alg_sigma_threshold=0.975,
                                   alg_guide_tail_frames=0,
                                   alg_guide_blur_strength=2.0, alg_guide_blur_sigma_threshold=0.975,
                                   mg_enabled=False, mg_strength=0.5, mg_decay=0.5, mg_sigma_threshold=0.975,
                                   quality_sharpness=0.0):
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
    alg_anchor_on = bool(alg_enabled)
    alg_tail_on = alg_guide_tail_frames > 0
    alg_sharp_latent_image = getattr(model, "latent_image", None) if (alg_anchor_on or alg_tail_on) else None
    alg_active = ((alg_anchor_on or alg_tail_on) and extra_args.get("denoise_mask") is not None
                  and alg_sharp_latent_image is not None)
    alg_latents = {}
    if alg_active:
        anchor_blurred = _alg_blur_frames(
            model, alg_sharp_latent_image, max(1.0, float(alg_strength)), frame_indices=(0,),
        ) if alg_anchor_on else None
        tail_kappa = max(1.0, float(alg_guide_blur_strength))
        tail_blurred = _alg_blur_frames(
            model, alg_sharp_latent_image, tail_kappa, tail_count=alg_guide_tail_frames,
        ) if alg_tail_on else None
        both_blurred = _alg_blur_frames(
            model, anchor_blurred, tail_kappa, tail_count=alg_guide_tail_frames,
        ) if (anchor_blurred is not None and tail_blurred is not None) else None
        alg_anchor_on = alg_anchor_on and anchor_blurred is not None
        alg_tail_on = alg_tail_on and tail_blurred is not None
        alg_latents = {
            (False, False): alg_sharp_latent_image,
            (True, False): anchor_blurred,
            (False, True): tail_blurred,
            (True, True): both_blurred if both_blurred is not None else (anchor_blurred or tail_blurred),
        }
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
        except Exception:
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
            if ab2_ramp and total_steps > 1:
                w = i / (total_steps - 1)  # 0 at first step -> 1 at last
                denoised_eff = denoised + (denoised_eff - denoised) * w

            # Audio rides plain 1st-order euler: keep the (ramped) AB2 estimate for video,
            # raw denoised for audio (2nd-order extrapolation corrupts the audio stream).
            denoised_eff = _video_only(denoised_eff, denoised, video_mask)

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
                # Audio rides the plain euler direction (d1); only video gets the Heun correction.
                d_use = _video_only((d1 + d2) / 2.0, d1, video_mask)
                d_use = _mg_step(d_use, video_mask)
                x = x + d_use * dt
                # Heun updates x differently; invalidate multistep history.
                prev_denoised = None
                prev_h = None
            else:
                d = k_diffusion_sampling.to_d(x, sigma, denoised_eff)
                d = _mg_step(d, video_mask)
                x = x + d * dt
                if s_noise > 0.0:
                    sigma_up = math.sqrt(max(0.0, float(sigma.item()) ** 2 - float(sigma_next.item()) ** 2))
                    if sigma_up > 0.0:
                        # Diversity noise on video only — ancestral-style noise corrupts audio.
                        x = _video_only(x + noise_sampler(sigma, sigma_next) * s_noise * sigma_up, x, video_mask)
    finally:
        if alg_active:
            model.latent_image = alg_sharp_latent_image

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

    def get_sampler(self, order=2, final_correction_steps=1, s_noise=0.0,
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
                    "default": 0.25, "min": 0.25, "max": 0.5, "step": 0.05,
                    "tooltip": "Guide attention strength for mid-scene anchor. 0.25 is the minimum — below that audio degrades and character appearance drifts. Above 0.35 causes spatial conflicts when scene composition shifts.",
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
                    "tooltip": "JoyAI-Echo cross-shot memory bank. Generalizes mid_scene_guide from one anchor to a managed set of clean prior-shot frames injected into each scene via LTX guide attention, so character/scene identity carries across the whole chain (JoyAI-Echo's story-level consistency). The first joyai_fix_frames scenes are pinned permanently as a global anchor; the rest is a rolling most-recent window capped at joyai_memory_size. Supersedes mid_scene_guide when on. Video memory only; pair it with joyai_audio_memory for the soundtrack.",
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
                    "tooltip": "JoyAI-Echo PAIRED AUDIO memory. Alongside each video memory frame, pin the prior shot's clean audio latent into the audio stream so voice/timbre/ambience carry across shots the way the face now does. Deliberately breaks the audio pass-through invariant — off by default. Requires joyai_memory on; no effect on single-stream (video-only) LTXV.",
                }),
                "v2a_grad_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.25,
                    "tooltip": "JoyAI-Echo video->audio coupling. Scales the model's trained video-to-audio cross-attention so the carried audio tracks the new shot's visuals (JoyAI uses 2.0). 1.0 = native model behavior (no change, zero overhead); 0.0 = audio ignores video this run. Only applies when joyai_audio_memory is on.",
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
                    "tooltip": "EXPERIMENTAL: extends ALG (see the sampler's alg_enabled) from just the i2v anchor to also blur newly-appended guide-attention frames this scene (mid_scene_guide / carry_i2v_guides-as-guide / configured per-scene guides / JoyAI memory), for the same early steps. Standalone: works even with the sampler's alg_enabled off (anchor stays sharp), with its own alg_guide_blur_strength / alg_guide_blur_sigma_threshold controls below. Requires the FunPack Distilled Flow sampler; no effect if no guide frames were appended this scene.",
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

    def _append_i2v_guides(self, chunk, template, positive, negative):
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
        tensors[1] = tensors[1][:, :, :-count]
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

    def _blend_tensors(self, left, right, overlap, use_slerp=True):
        if overlap <= 0:
            return torch.cat([left, right], dim=2)
        if left.shape[:2] != right.shape[:2] or left.shape[3:] != right.shape[3:]:
            raise ValueError("Cannot blend scene latents with different non-time dimensions.")
        right = right.to(left.device, left.dtype)
        left_ov = left[:, :, -overlap:]
        right_ov = right[:, :, :overlap]
        shape = [1] * left.dim()
        shape[2] = overlap
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
        return torch.cat([left[:, :, :-overlap], blended, right[:, :, overlap:]], dim=2)

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
            # index 0 is the video latent -> slerp+smoothstep; any further stream (audio) stays
            # on the untouched linear crossfade (audio-safety: never reshape audio nonlinearly).
            blended_tensors.append(self._blend_tensors(previous_tensors[index], tensor, overlap, use_slerp=(index == 0)))

        if self._is_nested(previous.get("samples")):
            result["samples"] = comfy.nested_tensor.NestedTensor(blended_tensors)
        else:
            result["samples"] = blended_tensors[0]
        result.pop("noise_mask", None)
        return result

    def _sample_chunk(self, model, sampler, sigmas, seed, cfg, positive, negative, latent,
                      pbar=None, step_offset=0, alg_guide_tail_frames=0,
                      alg_guide_blur_strength=2.0, alg_guide_blur_sigma_threshold=0.975,
                      bounded_attention_enabled=False):
        if sampler is None:
            raise ValueError("sampler input is required.")
        if not isinstance(sigmas, torch.Tensor):
            raise ValueError("sigmas input must be a SIGMAS tensor.")
        latent = self._clone_latent(latent)
        samples = latent["samples"]
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
        if isinstance(extra_options, dict) and sampler.sampler_function is sample_funpack_distilled_flow:
            extra_options["alg_guide_tail_frames"] = int(alg_guide_tail_frames)
            extra_options["alg_guide_blur_strength"] = float(alg_guide_blur_strength)
            extra_options["alg_guide_blur_sigma_threshold"] = float(alg_guide_blur_sigma_threshold)

        # EXPERIMENTAL Bounded Attention: model-level attention hooks (sampler-agnostic, unlike
        # the toggles above which only work on Distilled Flow), so install/remove here rather
        # than via extra_options. Cheap to attempt (no-ops fast without the right metadata).
        _ba_handles = self._install_bounded_attention(model, latent, positive) if bounded_attention_enabled else []

        try:
            sampled = comfy.sample.sample_custom(
                model, noise, float(cfg), sampler, sigmas, positive, negative, samples,
                noise_mask=latent.get("noise_mask"), seed=int(seed),
                callback=_progress_cb if pbar is not None else None,
            )
        finally:
            self._remove_bounded_attention(_ba_handles)
        latent["samples"] = sampled
        latent.pop("noise_mask", None)
        return latent

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
        except Exception:
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

    def _build_output_guidance_wrapper(self, model, value_fn, strength):
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
            scale = max(0.0, 1.0 - sigma * 2.0)  # same late-step ramp as embed_guidance
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

    def _build_dynashift_wrapper(self, model, negatives, strength, threshold):
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
        _desc_dim = 512
        _prep = {}  # device -> (desc [Tall,512] fp32, units [Tall,D] fp16, owner [Tall], conds)
        _warned = [False]

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
            cur = (c_dict or {}).get("c_crossattn")
            if cur is None or not any(isinstance(cv, torch.Tensor) for cv in conds):
                return torch.ones(len(conds), device=device)
            cm = cur.detach().float()
            cm = cm.mean(dim=tuple(range(cm.dim() - 1)))  # -> [D]
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
            ramp = max(0.0, 1.0 - sigma * 2.0)  # same late-step ramp as the other wrappers
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

    def _build_embed_guidance_wrapper(self, model, liked_dir, strength, value_fn=None):
        """Register a model_function_wrapper that nudges conditioning toward the
        liked quality direction at each denoising step. Uses value function gradient
        when available, falls back to the fixed liked direction otherwise."""
        old_wrapper = model.model_options.get("model_function_wrapper")
        fixed_dir = torch.nn.functional.normalize(liked_dir.float(), dim=-1)

        def _embed_wrapper(apply_fn, args, _ew=old_wrapper, _fixed=fixed_dir, _vf=value_fn, _s=strength):
            c = args.get("c") or {}
            cond = c.get("c_crossattn")
            if cond is not None:
                ts = args.get("timestep")
                try:
                    sigma = float(ts.max().item()) if ts is not None else 1.0
                except Exception:
                    sigma = 1.0
                scale = max(0.0, 1.0 - sigma * 2.0)
                if scale > 0:
                    if _vf is not None:
                        try:
                            grad = _vf.gradient(cond)
                            d = torch.nn.functional.normalize(grad.float(), dim=-1).to(cond.dtype)
                        except Exception as _e:
                            print(f"[FunPackSceneChain] embed_guidance: value function gradient failed ({_e}), using fixed direction")
                            d = _fixed.to(cond.device, cond.dtype).expand_as(cond)
                    else:
                        d = _fixed.to(cond.device, cond.dtype).expand_as(cond)
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

    def _build_score_slider_wrapper(self, model, liked_dir, eta, bad_dir=None):
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
        fixed_dir = torch.nn.functional.normalize(liked_dir.float(), dim=-1)
        bad_fixed = None
        if bad_dir is not None:
            bad_fixed = torch.nn.functional.normalize(bad_dir.float(), dim=-1)

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
            ramp = max(0.0, 1.0 - sigma * 2.0)  # base-only warmup at high sigma
            if cond is None or _eta == 0.0 or ramp <= 0.0:
                return _call(apply_fn, args)
            try:
                d = _fixed.to(cond.device, cond.dtype).expand_as(cond)
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
                    bd = _bad.to(cond.device, cond.dtype).expand_as(cond)
                    cond_minus = self._protect_audio(cond + bd * scale, cond)
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

    def _append_guide_latent(self, chunk, guide_frame, apply_at, strength, positive, negative, vae):
        """Append one guide latent frame with LTX guide attention at apply_at."""
        try:
            from comfy_extras.nodes_lt import LTXVAddGuide, _append_guide_attention_entry
        except ImportError:
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

    def _load_image_tensor(self, filename):
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
            img = Image.open(path).convert("RGB")
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

    def _build_mixed_anchor_chunk(self, vae, anchor_meta, latent_template, previous, video_overlap):
        """Mixed source: Img2Video anchor latent only — prior scene overlap is not used."""
        filename = (anchor_meta or {}).get("filename")
        strength = float((anchor_meta or {}).get("strength", 1.0))
        image = self._load_image_tensor(filename) if filename else None
        if image is None:
            if previous is None:
                return self._clone_latent(latent_template)
            return self._build_continuation_chunk(latent_template, previous, video_overlap)
        base = self._clone_latent(latent_template)
        return self._apply_img2video_to_video_latent(vae, image, base, strength)

    def _encode_image_guide_frame(self, filename, vae, ref_tensor):
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
                                 scene_outputs, scene_media_by_ref, positive, negative, vae):
        head_crop = 0
        tail_crop = 0
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
                chunk, positive, negative, head, tail = self._append_media_guide_at(
                    chunk, fn, frame_idx, apply_at, strength, positive, negative, vae,
                )
            else:
                continue
            head_crop += head
            tail_crop += tail
        return chunk, positive, negative, head_crop, tail_crop

    def _append_mid_scene_guide(self, chunk, previous_output, positive, negative, vae, strength):
        """Append the middle frame of the previous scene as a guide for the current chunk
        using LTX's guide attention mechanism (keyframe_idxs + guide_attention_entries).
        Audio-safe: appends only to the video tensor, guide tokens influence denoising
        through attention weights rather than overwriting hidden states."""
        try:
            from comfy_extras.nodes_lt import LTXVAddGuide, _append_guide_attention_entry
        except ImportError:
            return chunk, positive, negative, 0

        prev_tensors = self._latent_tensors(previous_output)
        chunk_tensors = self._latent_tensors(chunk)
        if not prev_tensors or not chunk_tensors:
            return chunk, positive, negative, 0

        # Middle frame of previous scene as guide source
        F_prev = self._tensor_frames(prev_tensors[0])
        guide_frame = self._time_slice(prev_tensors[0], F_prev // 2, F_prev // 2 + 1)
        guide_frame = guide_frame.to(device=chunk_tensors[0].device, dtype=chunk_tensors[0].dtype)

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
        s = max(0.25, float(strength))  # same audio-safe floor as mid_scene_guide
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
        F = self._tensor_frames(tensors[1])
        if F <= 0:
            return None
        if select == "first":
            idx = 0
        elif select == "random":
            idx = int(torch.randint(0, F, (1,)).item())
        else:
            idx = F // 2
        return self._time_slice(tensors[1], idx, idx + 1).detach()

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
        appended = 0
        for af in frames:
            af = af.to(device=audio.device, dtype=audio.dtype)
            # Only append shape-compatible frames (channels + spatial dims must match the stream).
            if af.shape[1] != audio.shape[1] or af.shape[3:] != audio.shape[3:]:
                continue
            clean = torch.zeros(
                amask.shape[0], amask.shape[1], af.shape[2], *amask.shape[3:],
                device=amask.device, dtype=amask.dtype,
            )
            audio = torch.cat([audio, af], dim=2)
            amask = torch.cat([amask, clean], dim=2)
            appended += int(af.shape[2])
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

    def _bounded_attention_region_mask(self, t, h, w, device):
        """[T*H*W] region id per video token: 0 = left half (by width), 1 = right half.
        Assumes (t, h, w) row-major flattening — matches the packed-latent layout
        _alg_blur_frames already relies on; LTX's patchify is 1:1 so the transformer's
        token order should match. Best-effort for an experimental feature, not guaranteed
        if that assumption ever changes upstream."""
        idx = torch.arange(t * h * w, device=device)
        w_idx = idx % w
        return (w_idx >= (w // 2)).long()

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
            except Exception:
                pass

    def _vae_with_decode_noise(self, vae, timestep, scale, seed):
        """Return a shallow copy of the VAE stamped with LTX decode-time noise settings so its
        internal decoder restores fine detail/grain. Never mutates the shared input VAE. Mirrors
        LTXV's 'Set VAE Decoder Noise', but owned by the Chain Sampler (it does the decode)."""
        try:
            result = copy.copy(vae)
        except Exception:
            return vae
        if hasattr(result, "first_stage_model"):
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
               unique_id=None, prompt=None):
        if not isinstance(positive, list) or not positive:
            raise ValueError("positive conditioning must contain at least one scene entry.")
        if negative is None:
            negative = []

        # Defensively strip any enhancement block hooks left on the shared diffusion
        # model by a previous run (build_enhancements only removes them on scene
        # transitions, not at end-of-sampling). This covers runs that don't go through
        # build_enhancements, so stale hooks can't fire on an unenhanced generation.
        try:
            try:
                from .ltx_enhancements import strip_funpack_block_hooks
            except ImportError:
                from ltx_enhancements import strip_funpack_block_hooks
            strip_funpack_block_hooks(model)
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
                bounded_attention_enabled=bounded_attention_enabled,
                output_guidance=output_guidance, output_guidance_strength=output_guidance_strength,
                dynashift=dynashift, dynashift_strength=dynashift_strength,
                dynashift_threshold=dynashift_threshold,
            )

        max_scene_count = max(1, int(max_scenes))
        scene_conditionings = positive[:max_scene_count]
        scene_count = len(scene_conditionings)
        time_scale = self._time_scale(vae)
        video_frames = self._validate_template_length(latent_template, num_frames_per_scene, time_scale)
        video_overlap = self._overlap_frames(latent_template, frame_overlap, time_scale)

        output = None
        report_lines = []
        carried_guide_frames = 0
        boundary_entries = []
        cumulative_latent_frames = 0

        # Load liked direction once for embed_guidance. Source selects which learned direction:
        # 'relative' = this prompt's key; 'absolute' = the global, prompt-agnostic taste store
        # (keyless, so it works even without refinement_key_input).
        _liked_dir = None
        _bad_dir = None
        _value_fn = None
        _eg_source = str(embed_guidance_source or "relative").lower()
        _eg_key = self._absolute_key() if _eg_source == "absolute" else refinement_key_input
        if (embed_guidance or score_slider) and _eg_key:
            _liked_dir = self._load_liked_direction(_eg_key)
            if _liked_dir is None:
                print(f"[FunPackSceneChain] taste steering ({_eg_source}): no liked direction found (need 3+ liked generations)")
            else:
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

        steps_per_scene = max(1, int(len(sigmas)) - 1)
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

        for scene_index, scene_cond in enumerate(scene_conditionings):
            scene_positive = [scene_cond]
            scene_negative = negative

            provided_seed = self._scene_seed(scene_cond)
            if use_same_seed:
                scene_seed = first_scene_seed
            else:
                scene_seed = provided_seed if provided_seed is not None else int(seed) + scene_index
            carried = 0
            soft_carried = 0
            guide_tail = 0
            audio_tail = 0
            run_mechanisms: list = []
            anchor_meta = (scene_anchors or {}).get(str(scene_index))
            if output is None:
                chunk = self._clone_latent(latent_template)
                custom_guides = None
                if per_scene_guides and scene_index < len(per_scene_guides):
                    custom_guides = per_scene_guides[scene_index]
                if custom_guides:
                    run_mechanisms.append("custom_guide_stack")
                    chunk, scene_positive, scene_negative, carried, guide_tail = self._apply_configured_guides(
                        chunk, scene_index, custom_guides, latent_template, scene_outputs, scene_media_by_ref,
                        scene_positive, scene_negative, vae,
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
                    vae, anchor_meta, latent_template, output, 0,
                )
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
                chunk = self._build_continuation_chunk(latent_template, output, video_overlap)
                if video_overlap == 0:
                    chunk, soft_carried = self._prepend_soft_continuation(chunk, output)
                    if soft_carried > 0:
                        run_mechanisms.append(f"soft_continuation({soft_carried})")
                elif video_overlap > 0:
                    run_mechanisms.append(f"latent_overlap({frame_overlap}px)")
                custom_guides = None
                if per_scene_guides and scene_index < len(per_scene_guides):
                    custom_guides = per_scene_guides[scene_index]
                if custom_guides:
                    run_mechanisms.append("custom_guide_stack")
                    chunk, scene_positive, scene_negative, carried, guide_tail = self._apply_configured_guides(
                        chunk, scene_index, custom_guides, latent_template, scene_outputs, scene_media_by_ref,
                        scene_positive, scene_negative, vae,
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

            # Everything from here through sampling installs per-scene state on the SHARED
            # model (function wrappers, forward hooks). One snapshot + one finally guarantees
            # the model leaves this scene exactly as it entered it — even on interrupt/OOM
            # mid-sampling, where the old per-feature unwind blocks were never reached and
            # the wrappers leaked in-process, double-steering every later run (same failure
            # mode as the block-hook leak; see _strip_funpack_scene_wrappers).
            _scene_base_wrapper = model.model_options.get("model_function_wrapper")
            _v2a_handles = []
            try:
                if embed_guidance and _value_fn is not None and _value_fn.is_ready():
                    run_mechanisms.append("embed_guidance_vf_ascend")
                    orig_cond, orig_extra = scene_positive[0][0], scene_positive[0][1]
                    ascended = self._protect_audio(_value_fn.ascend(orig_cond), orig_cond)
                    scene_positive = [[ascended, orig_extra]] + list(scene_positive[1:])
                if embed_guidance and _liked_dir is not None:
                    run_mechanisms.append(f"embed_guidance({_eg_source},{embed_guidance_strength})")
                    self._build_embed_guidance_wrapper(model, _liked_dir, embed_guidance_strength, value_fn=_value_fn)
                if score_slider and _liked_dir is not None:
                    _pole = "contrastive" if _bad_dir is not None else "symmetric"
                    run_mechanisms.append(f"score_slider({_eg_source},{score_slider_strength},{_pole})")
                    self._build_score_slider_wrapper(model, _liked_dir, score_slider_strength, bad_dir=_bad_dir)
                if dynashift and _dynashift_negatives:
                    run_mechanisms.append(
                        f"dynashift({len(_dynashift_negatives)}neg,{dynashift_strength},thr={dynashift_threshold})")
                    self._build_dynashift_wrapper(
                        model, _dynashift_negatives, dynashift_strength, dynashift_threshold)
                if output_guidance and _output_value_fn is not None:
                    # Installed outermost (after embed_guidance/score_slider/dynashift) so it
                    # corrects whatever prediction those already produced, not the raw base one.
                    run_mechanisms.append(f"output_guidance({output_guidance_strength})")
                    self._build_output_guidance_wrapper(model, _output_value_fn, output_guidance_strength)
                # Per-scene temporal style (auto / pulse): layer a frame_rate wrapper on top
                # of whatever is installed (e.g. embed guidance).
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
                _t_sample0 = _time.perf_counter()
                sampled = self._sample_chunk(
                    model, sampler, sigmas, scene_seed, cfg, scene_positive, scene_negative, chunk,
                    pbar=pbar, step_offset=scene_index * steps_per_scene,
                    alg_guide_tail_frames=(guide_tail if (alg_blur_guides and guide_tail > 0) else 0),
                    alg_guide_blur_strength=alg_guide_blur_strength,
                    alg_guide_blur_sigma_threshold=alg_guide_blur_sigma_threshold,
                    bounded_attention_enabled=bounded_attention_enabled,
                )
                _scene_sample_s = _time.perf_counter() - _t_sample0
                _phase_sampling += _scene_sample_s
            finally:
                self._remove_v2a_scale(_v2a_handles)
                if model.model_options.get("model_function_wrapper") is not _scene_base_wrapper:
                    if _scene_base_wrapper is not None:
                        model.model_options["model_function_wrapper"] = _scene_base_wrapper
                    else:
                        model.model_options.pop("model_function_wrapper", None)
            if carried + soft_carried > 0:
                sampled = self._crop_video_head(sampled, carried + soft_carried)
            if guide_tail > 0:
                sampled = self._crop_video_tail(sampled, guide_tail)
            if audio_tail > 0:
                sampled = self._crop_audio_tail(sampled, audio_tail)
            if joyai_bank is not None:
                # Harvest from the clean, fully-cropped scene so injected memory tails never re-enter
                # the bank. Scene 0 seeds the pinned anchor (num_fix); later scenes roll in. The audio
                # half is harvested only when audio memory is on, else stored as None (video-only).
                v_frame = self._harvest_joyai_frame(sampled, joyai_frame_select)
                a_frame = self._harvest_joyai_audio(sampled, joyai_frame_select) if joyai_audio_memory else None
                joyai_bank.add(v_frame, a_frame)
            scene_outputs.append(self._clone_latent(sampled))
            blend_overlap = 0 if anchor_meta else video_overlap
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
        want_image = self._output_connected(prompt, unique_id, 1)

        images = None
        if want_image:
            _t_dec0 = _time.perf_counter()
            video_tensor = self._latent_tensors(output)[0]
            if decode_tile_size > 0:
                try:
                    decoded = vae.decode_tiled(video_tensor, tile_x=decode_tile_size // 8, tile_y=decode_tile_size // 8)
                except Exception:
                    decoded = vae.decode(video_tensor)
            else:
                decoded = vae.decode(video_tensor)
            if decoded.dim() == 5:
                b, t, h, w, c = decoded.shape
                decoded = decoded.reshape(b * t, h, w, c)
            images = self._apply_transitions_pixel(decoded, boundary_entries, transition_duration)
            _phase_decode = _time.perf_counter() - _t_dec0

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
                decoded = vae.decode_tiled(video_tensor, tile_x=decode_tile_size // 8, tile_y=decode_tile_size // 8)
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
                            alg_guide_blur_strength=2.0, alg_guide_blur_sigma_threshold=0.975):
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
                bounded_attention_enabled=bounded_attention_enabled,
                output_guidance=output_guidance, output_guidance_strength=output_guidance_strength,
                dynashift=dynashift, dynashift_strength=dynashift_strength,
                dynashift_threshold=dynashift_threshold,
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
