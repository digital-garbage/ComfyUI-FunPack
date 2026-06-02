import hashlib
import math
import os

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
                          quality_sharpness=0.0, velocity_bias_source="mean"):
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

    # Audio-safe sampling: on a packed LTXAV latent, keep ancestral noise + steering on the
    # video stream and let audio ride the clean deterministic flow (ancestral re-noising
    # corrupts audio). None for single-stream LTXV -> all of this is a no-op.
    video_mask = _packed_video_mask(model, x)
    if video_mask is not None:
        n_aud = int((video_mask < 0.5).sum().item())
        print(f"[FunPack AV] packed audio+video latent detected -> audio-safe sampling "
              f"(audio held deterministic on {n_aud} of {video_mask.shape[-1]} packed dims)")

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
                    sigmas=None, eta_final=1.0):
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
                "transition_duration": ("INT", {
                    "default": 16, "min": 0, "max": 128, "step": 2,
                    "tooltip": "Extra pixel frames of fade beyond the blend zone on each side of a scene boundary. 0 = disable all transition effects.",
                }),
            },
            "optional": {
                "decode_tile_size": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 64,
                    "tooltip": "Tile size for VAE decode (0 = no tiling). Set to e.g. 512 if decode OOMs.",
                }),
                "refinement_key_input": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "forceInput": True,
                    "tooltip": "Connect to the same refinement key as your V2 Refiner. When wired, the sampler writes carry_i2v_guides, frame_overlap, and scene count into the refinement state so the Refiner can reason about what changed between rated runs.",
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
            model, noise, float(cfg), sampler, sigmas, positive, negative, samples,
            noise_mask=latent.get("noise_mask"), seed=int(seed),
        )
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

    def _load_liked_direction(self, refinement_key):
        """Read the liked conditioning direction from the Refiner's state file."""
        try:
            try:
                from .conditioning import refinement_state_path, serializable_to_tensor
            except ImportError:
                from conditioning import refinement_state_path, serializable_to_tensor
            import json as _json
            path = refinement_state_path(refinement_key, "clip", prefix="refine_v2")
            with open(path, "r", encoding="utf-8") as f:
                state = _json.load(f)
            global_state = state.get("global", state)  # liked_dir lives under state["global"]
            liked_dir_slot = global_state.get("liked_dir", {})
            if int(liked_dir_slot.get("direction_count", 0)) < 3:
                return None
            raw = liked_dir_slot.get("direction")
            if raw is None:
                return None
            return serializable_to_tensor(raw)
        except Exception:
            return None

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
                    new_c["c_crossattn"] = cond + (_s * scale) * d
                    args = dict(args)
                    args["c"] = new_c
            if _ew is not None:
                return _ew(apply_fn, args)
            return apply_fn(args["input"], args["timestep"], **args.get("c", {}))

        model.model_options["model_function_wrapper"] = _embed_wrapper
        return old_wrapper

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


    def sample(self, model, vae, positive, negative, sampler, sigmas, seed, latent_template,
               num_frames_per_scene, frame_overlap, cfg, max_scenes, use_same_seed=False,
               carry_i2v_guides=False,
               mid_scene_guide=False, mid_scene_guide_strength=0.4,
               embed_guidance=False, embed_guidance_strength=0.02,
               transition_duration=16, decode_tile_size=0,
               refinement_key_input="", unique_id=None, prompt=None):
        if not isinstance(positive, list) or not positive:
            raise ValueError("positive conditioning must contain at least one scene entry.")
        if negative is None:
            negative = []

        # Batch Training: Studio (the hub) packs N conditionings into positive, each scene entry
        # tagged 'funpack_batch_variant'. That marker is the only trigger — the sampler has no
        # batch-count input. Sample one chain per packed entry, persist each for rating in Studio.
        if self._split_batch_variants(positive) is not None:
            return self._run_batch_training(
                model, vae, positive, negative, sampler, sigmas, seed, latent_template,
                num_frames_per_scene, frame_overlap, cfg, max_scenes, use_same_seed,
                carry_i2v_guides, mid_scene_guide, mid_scene_guide_strength,
                embed_guidance, embed_guidance_strength, transition_duration,
                decode_tile_size, refinement_key_input,
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

        # Load liked direction once for embed_guidance
        _liked_dir = None
        _value_fn = None
        if embed_guidance and refinement_key_input:
            _liked_dir = self._load_liked_direction(refinement_key_input)
            if _liked_dir is None:
                print("[FunPackSceneChain] embed_guidance: no liked direction found (need 3+ liked generations)")
            else:
                _value_fn = self._load_value_function(refinement_key_input)
                if _value_fn:
                    ready = _value_fn.is_ready()
                    mode = f"value function ({_value_fn.n_trained} samples, ascent {'on' if ready else 'pending'})"
                else:
                    mode = "fixed direction"
                print(f"[FunPackSceneChain] embed_guidance: active via {mode}, strength={embed_guidance_strength}")

        first_scene_seed = self._scene_seed(scene_conditionings[0])
        if first_scene_seed is None:
            first_scene_seed = int(seed)
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
            if output is None:
                chunk = self._clone_latent(latent_template)
            else:
                # Record boundary before blending
                effect = self._scene_transition_effect(scene_cond)
                if effect and transition_duration > 0:
                    boundary_latent = cumulative_latent_frames
                    boundary_pixel = int((boundary_latent - 1) * time_scale + 1) if time_scale > 1 else boundary_latent
                    boundary_entries.append({
                        "boundary_latent": boundary_latent,
                        "pixel_frame": max(0, boundary_pixel),
                        "effect": effect,
                    })
                chunk = self._build_continuation_chunk(latent_template, output, video_overlap)
                if video_overlap == 0:
                    chunk, soft_carried = self._prepend_soft_continuation(chunk, output)
                if carry_i2v_guides:
                    chunk, scene_positive, scene_negative, carried = self._append_i2v_guides(
                        chunk, latent_template, scene_positive, scene_negative,
                    )
                    carried_guide_frames = max(carried_guide_frames, carried)
                if mid_scene_guide:
                    chunk, scene_positive, scene_negative, guide_tail = self._append_mid_scene_guide(
                        chunk, output, scene_positive, scene_negative, vae, mid_scene_guide_strength,
                    )
                else:
                    guide_tail = 0

            if embed_guidance and _value_fn is not None and _value_fn.is_ready():
                orig_cond, orig_extra = scene_positive[0][0], scene_positive[0][1]
                ascended = _value_fn.ascend(orig_cond)
                scene_positive = [[ascended, orig_extra]] + list(scene_positive[1:])
            if embed_guidance and _liked_dir is not None:
                _eg_old_wrapper = self._build_embed_guidance_wrapper(model, _liked_dir, embed_guidance_strength, value_fn=_value_fn)
            sampled = self._sample_chunk(
                model, sampler, sigmas, scene_seed, cfg, scene_positive, scene_negative, chunk,
            )
            if embed_guidance and _liked_dir is not None:
                if _eg_old_wrapper is not None:
                    model.model_options["model_function_wrapper"] = _eg_old_wrapper
                elif "model_function_wrapper" in model.model_options:
                    del model.model_options["model_function_wrapper"]
            if carried + soft_carried > 0:
                sampled = self._crop_video_head(sampled, carried + soft_carried)
            if guide_tail > 0:
                sampled = self._crop_video_tail(sampled, guide_tail)
            output = sampled if output is None else self._blend_latents(output, sampled, video_overlap)
            cumulative_latent_frames = self._tensor_frames(self._latent_tensors(output)[0])
            report_lines.append(f"Scene {scene_index + 1}: seed={scene_seed}, text={self._scene_text(scene_cond, scene_index)}")

        del scene_cond, scene_positive, scene_negative, scene_conditionings, chunk, sampled

        # RETURN_TYPES slot indices: 0=latent, 1=images, 2=status...
        # Sampling is fully complete. Latent is untouched and returned as-is.
        # IMAGES: decode the whole latent in one pass, then apply transition effects.
        want_image = self._output_connected(prompt, unique_id, 1)

        images = None
        if want_image:
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

        if images is None:
            images = torch.zeros(1, 8, 8, 3)

        import json as _json
        final_frames = self._tensor_frames(self._latent_tensors(output)[0])
        status = (
            f"Scene chain complete: {scene_count} scene(s), "
            f"template={video_frames} latent frames, overlap={video_overlap}, output={final_frames}"
        )
        if carry_i2v_guides and carried_guide_frames > 0:
            status += f", i2v guide tokens={carried_guide_frames} latent frame(s)"
        boundaries_out = [{"pixel_frame": e["pixel_frame"], "effect": e["effect"]} for e in boundary_entries]
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
                })
            except Exception as e:
                print(f"[FunPackLTXAVSceneChainSampler] Failed to write sampler context: {e}")

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
                            decode_tile_size, refinement_key_input):
        """Sample one chain per Studio-packed variant entry (seed + index), persisting each result
        (latent + preview + per-entry cond + manifest) under ComfyUI temp for rating in Studio.
        Reuses sample() per entry with only the seed changed, so each entry is a clean generation."""
        import json as _json, time as _time
        key = str(refinement_key_input or "").strip()
        if not key:
            print("[FunPackSceneChain] Batch Training requires refinement_key_input; running once instead.")
            return self.sample(
                model, vae, positive, negative, sampler, sigmas, seed, latent_template,
                num_frames_per_scene, frame_overlap, cfg, max_scenes, use_same_seed=use_same_seed,
                carry_i2v_guides=carry_i2v_guides, mid_scene_guide=mid_scene_guide,
                mid_scene_guide_strength=mid_scene_guide_strength, embed_guidance=embed_guidance,
                embed_guidance_strength=embed_guidance_strength, transition_duration=transition_duration,
                decode_tile_size=decode_tile_size, refinement_key_input="",
            )
        stamp = _time.strftime("%Y-%m-%d_%H-%M-%S")
        batch_dir, safe, rel = self._batch_dir(key, stamp)
        scene_prompts = [self._scene_text(c, i) for i, c in enumerate(positive[:max(1, int(max_scenes))])]
        manifest = {"key": key, "created": stamp,
                    "subfolder": rel.replace(os.sep, "/"), "scene_prompts": scene_prompts, "items": []}
        # Studio packed N entries into positive (each scene entry tagged 'funpack_batch_variant').
        # Sample one chain per entry. Entries may be genuinely different (shortcut variants) or
        # identical (seed-only batch — Studio packed N copies); either way each saves its own cond.
        variants = self._split_batch_variants(positive) or [positive]
        entries = list(enumerate(variants))   # [(idx, variant_positive_list), ...]
        manifest["iterations"] = len(entries)
        last = None
        base_seed = int(seed)
        for idx, pos_i in entries:
            iter_seed = base_seed + idx
            print(f"[FunPackSceneChain] Batch Training {idx + 1}/{len(entries)}")
            out = self.sample(
                model, vae, pos_i, negative, sampler, sigmas, iter_seed, latent_template,
                num_frames_per_scene, frame_overlap, cfg, max_scenes, use_same_seed=use_same_seed,
                carry_i2v_guides=carry_i2v_guides, mid_scene_guide=mid_scene_guide,
                mid_scene_guide_strength=mid_scene_guide_strength, embed_guidance=embed_guidance,
                embed_guidance_strength=embed_guidance_strength, transition_duration=transition_duration,
                decode_tile_size=decode_tile_size, refinement_key_input=key,
                unique_id=None, prompt=None,
            )
            last = out
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
                "rating": None,
            })
        try:
            with open(os.path.join(batch_dir, "batch.json"), "w") as f:
                _json.dump(manifest, f, indent=2)
        except Exception as e:
            print(f"[FunPackSceneChain] batch manifest save failed: {e}")
        status = (f"Batch Training complete: {manifest['iterations']} generation(s) in "
                  f"temp/{rel.replace(os.sep, '/')} (cleared on restart) — rate them in Studio.")
        print(f"[FunPackSceneChain] {status}")
        if last is None:
            raise RuntimeError("Batch Training produced no output.")
        return (last[0], last[1], status, last[3], last[4], last[5])
