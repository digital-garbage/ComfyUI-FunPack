"""
LTX-V model enhancements for FunPack - applied automatically via the refiner rating.

Technique 2: Per-block attention temperature
  Sharp semantic blocks when quality/concept is missing, looser early blocks when details missing.

Technique 3: Temporal RoPE style
  Manipulate frame_rate in the model's positional embedding to change motion character.
  Styles: natural / accelerate / decelerate / loop / freeze

Technique 4: Denoise creativity mask
  Spatial noise weighting derived from latent variance: high-variance regions get more freedom.

Technique 5: Attention anchor transfer
  Capture hidden states at semantic anchor blocks (14, 19) during every run.
  When a run is rated Perfect, bless those maps. Inject them into subsequent runs
  to transfer semantic structure from good outputs onto future generations.
"""

import math
import os
from hashlib import md5

import torch

TEMPORAL_STYLES = ["natural", "accelerate", "decelerate", "loop", "freeze"]

# Confirmed semantic focal points (PAG default=14, STG defaults=14,19)
ANCHOR_BLOCKS = [14, 19]

# Block zones for temperature mapping (normalized for 48-block LTXAV)
_ZONE_EARLY = set(range(0, 14))       # texture / low-level noise
_ZONE_SEMANTIC = frozenset({14, 19})  # primary semantic anchors
_ZONE_CONCEPT = set(range(20, 36))    # concept formation / identity
_ZONE_LATE = set(range(36, 48))       # high-level refinement


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _maps_dir():
    base = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(base, "refinements", "attn_maps")
    os.makedirs(d, exist_ok=True)
    return d


def _safe_key(refinement_key):
    return md5(f"attn::{refinement_key}".encode()).hexdigest()[:16]


def _temp_maps_path(refinement_key):
    return os.path.join(_maps_dir(), f"temp_{_safe_key(refinement_key)}.pt")


def _blessed_maps_path(refinement_key):
    return os.path.join(_maps_dir(), f"blessed_{_safe_key(refinement_key)}.pt")


def _creativity_latent_path(refinement_key):
    return os.path.join(_maps_dir(), f"creativity_latent_{_safe_key(refinement_key)}.pt")


# ---------------------------------------------------------------------------
# Creativity latent save / load
# ---------------------------------------------------------------------------

def save_creativity_latent(latent, refinement_key):
    """
    Save a video latent for creativity masking. Called by Save Refinement Latent node.
    Audio latents and combined AV latents are skipped - video samples only.
    """
    if not refinement_key:
        return False
    if not isinstance(latent, dict):
        return False
    if latent.get("type") == "audio":
        return False
    samples = latent.get("samples")
    if not isinstance(samples, torch.Tensor) or samples.dim() not in (4, 5):
        return False
    try:
        torch.save({"samples": samples.detach().cpu()}, _creativity_latent_path(refinement_key))
        return True
    except Exception as e:
        print(f"[FunPackEnhancements] Save creativity latent failed: {e}")
        return False


def load_and_apply_creativity_mask(refinement_key, rating_profile, reward, latent=None):
    """
    Computes creativity mask and returns noise-modified latent dict or None.
    Source priority: connected latent arg > saved file by key > None.
    At high reward the global floor is 0 and only gentle spatial variance applies.
    """

    # Resolve source latent
    samples = None
    if isinstance(latent, dict) and latent.get("type") != "audio":
        s = latent.get("samples")
        if isinstance(s, torch.Tensor) and s.dim() in (4, 5):
            samples = s.detach().cpu()

    if samples is None and refinement_key:
        path = _creativity_latent_path(refinement_key)
        if os.path.exists(path):
            try:
                saved = torch.load(path, map_location="cpu", weights_only=True)
                s = saved.get("samples")
                if isinstance(s, torch.Tensor) and s.dim() in (4, 5):
                    samples = s
            except Exception as e:
                print(f"[FunPackEnhancements] Load creativity latent failed: {e}")

    if samples is None:
        return None

    try:
        mask = build_creativity_mask({"samples": samples}, rating_profile, reward)
        if mask is None:
            return None

        latent_std = float(samples.std().clamp_min(1e-8).item())
        noise = torch.randn_like(samples)
        noise_scale = (mask * latent_std).unsqueeze(1)  # broadcast over channel dim
        return {"samples": samples + noise * noise_scale}
    except Exception as e:
        print(f"[FunPackEnhancements] Apply creativity mask failed: {e}")
        return None


def clear_refinement_data(refinement_key):
    """Remove all enhancement files for a key. Called on reset_session."""
    if not refinement_key:
        return
    for path in (
        _temp_maps_path(refinement_key),
        _blessed_maps_path(refinement_key),
        _creativity_latent_path(refinement_key),
    ):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"[FunPackEnhancements] Cleanup failed for {path}: {e}")


def bless_attention_maps(refinement_key):
    """Promote temp maps to blessed. Call when user rates a generation Perfect."""
    src = _temp_maps_path(refinement_key)
    dst = _blessed_maps_path(refinement_key)
    if not os.path.exists(src):
        return False
    try:
        data = torch.load(src, map_location="cpu", weights_only=True)
        torch.save(data, dst)
        print(f"[FunPackEnhancements] Blessed attention maps for key '{refinement_key}'")
        return True
    except Exception as e:
        print(f"[FunPackEnhancements] Bless failed: {e}")
        return False


def _load_blessed_maps(refinement_key):
    path = _blessed_maps_path(refinement_key)
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"[FunPackEnhancements] Load blessed maps failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Temperature map
# ---------------------------------------------------------------------------

def _derive_temperature_map(rating_profile, reward):
    """
    Returns {block_idx: temperature} for blocks that should deviate from 1.0.
      temperature < 1.0  sharper/colder - confident, focused decisions
      temperature > 1.0  softer/warmer  - more varied, exploratory
    Returns {} when previous output was good (reward >= 0.8).
    """
    if reward >= 0.8:
        return {}

    quality = float(rating_profile.get("quality_signal", 0.0))
    concept = float(rating_profile.get("concept_signal", 0.0))
    detail = float(rating_profile.get("detail_signal", 0.0))

    temps = {}

    # Semantic anchors: sharpen when quality or concept missing
    if quality < 0 or concept < 0:
        worst = min(quality, concept)
        anchor_temp = max(0.72, 1.0 + worst * 0.22)
        for b in ANCHOR_BLOCKS:
            temps[b] = anchor_temp

    # Concept zone: sharpen when concept is clearly missing
    if concept < -0.3:
        concept_temp = max(0.78, 1.0 + concept * 0.18)
        for b in _ZONE_CONCEPT:
            if b not in temps:
                temps[b] = concept_temp

    # Early zone: loosen when details missing - more variety in texture exploration
    if detail < -0.5:
        detail_temp = min(1.22, 1.0 + abs(detail) * 0.18)
        for b in _ZONE_EARLY:
            if b not in temps:
                temps[b] = detail_temp

    return temps


# ---------------------------------------------------------------------------
# Denoise creativity mask
# ---------------------------------------------------------------------------

def build_creativity_mask(latent, rating_profile, reward):
    """
    Returns a noise scale mask or None if latent is unusable.
    High-variance regions get more creative freedom. Global floor is 0 at
    high reward (formula clamps naturally) and rises toward 0.35 at reward -1.0.
    """

    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not isinstance(samples, torch.Tensor):
        return None
    if samples.dim() < 4:
        return None

    try:
        quality = float(rating_profile.get("quality_signal", 0.0))
        concept = float(rating_profile.get("concept_signal", 0.0))

        # Global creativity floor: worse rating = more freedom
        global_floor = max(0.0, min(0.35, (0.0 - reward) * 0.25))

        # Spatial variance map: high-variance regions get extra freedom
        # samples: [B, C, T, H, W] or [B, C, H, W]
        if samples.dim() == 5:
            var_map = samples.detach().float().var(dim=1, keepdim=False)  # [B, T, H, W]
        else:
            var_map = samples.detach().float().var(dim=1, keepdim=False)  # [B, H, W]

        # Normalize variance map to [0, 1]
        v_min = var_map.amin(dim=list(range(1, var_map.dim())), keepdim=True)
        v_max = var_map.amax(dim=list(range(1, var_map.dim())), keepdim=True)
        var_norm = (var_map - v_min) / (v_max - v_min + 1e-8)

        # Spatial boost: high-variance areas get more freedom
        spatial_boost = 0.15 if concept < -0.3 else 0.08

        mask = global_floor + var_norm * spatial_boost
        mask = mask.clamp(0.0, 0.5)  # cap at 50% freedom - we don't want to fully destroy structure
        return mask

    except Exception as e:
        print(f"[FunPackEnhancements] Creativity mask failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Core block replacement builder
# ---------------------------------------------------------------------------

def _build_block_replacement(block_idx, temp_scale, capture_buf, inject_tensor, inject_strength):
    """
    Builds a single patches_replace["dit"] replacement function that applies:
      - attention temperature (temp_scale != 1.0)
      - hidden state capture (capture_buf is not None)
      - hidden state injection (inject_tensor is not None)
    All in one pass through the block.
    """
    import comfy.ldm.modules.attention as attn_mod

    do_temperature = temp_scale is not None and abs(temp_scale - 1.0) > 0.001
    do_capture = capture_buf is not None
    do_inject = inject_tensor is not None

    if not do_temperature and not do_capture and not do_inject:
        return None

    def replacement(args, extra):
        # --- Temperature ---
        if do_temperature:
            s = temp_scale
            orig_attn = attn_mod.optimized_attention
            orig_attn_masked = attn_mod.optimized_attention_masked

            def _scaled(q, k, v, heads, *a, **kw):
                return orig_attn(q * s, k * s, v, heads, *a, **kw)

            def _scaled_masked(q, k, v, heads, mask, *a, **kw):
                return orig_attn_masked(q * s, k * s, v, heads, mask, *a, **kw)

            attn_mod.optimized_attention = _scaled
            attn_mod.optimized_attention_masked = _scaled_masked
            try:
                out = extra["original_block"](args)
            finally:
                attn_mod.optimized_attention = orig_attn
                attn_mod.optimized_attention_masked = orig_attn_masked
        else:
            out = extra["original_block"](args)

        # --- Capture ---
        if do_capture:
            try:
                hidden = out["img"].detach().cpu()
                capture_buf[block_idx] = hidden
            except Exception:
                pass

        # --- Inject ---
        if do_inject:
            try:
                b = inject_tensor.to(device=out["img"].device, dtype=out["img"].dtype)
                if b.shape == out["img"].shape:
                    out = {"img": out["img"].lerp(b, inject_strength)}
            except Exception:
                pass

        return out

    return replacement


# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------

def _is_ltx_model(model):
    """Return True if model is from the LTX-V / LTXAV family."""
    try:
        # Check the diffusion model class name
        inner = getattr(model, "model", None) or getattr(model, "inner_model", None)
        if inner is not None:
            cls_name = type(inner).__name__
            if any(x in cls_name for x in ("LTXV", "LTXAVModel", "LTXVModel", "LTXBaseModel")):
                return True
        # Check unet_config image_model key set by ComfyUI model detection
        cfg = getattr(model, "model_config", None)
        if cfg is not None:
            unet_cfg = getattr(cfg, "unet_config", {}) or {}
            image_model = str(unet_cfg.get("image_model", "")).lower()
            if image_model in {"ltxv", "ltxav"}:
                return True
        # Fallback: check model_type string if present
        model_type = str(getattr(model, "model_type", "") or "").lower()
        if "ltx" in model_type:
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Main enhancement builder
# ---------------------------------------------------------------------------

def build_enhancements(model, rating_profile, temporal_style, refinement_key, reward):
    """
    Apply all active LTX enhancements to the model based on rating.
    Returns patched model (already cloned).

    Called by refine_v2 / Studio.run after the attn2 direction patch is applied.
    """
    if model is None:
        return model

    if not _is_ltx_model(model):
        print("[FunPackEnhancements] Non-LTX model detected - skipping all LTX enhancements, passing model through unchanged.")
        return model

    temporal_style = str(temporal_style or "natural").strip().lower()
    if temporal_style not in TEMPORAL_STYLES:
        temporal_style = "natural"
    reward = float(reward) if reward is not None else 0.0

    model = model.clone()

    # --- Technique 2: temperature map ---
    temperature_map = _derive_temperature_map(rating_profile, reward)
    # temperature → q/k scale: scale = 1/sqrt(temp)
    temp_scales = {
        b: (1.0 / math.sqrt(max(0.1, t))) if abs(t - 1.0) > 0.01 else None
        for b, t in temperature_map.items()
    }

    # --- Technique 5: injection data ---
    blessed_maps = _load_blessed_maps(refinement_key) if refinement_key else None
    inject_strength = max(0.05, min(0.28, reward * 0.28)) if blessed_maps else 0.0

    # Shared capture buffer - populated during sampling, saved at end via wrapper
    capture_buf = {} if refinement_key else None

    # --- Install per-block patches_replace ---
    all_blocks = set(temperature_map.keys()) | (set(ANCHOR_BLOCKS) if (capture_buf is not None or blessed_maps) else set())

    if all_blocks:
        to = model.model_options.setdefault("transformer_options", {})
        pr = to.setdefault("patches_replace", {})
        dit = pr.setdefault("dit", {})

        for block_idx in all_blocks:
            t_scale = temp_scales.get(block_idx)
            inj_tensor = (blessed_maps.get(block_idx) if blessed_maps and block_idx in ANCHOR_BLOCKS else None)
            cap = capture_buf if (capture_buf is not None and block_idx in ANCHOR_BLOCKS) else None

            replacement = _build_block_replacement(block_idx, t_scale, cap, inj_tensor, inject_strength)
            if replacement is not None:
                # Compose with any existing replacement (e.g. from STG/PAG)
                existing = dit.get(("double_block", block_idx))
                if existing is not None:
                    _inner = existing
                    _outer = replacement

                    def _composed(args, extra, _i=_inner, _o=_outer):
                        inner_out = _i(args, extra)
                        # Wrap inner_out as the "original_block" for the outer
                        def _inner_as_block(a):
                            return inner_out
                        return _o(args, {"original_block": _inner_as_block})

                    dit[("double_block", block_idx)] = _composed
                else:
                    dit[("double_block", block_idx)] = replacement

    # --- Technique 3: temporal RoPE via model_function_wrapper ---
    if temporal_style != "natural":
        fps_multiplier = {
            "accelerate": 1.35,
            "decelerate": 0.72,
            "loop": 1.0,   # same fps but coordinate trick below
            "freeze": 2.0,
        }.get(temporal_style, 1.0)

        old_wrapper = model.model_options.get("model_function_wrapper")

        def _temporal_wrapper(apply_fn, args, _mult=fps_multiplier, _old=old_wrapper):
            c = args.get("c")
            if isinstance(c, dict) and "frame_rate" in c:
                try:
                    fr_cond = c["frame_rate"]
                    if hasattr(fr_cond, "cond"):
                        original_fr = float(fr_cond.cond)
                        new_fr = original_fr * _mult
                        new_cond = type(fr_cond)(new_fr)
                        new_c = dict(c)
                        new_c["frame_rate"] = new_cond
                        args = dict(args)
                        args["c"] = new_c
                except Exception:
                    pass
            if _old is not None:
                return _old(apply_fn, args)
            return apply_fn(args["input"], args["timestep"], **args.get("c", {}))

        model.model_options["model_function_wrapper"] = _temporal_wrapper

    # --- Technique 5 (capture side): save capture_buf to disk after sampling ---
    # We wrap model_function_wrapper to finalize capture on the last call.
    # "Last call" detection: we accumulate hidden states and save after first
    # call that has a small timestep (late denoising = clean signal).
    if capture_buf is not None and refinement_key:
        _rk = refinement_key
        _buf = capture_buf
        existing_wrapper = model.model_options.get("model_function_wrapper")

        def _capture_finalizer(apply_fn, args, _rk=_rk, _buf=_buf, _ew=existing_wrapper):
            if _ew is not None:
                result = _ew(apply_fn, args)
            else:
                result = apply_fn(args["input"], args["timestep"], **args.get("c", {}))

            # Opportunistically save the capture buffer when it has anchor maps.
            # We do this every call so the last (cleanest) capture wins.
            if _buf and len(_buf) >= len(ANCHOR_BLOCKS):
                try:
                    # Merge with existing temp file to preserve blocks we haven't hit
                    path = _temp_maps_path(_rk)
                    existing = {}
                    if os.path.exists(path):
                        try:
                            existing = torch.load(path, map_location="cpu", weights_only=True)
                        except Exception:
                            existing = {}
                    existing.update(_buf)
                    torch.save(existing, path)
                except Exception as e:
                    print(f"[FunPackEnhancements] Capture save failed: {e}")

            return result

        model.model_options["model_function_wrapper"] = _capture_finalizer

    return model
