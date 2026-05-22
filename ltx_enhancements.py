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
  When a run is rated Perfect or Loved it, bless those maps. Inject them into
  subsequent runs concentrated in the early high-sigma phase (sigma > 0.5) where
  anatomy and physics are formed, fading to zero by sigma 0.2.
"""

import math
import os
from hashlib import md5

import torch

TEMPORAL_STYLES = ["natural", "accelerate", "decelerate", "loop", "freeze"]

# Confirmed semantic focal points (PAG default=14, STG defaults=14,19)
ANCHOR_BLOCKS = [14, 19]

# Representative identity blocks from the concept-formation zone (20-35).
# Spaced evenly across the zone; capture and inject character appearance details.
IDENTITY_BLOCKS = [14, 20, 21, 30, 33]

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
    If mask cannot fire and a latent was explicitly connected, returns it unchanged.
    """

    # Track whether a latent was explicitly provided (for passthrough fallback)
    explicit_latent = latent if isinstance(latent, dict) and latent.get("type") != "audio" else None

    # Resolve samples source
    samples = None
    if explicit_latent is not None:
        s = explicit_latent.get("samples")
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
        return explicit_latent  # nothing to work with; pass through explicit input if any

    try:
        mask = build_creativity_mask({"samples": samples}, rating_profile, reward)
        if mask is None:
            return explicit_latent  # mask flat/invalid; pass through explicit input if any

        latent_std = float(samples.std().clamp_min(1e-8).item())
        noise = torch.randn_like(samples)
        noise_scale = (mask * latent_std).unsqueeze(1)  # broadcast over channel dim
        modified = samples + noise * noise_scale

        # Preserve all original keys from the source latent; only replace samples.
        # Keep samples on the same device as the original to avoid device mismatches.
        if explicit_latent is not None:
            result = dict(explicit_latent)
            orig_samples = explicit_latent.get("samples")
            if isinstance(orig_samples, torch.Tensor):
                modified = modified.to(device=orig_samples.device, dtype=orig_samples.dtype)
            result["samples"] = modified
            return result
        return {"samples": modified}
    except Exception as e:
        print(f"[FunPackEnhancements] Apply creativity mask failed: {e}")
        return explicit_latent  # on error, pass through explicit input unchanged


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
# Reference map extraction from i2v latent
# ---------------------------------------------------------------------------

class _LazyInject:
    """Placeholder for an inject tensor that gets populated on the first generation step."""
    __slots__ = ("tensor",)
    def __init__(self): self.tensor = None
    def set(self, t): self.tensor = t


def _has_i2v_reference(latent):
    """Return True if latent has a protected first frame (i2v reference image)."""
    if not isinstance(latent, dict):
        return False
    mask = latent.get("noise_mask")
    if mask is None:
        return False
    if getattr(mask, "is_nested", False):
        tensors = list(mask.unbind())
        if not tensors:
            return False
        mask = tensors[0]
    if not isinstance(mask, torch.Tensor) or mask.dim() < 3:
        return False
    # LTX i2v sets image frame mask = 1 - strength (e.g. 0.3 at strength=0.7).
    # Any frame below 1.0 means a reference image is present.
    return float(mask[:, :, 0].float().mean()) < 0.95


def _get_reference_video_tensor(latent):
    """Extract the video tensor from an i2v latent (index 0 of NestedTensor or flat)."""
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if samples is None:
        return None
    if getattr(samples, "is_nested", False):
        tensors = list(samples.unbind())
        return tensors[0] if tensors else None
    return samples if isinstance(samples, torch.Tensor) else None


def _run_reference_extraction(model, apply_fn, ref_x, lazy_injects, args):
    """Run one forward pass at sigma=0.03 with real conditioning to populate lazy_injects.
    Uses apply_fn directly (the base model function, works with ModelPatcherDynamic).
    Temporarily installs capture-only patches, restores them after.
    """
    saved = {}
    dit = None
    try:
        device = args["input"].device
        dtype = args["input"].dtype
        x = ref_x.to(device=device, dtype=dtype)

        to = model.model_options.setdefault("transformer_options", {})
        pr = to.setdefault("patches_replace", {})
        dit = pr.setdefault("dit", {})
        ref_cap = {}

        for idx in lazy_injects:
            saved[idx] = dit.get(("double_block", idx))
            _c, _i = ref_cap, idx

            def _rcap(a, e, c=_c, i=_i):
                out = e["original_block"](a)
                try:
                    c[i] = out["img"].detach().cpu()
                except Exception:
                    pass
                return out

            dit[("double_block", idx)] = _rcap

        ref_sigma = torch.full((x.shape[0],), 0.03, device=device, dtype=dtype)
        with torch.no_grad():
            apply_fn(x, ref_sigma, **args.get("c", {}))

        for idx, lazy in lazy_injects.items():
            if idx in ref_cap:
                lazy.set(ref_cap[idx])

        captured = sum(1 for lz in lazy_injects.values() if lz.tensor is not None)
        print(f"[FunPackEnhancements] i2v reference maps captured ({captured}/{len(lazy_injects)} blocks)")

    except Exception as e:
        print(f"[FunPackEnhancements] Reference extraction failed: {e}")
    finally:
        if dit is not None:
            for idx in lazy_injects:
                if saved.get(idx) is None:
                    dit.pop(("double_block", idx), None)
                else:
                    dit[("double_block", idx)] = saved[idx]


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

def _report_block_similarity(scene_captures):
    """Compare per-block hidden states across scenes and rank by consistency.
    High similarity = block encodes stable character features.
    Low similarity = block encodes scene-specific composition.
    """
    try:
        import torch.nn.functional as F
        all_blocks = sorted(set().union(*[s.keys() for s in scene_captures]))
        sims = {}
        for idx in all_blocks:
            tensors = [s[idx].float().flatten() for s in scene_captures if idx in s]
            if len(tensors) < 2:
                continue
            pair_sims = []
            for i in range(len(tensors)):
                for j in range(i + 1, len(tensors)):
                    a, b = tensors[i], tensors[j]
                    if a.shape == b.shape and a.norm() > 0 and b.norm() > 0:
                        pair_sims.append(float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))))
            if pair_sims:
                sims[idx] = sum(pair_sims) / len(pair_sims)
        if sims:
            ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            lines = "  ".join(f"b{i}:{v:.3f}" for i, v in ranked)
            print(f"[FunPackEnhancements] Cross-scene block consistency ({len(scene_captures)} scenes):\n  {lines}")
            top = [i for i, v in ranked if v >= 0.70]
            low = [i for i, v in ranked if v <= 0.40]
            if top:
                print(f"  -> Candidate identity blocks (sim>=0.70): {top}")
            if low:
                print(f"  -> Scene-specific blocks (sim<=0.40): {low}")
    except Exception as e:
        print(f"[FunPackEnhancements] Similarity report failed: {e}")


def _sigma_gated_strength(base_strength, sigma, sigma_high=0.5, sigma_low=0.2):
    """Scale inject_strength by sigma within [sigma_low, sigma_high] window."""
    if sigma >= sigma_high:
        return base_strength
    if sigma <= sigma_low:
        return 0.0
    return base_strength * (sigma - sigma_low) / (sigma_high - sigma_low)


def _build_block_replacement(block_idx, temp_scale, capture_buf, inject_tensor, inject_strength,
                              sigma_state=None, sigma_gate=(0.2, 0.5)):
    """
    Builds a single patches_replace["dit"] replacement function that applies:
      - attention temperature (temp_scale != 1.0)
      - hidden state capture (capture_buf is not None)
      - hidden state injection (inject_tensor is not None), sigma-gated when sigma_state provided
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
                sigma = sigma_state[0] if sigma_state is not None else 1.0
                effective = _sigma_gated_strength(inject_strength, sigma, sigma_gate[1], sigma_gate[0])
                if effective > 0.0:
                    b = inject_tensor.tensor if isinstance(inject_tensor, _LazyInject) else inject_tensor
                    if b is not None:
                        b = b.to(device=out["img"].device, dtype=out["img"].dtype)
                        if b.shape == out["img"].shape:
                            out = {"img": out["img"].lerp(b, effective)}
            except Exception:
                pass

        return out

    return replacement


# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------

def _is_ltx_model(model):
    """Return True if model is from the LTX-V / LTXAV family.

    ModelPatcher stores the BaseModel at .model, which in turn stores
    model_config (from supported_models) at .model_config. The unet_config
    inside that has image_model = 'ltxv' or 'ltxav'.
    """
    try:
        inner = getattr(model, "model", None)
        if inner is None:
            return False
        # Class name check - must include both LTXV and LTXAV explicitly
        # ('LTXV' in 'LTXAV' is False, so substring search doesn't catch LTXAV)
        cls_name = type(inner).__name__
        if cls_name in {"LTXV", "LTXAV"} or "LTXVModel" in cls_name or "LTXBaseModel" in cls_name:
            return True
        # unet_config path via model_config stored by BaseModel.__init__
        cfg = getattr(inner, "model_config", None)
        if cfg is not None:
            unet_cfg = getattr(cfg, "unet_config", {}) or {}
            if str(unet_cfg.get("image_model", "")).lower() in {"ltxv", "ltxav"}:
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Main enhancement builder
# ---------------------------------------------------------------------------

def build_enhancements(model, rating_profile, temporal_style, refinement_key, reward, reference_latent=None):
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
    has_i2v = reference_latent is not None and _has_i2v_reference(reference_latent)
    if has_i2v:
        # i2v latent: extract reference maps on first generation step (real conditioning available).
        # Never fall back to blessed maps - they may represent a completely different character.
        # Only inject at identity blocks (21, 26, 31) - anchor blocks (14, 19) encode scene
        # composition which is scene-specific and must not be overridden from the reference.
        blessed_maps = None
        ref_x = _get_reference_video_tensor(reference_latent)
        lazy_injects = {idx: _LazyInject() for idx in list(IDENTITY_BLOCKS)}
        anchor_strength = 0.0   # no anchor injection for i2v - would bleed scene content
        identity_strength = 0.02  # very light - hidden states encode full scene context
    else:
        blessed_maps = _load_blessed_maps(refinement_key) if refinement_key else None
        ref_x = None
        lazy_injects = {}
        anchor_strength = max(0.05, min(0.28, reward * 0.28)) if blessed_maps else 0.0
        identity_strength = max(0.04, min(0.12, reward * 0.12)) if blessed_maps else 0.0

    all_injection_blocks = set(ANCHOR_BLOCKS) | set(IDENTITY_BLOCKS)

    # Shared capture buffer - always created when i2v reference extraction is active
    # (so block replacements capture even without a refinement key), also used for
    # blessed map saving when a refinement key is present.
    capture_buf = {} if (refinement_key or has_i2v) else None

    # Sigma state - updated by wrapper before each model call so block replacements
    # can gate injection strength to the correct sigma window per block type.
    sigma_state = [1.0]

    # --- Install per-block patches_replace ---
    needs_injection = bool(lazy_injects) or bool(blessed_maps)
    all_blocks = set(temperature_map.keys()) | (all_injection_blocks if (capture_buf is not None or needs_injection) else set())

    if all_blocks:
        to = model.model_options.setdefault("transformer_options", {})
        pr = to.setdefault("patches_replace", {})
        dit = pr.setdefault("dit", {})

        for block_idx in all_blocks:
            t_scale = temp_scales.get(block_idx)
            is_anchor = block_idx in ANCHOR_BLOCKS
            is_identity = block_idx in IDENTITY_BLOCKS
            if is_anchor:
                inj_tensor = lazy_injects.get(block_idx) or (blessed_maps.get(block_idx) if blessed_maps else None)
                inj_strength = anchor_strength
                inj_gate = (0.2, 0.5)
            elif is_identity:
                inj_tensor = lazy_injects.get(block_idx) or (blessed_maps.get(block_idx) if blessed_maps else None)
                inj_strength = identity_strength
                inj_gate = (0.2, 0.55)
            else:
                inj_tensor = None
                inj_strength = 0.0
                inj_gate = (0.2, 0.5)
            cap = capture_buf if (capture_buf is not None and (is_anchor or is_identity)) else None

            replacement = _build_block_replacement(
                block_idx, t_scale, cap, inj_tensor, inj_strength,
                sigma_state=sigma_state if inj_tensor is not None else None,
                sigma_gate=inj_gate,
            )
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

    # --- Technique 5 (forward hooks): capture + inject via nn.Module hooks ---
    # patches_replace["dit"] reaches transformer_options but LTXAV blocks never call it.
    # Forward hooks on the actual nn.Modules are the only reliable mechanism for both
    # capturing hidden states AND injecting reference maps.
    hook_handles = []
    if capture_buf is not None:
        try:
            diff = getattr(getattr(model, "model", None), "diffusion_model", None)
            if diff is not None:
                block_list = None
                for attr in ["transformer_blocks", "blocks", "joint_blocks", "layers"]:
                    candidate = getattr(diff, attr, None)
                    if isinstance(candidate, torch.nn.ModuleList) and len(candidate) >= 28:
                        block_list = candidate
                        break
                if block_list is not None:
                    # Diagnostic: when i2v active, capture ALL candidate blocks (14,19 + 20-35)
                    # in a per-scene buffer so we can compare consistency across scenes.
                    _diag_active = has_i2v
                    _diag_blocks = (set(ANCHOR_BLOCKS) | set(range(20, 36))) if _diag_active else set()
                    _scene_cur_buf = [{}]    # hooks write here; redirected per scene
                    _scene_captures = []     # list of dicts, one per scene snapshot

                    # For i2v: only hook identity blocks (anchor blocks encode scene structure).
                    # For blessed maps: hook all injection blocks for full capture.
                    blocks_to_hook = set(lazy_injects.keys()) | (set(ANCHOR_BLOCKS) if not has_i2v else set())
                    for idx in blocks_to_hook:
                        if idx < len(block_list):
                            is_anchor = idx in ANCHOR_BLOCKS
                            inj_strength = anchor_strength if is_anchor else identity_strength
                            inj_gate = (0.2, 0.5) if is_anchor else (0.35, 0.80)
                            lazy = lazy_injects.get(idx)

                            def _make_hook(block_idx, buf, lazy_ref, strength, gate, s_state, scene_cur=_scene_cur_buf):
                                def _extract_tensor(out):
                                    if isinstance(out, dict):
                                        return out.get("img") or out.get("hidden_states") or next(iter(out.values()), None), out
                                    elif isinstance(out, tuple):
                                        return out[0], out
                                    return out, out

                                def _hook(module, inp, out):
                                    # Capture only at mid-sigma (avoid per-step tensor copies)
                                    if 0.85 <= s_state[0] <= 0.95:
                                        try:
                                            t, _ = _extract_tensor(out)
                                            if isinstance(t, torch.Tensor):
                                                buf[block_idx] = t.detach().cpu()
                                                if scene_cur is not None:
                                                    scene_cur[0][block_idx] = buf[block_idx]
                                        except Exception:
                                            pass
                                    # Inject (only when lazy has a tensor)
                                    if lazy_ref is None or lazy_ref.tensor is None:
                                        return None
                                    try:
                                        sigma = s_state[0]
                                        effective = _sigma_gated_strength(strength, sigma, gate[1], gate[0])
                                        if effective <= 0.0:
                                            return None
                                        t, _ = _extract_tensor(out)
                                        if not isinstance(t, torch.Tensor):
                                            return None
                                        b = lazy_ref.tensor.to(device=t.device, dtype=t.dtype)
                                        if b.shape != t.shape:
                                            return None
                                        injected = t.lerp(b, effective)
                                        if isinstance(out, dict):
                                            key = "img" if "img" in out else "hidden_states" if "hidden_states" in out else next(iter(out))
                                            return {**out, key: injected}
                                        elif isinstance(out, tuple):
                                            return (injected,) + out[1:]
                                        return injected
                                    except Exception:
                                        return None
                                return _hook

                            h = block_list[idx].register_forward_hook(
                                _make_hook(idx, capture_buf, lazy, inj_strength, inj_gate, sigma_state)
                            )
                            hook_handles.append(h)
                    # Register capture-only diagnostic hooks on extra candidate blocks.
                    # Gate by sigma to avoid capturing (and copying tensors) on every step.
                    for idx in _diag_blocks - blocks_to_hook:
                        if idx < len(block_list):
                            def _make_diag_hook(block_idx, scene_cur, s_state):
                                def _hook(module, inp, out):
                                    # Only capture in the mid-sigma window once per scene
                                    if not (0.85 <= s_state[0] <= 0.95):
                                        return None
                                    try:
                                        if isinstance(out, dict):
                                            t = out.get("img") or out.get("hidden_states") or next(iter(out.values()), None)
                                        elif isinstance(out, tuple):
                                            t = out[0]
                                        else:
                                            t = out
                                        if isinstance(t, torch.Tensor):
                                            scene_cur[0][block_idx] = t.detach().cpu()
                                    except Exception:
                                        pass
                                    return None
                                return _hook
                            h = block_list[idx].register_forward_hook(_make_diag_hook(idx, _scene_cur_buf, sigma_state))
                            hook_handles.append(h)

                    # Also redirect the main capture hooks to write into scene_cur_buf too
                    capture_buf["__scene_cur__"] = _scene_cur_buf
                    capture_buf["__scene_captures__"] = _scene_captures

                    if hook_handles:
                        print(f"[FunPackEnhancements] Registered {len(hook_handles)} capture+inject hooks on transformer blocks")
                else:
                    print("[FunPackEnhancements] Could not find transformer block list for hooks")
        except Exception as e:
            print(f"[FunPackEnhancements] Hook registration failed: {e}")

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

    # --- Technique 5 (sigma tracking + i2v reference extraction) ---
    # Updates sigma_state before each model call so injection gating has the current
    # timestep. Also fires reference map extraction on the first call when i2v is used.
    if needs_injection or has_i2v:
        _existing_for_sigma = model.model_options.get("model_function_wrapper")
        _state = sigma_state
        _ref_extracted = [False]
        _ref_x = ref_x
        _lazy = lazy_injects

        _prev_sigma = [1.0]
        _scene_snapped = [False]

        def _sigma_tracker(apply_fn, args, _ew=_existing_for_sigma, _state=_state,
                           _ref_extracted=_ref_extracted, _ref_x=_ref_x,
                           _lazy=_lazy, _cap_buf=capture_buf,
                           _prev_sigma=_prev_sigma, _scene_snapped=_scene_snapped):
            ts = args.get("timestep")
            try:
                sigma = float(ts.max().item()) if ts is not None else 1.0
            except Exception:
                sigma = 1.0

            # Detect scene transition: sigma jumped back up = new scene started
            scene_cur = _cap_buf.get("__scene_cur__") if isinstance(_cap_buf, dict) else None
            scene_captures = _cap_buf.get("__scene_captures__") if isinstance(_cap_buf, dict) else None
            if scene_cur is not None and sigma > _prev_sigma[0] + 0.05:
                # New scene: only save previous scene if it was properly snapped
                # (filters out phantom scenes from model init passes)
                if _scene_snapped[0] and scene_cur[0]:
                    scene_captures.append(dict(scene_cur[0]))
                scene_cur[0] = {}
                _scene_snapped[0] = False
            _prev_sigma[0] = sigma
            _state[0] = sigma

            # Run the actual generation step
            if _ew is not None:
                result = _ew(apply_fn, args)
            else:
                result = apply_fn(args["input"], args["timestep"], **args.get("c", {}))

            # Snapshot for injection: first time sigma < 0.95
            if _ref_x is not None and not _ref_extracted[0] and _cap_buf and _lazy:
                if sigma < 0.95 and len({k: v for k, v in _cap_buf.items() if not isinstance(k, str)}) >= len(_lazy):
                    _ref_extracted[0] = True
                    for idx, lazy in _lazy.items():
                        if idx in _cap_buf:
                            lazy.set(_cap_buf[idx])
                    captured = sum(1 for lz in _lazy.values() if lz.tensor is not None)
                    print(f"[FunPackEnhancements] i2v reference maps ready at sigma={sigma:.2f} ({captured}/{len(_lazy)} blocks)")

            # Per-scene diagnostic snapshot at mid-sigma
            if scene_cur is not None and not _scene_snapped[0] and sigma < 0.95:
                _scene_snapped[0] = True  # hooks already wrote into scene_cur[0] at this sigma

            # Fire similarity report at end of each scene (sigma low) when ≥2 scenes captured
            if scene_captures is not None and sigma < 0.5 and _scene_snapped[0]:
                if scene_cur is not None and scene_cur[0] and (not scene_captures or scene_captures[-1] is not scene_cur[0]):
                    scene_captures.append(dict(scene_cur[0]))
                if len(scene_captures) >= 2:
                    _report_block_similarity(scene_captures)

            return result

        model.model_options["model_function_wrapper"] = _sigma_tracker

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

    # --- Hook cleanup: remove forward hooks after sampling completes ---
    if hook_handles:
        _handles = hook_handles
        existing_wrapper = model.model_options.get("model_function_wrapper")

        _scene_cur_ref = capture_buf.get("__scene_cur__") if isinstance(capture_buf, dict) else None
        _scene_cap_ref = capture_buf.get("__scene_captures__") if isinstance(capture_buf, dict) else None

        def _hook_remover(apply_fn, args, _ew=existing_wrapper, _handles=_handles,
                          _removed=[False], _sc=_scene_cur_ref, _sca=_scene_cap_ref):
            if _ew is not None:
                result = _ew(apply_fn, args)
            else:
                result = apply_fn(args["input"], args["timestep"], **args.get("c", {}))
            ts = args.get("timestep")
            if ts is not None and not _removed[0]:
                try:
                    if float(ts.max().item()) < 0.05:
                        for h in _handles:
                            h.remove()
                        _removed[0] = True
                        # Save last scene and report cross-scene block similarity
                        if _sc is not None and _sca is not None:
                            if _sc[0]:
                                _sca.append(dict(_sc[0]))
                            if len(_sca) >= 2:
                                _report_block_similarity(_sca)
                except Exception:
                    pass
            return result

        model.model_options["model_function_wrapper"] = _hook_remover

    return model
