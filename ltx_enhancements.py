"""
LTX-V model enhancements for FunPack - applied automatically via the refiner rating.

Technique 2: Per-block attention temperature
  Sharp semantic blocks when quality/concept is missing, looser early blocks when details missing.

Technique 3: Temporal RoPE style
  Manipulate frame_rate in the model's positional embedding to change motion character.
  Styles: natural / accelerate / decelerate / loop / freeze / pulse

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
import weakref
from hashlib import md5

import torch

TEMPORAL_STYLES = ["natural", "auto", "accelerate", "decelerate", "loop", "freeze", "pulse"]

# frame_rate multiplier fed to the LTX RoPE per style. >1 => model assumes more
# frames cover the same wall-clock time => smaller inter-frame deltas => smoother /
# more "held" motion (freeze = strongest hold). <1 => larger deltas => punchier,
# more dynamic motion. Single source of truth, shared by the global (concrete-style)
# path in build_enhancements and the per-scene "auto" path in the chain sampler.
TEMPORAL_STYLE_MULT = {
    "accelerate": 1.35,
    "decelerate": 0.72,
    "loop": 1.0,   # loop = Mobius latent roll (make_loop_temporal_wrapper), not a frame_rate mult
    "freeze": 2.0,
}

# Pulse: repeated ease-down segments per scene. Each segment starts punchy (peak mult)
# and eases toward a temporal hold (floor mult) before the next segment resets.
PULSE_SEGMENT_COUNT = 3
PULSE_PEAK_MULT = 0.88
PULSE_FLOOR_MULT = 1.65


# Keyword -> motion-energy intent for the "auto" director. The classifier scans a
# scene's prompt and emits a continuous frame_rate multiplier (plus a loop flag).
# "Energy" words want larger inter-frame deltas (mult < 1, toward decelerate);
# "stillness" words want a hold (mult > 1, toward freeze). Tuned to stay inside the
# same envelope as the manual presets (~0.72 .. 2.0) so auto never goes wilder than
# a human would pick by hand.
_TEMPORAL_ENERGY_WORDS = (
    "run", "running", "sprint", "chase", "fight", "punch", "kick", "explosion",
    "explode", "burst", "jump", "leap", "dash", "race", "fast", "rapid", "quick",
    "frantic", "spin", "whirl", "fall", "crash", "swing", "throw", "action",
    "dance", "dancing", "gallop", "charge", "shatter", "blast", "speeding",
)
_TEMPORAL_STILL_WORDS = (
    "still", "stillness", "motionless", "static", "frozen", "freeze", "portrait",
    "calm", "serene", "quiet", "slow", "slowly", "gentle", "gradual", "drift",
    "drifting", "float", "floating", "hover", "stare", "gaze", "meditat",
    "sleep", "resting", "rest", "standing", "posed", "pose", "close-up", "closeup",
    "macro", "landscape", "ambient", "tranquil", "lingering",
)
_TEMPORAL_LOOP_WORDS = (
    "loop", "looping", "seamless", "cycle", "cyclic", "repeat", "repeating",
    "endless", "perpetual", "continuous loop",
)


def classify_temporal_intent(text):
    """Heuristic per-scene motion director for temporal_style="auto".

    Reads a scene's prompt and returns a dict describing the temporal treatment:
      {"mult": float, "loop": bool, "label": str}
    `mult` is a continuous frame_rate multiplier (1.0 = no change). This is the
    baseline director; it is structured so a learned (reward-driven) override can
    replace `mult` per intent-family later without touching callers. See
    [[project-velocity-bias-role]] for why motion energy is treated as a dial.
    """
    t = str(text or "").lower()
    if not t.strip():
        return {"mult": 1.0, "loop": False, "label": "natural"}
    energy = sum(1 for w in _TEMPORAL_ENERGY_WORDS if w in t)
    still = sum(1 for w in _TEMPORAL_STILL_WORDS if w in t)
    loop = any(w in t for w in _TEMPORAL_LOOP_WORDS)
    net = still - energy  # >0 => want a hold; <0 => want energy
    if net == 0:
        mult, label = 1.0, "natural"
    elif net > 0:
        # ramp toward freeze (2.0); 1 hint ~ accelerate-grade hold, 2+ => stronger
        mult = min(2.0, 1.0 + 0.35 * net)
        label = "freeze" if mult >= 1.7 else "accelerate"
    else:
        # ramp toward decelerate (0.72) for punchier motion
        mult = max(0.72, 1.0 - 0.14 * (-net))
        label = "decelerate"
    return {"mult": round(float(mult), 3), "loop": bool(loop), "label": label}


def _ease_down(t):
    """Smooth ease from 0→1 for pulse segment ramps."""
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def pulse_mult_for_progress(progress, segment_count=None, peak_mult=None, floor_mult=None):
    """Map denoise progress [0,1] to a frame_rate multiplier with repeated ease-down segments."""
    segs = PULSE_SEGMENT_COUNT if segment_count is None else max(1, int(segment_count))
    peak = PULSE_PEAK_MULT if peak_mult is None else float(peak_mult)
    floor = PULSE_FLOOR_MULT if floor_mult is None else float(floor_mult)
    progress = max(0.0, min(1.0, float(progress)))
    seg_idx = min(int(progress * segs), segs - 1)
    local_t = progress * segs - seg_idx
    return peak + (floor - peak) * _ease_down(local_t)


def _scale_frame_rate_in_args(args, mult):
    """Return a shallow-copied args dict with frame_rate scaled by mult (no-op on failure)."""
    c = args.get("c")
    if not (isinstance(c, dict) and "frame_rate" in c):
        return args
    try:
        fr_cond = c["frame_rate"]
        if hasattr(fr_cond, "cond"):
            original_fr = float(fr_cond.cond)
            new_cond = type(fr_cond)(original_fr * float(mult))
            new_c = dict(c)
            new_c["frame_rate"] = new_cond
            new_args = dict(args)
            new_args["c"] = new_c
            return new_args
    except Exception:
        pass
    return args


def make_temporal_wrapper(old_wrapper, mult):
    """Build a model_function_wrapper that scales the LTX `frame_rate` conditioning by
    `mult`, chaining any existing wrapper. Returns None when mult is effectively 1.0
    (nothing to do). Shared by the global concrete-style path and the per-scene auto
    path so both apply identical RoPE manipulation."""
    try:
        mult = float(mult)
    except (TypeError, ValueError):
        return None
    if abs(mult - 1.0) < 1e-3:
        return None

    def _temporal_wrapper(apply_fn, args, _mult=mult, _old=old_wrapper):
        args = _scale_frame_rate_in_args(args, _mult)
        if _old is not None:
            return _old(apply_fn, args)
        return apply_fn(args["input"], args["timestep"], **args.get("c", {}))

    return _temporal_wrapper


def make_pulse_temporal_wrapper(old_wrapper, segment_count=None, peak_mult=None, floor_mult=None):
    """Build a model_function_wrapper that cycles ease-down temporal segments per scene.

    Denoise progress (derived from sigma) is divided into `segment_count` segments. Within
    each segment the frame_rate multiplier eases from `peak_mult` (punchier motion) toward
    `floor_mult` (temporal hold). A fresh wrapper should be installed per scene so sigma
    bounds reset between scenes."""
    segs = PULSE_SEGMENT_COUNT if segment_count is None else max(1, int(segment_count))
    peak = PULSE_PEAK_MULT if peak_mult is None else float(peak_mult)
    floor = PULSE_FLOOR_MULT if floor_mult is None else float(floor_mult)
    sigma_start = [None]

    def _pulse_wrapper(apply_fn, args, _old=old_wrapper, _segs=segs, _peak=peak, _floor=floor):
        ts = args.get("timestep")
        try:
            sigma = float(ts.max().item()) if ts is not None else 1.0
        except Exception:
            sigma = 1.0
        if sigma_start[0] is None:
            sigma_start[0] = max(sigma, 1e-6)
        progress = max(0.0, min(1.0, 1.0 - sigma / sigma_start[0]))
        mult = pulse_mult_for_progress(progress, _segs, _peak, _floor)
        args = _scale_frame_rate_in_args(args, mult)
        if _old is not None:
            return _old(apply_fn, args)
        return apply_fn(args["input"], args["timestep"], **args.get("c", {}))

    return _pulse_wrapper

# --- Loop temporal style: Mobius-style latent roll (arXiv:2502.20307) ---------------
# Seamless looping without training or extra forward passes: on eligible denoise steps
# the latent is cyclically rolled along its temporal dim before the forward and the
# prediction unrolled after. The model then repeatedly sees the video's seam
# (last frame | first frame) as an INTERIOR frame pair at a step-varying position and
# smooths it like any other cut — after enough steps the sequence is consistent under
# rotation, i.e. it loops. WanVideoWrapper's "Loop Args" is the same trick for WAN.
#
# Roll starts below the near-noise plateau: content is ~pure noise up there (nothing to
# smooth) and the plateau step-cache reuses base forwards, which a per-step roll would
# invalidate. Guides pin absolute frame positions (keyframe_idxs / attention entries
# reference appended frames inside the same T range), so calls carrying them are left
# canonical rather than rolled.
LOOP_ROLL_MAX_SIGMA = 0.95
LOOP_ROLL_MIN_FRAMES = 4  # below this many latent frames there is nothing to loop


def _van_der_corput(n):
    """Base-2 bit-reversal sequence in (0,1): 0.5, 0.25, 0.75, 0.125, 0.625… — every
    prefix covers the unit interval near-uniformly, so however many steps end up
    eligible, the seam visits well-spread temporal positions."""
    v, denom = 0.0, 1.0
    n = int(n)
    while n:
        denom *= 2.0
        v += (n & 1) / denom
        n >>= 1
    return v


def _loop_stream_shapes(args):
    """Per-stream latent shapes for a packed [B,1,N] input, or None when the layout
    can't be trusted. Mirrors the samplers' packed-latent handling (video first, then
    audio; both streams keep their temporal dim at index 2)."""
    c = args.get("c") or {}
    shapes = c.get("latent_shapes")
    if hasattr(shapes, "cond"):
        shapes = shapes.cond
    x = args.get("input")
    if x is None:
        return None
    if not shapes:
        # Single-stream (plain LTXV) fed unpacked: treat the input itself as one stream.
        return [tuple(int(d) for d in x.shape)] if x.ndim == 5 else None
    try:
        shapes = [tuple(int(d) for d in s) for s in shapes]
        import math as _math
        if sum(_math.prod(s[1:]) for s in shapes) != int(x.shape[-1]):
            return None  # packed layout doesn't match — don't risk rolling blind
    except (TypeError, ValueError):
        return None
    return shapes


def _loop_roll_packed(x, shapes, frac, direction):
    """Cyclically roll every stream's temporal dim (dim 2) of a packed [B,1,N] tensor by
    round(frac * T_stream) frames (video and audio each by their own frame count, so the
    two stay time-aligned). direction=+1 rolls, -1 unrolls. Returns a new tensor."""
    import math as _math
    if x.ndim == 5:  # unpacked single stream
        t = x.shape[2]
        shift = int(round(frac * t)) % t
        return torch.roll(x, shifts=direction * shift, dims=2) if shift else x
    out = x.clone()
    off = 0
    for dims in shapes:
        sz = _math.prod(dims[1:])
        if len(dims) >= 3 and dims[2] >= 2:
            t = dims[2]
            shift = int(round(frac * t)) % t
            if shift:
                stream = x[..., off:off + sz].reshape([x.shape[0]] + list(dims[1:]))
                out[..., off:off + sz] = torch.roll(
                    stream, shifts=direction * shift, dims=2).reshape(x.shape[0], 1, sz)
        off += sz
    return out


def _loop_roll_mask(mask, frac, direction):
    """Roll a denoise mask ([B,1,T,H,W] video / [B,C,T,F] audio) in step with its stream."""
    if not isinstance(mask, torch.Tensor) or mask.ndim < 3 or mask.shape[2] < 2:
        return mask
    t = mask.shape[2]
    shift = int(round(frac * t)) % t
    return torch.roll(mask, shifts=direction * shift, dims=2) if shift else mask


def make_loop_temporal_wrapper(old_wrapper):
    """Build the loop-style model_function_wrapper. Installed INNERMOST (closest to
    apply_model): prediction-modifying wrappers layered above it (dynashift, output
    guidance, …) must see canonical-orientation inputs and outputs — the roll exists
    only for the duration of the base forward. The per-step shift follows a van der
    Corput sequence, reset whenever sigma jumps back up (a new scene/run)."""
    state = {"count": 0, "last_sigma": None, "logged": False}

    def _loop_wrapper(apply_fn, args, _old=old_wrapper):
        def _call(a):
            if _old is not None:
                return _old(apply_fn, a)
            return apply_fn(a["input"], a["timestep"], **a.get("c", {}))

        try:
            ts = args.get("timestep")
            sigma = float(ts.max().item()) if ts is not None else 1.0
        except Exception:
            sigma = 1.0
        if state["last_sigma"] is not None and sigma > state["last_sigma"] + 1e-4:
            state["count"] = 0  # sigma went back up: new scene/run
        state["last_sigma"] = sigma

        c = args.get("c") or {}
        if (
            sigma > LOOP_ROLL_MAX_SIGMA
            or c.get("keyframe_idxs") is not None
            or c.get("guide_attention_entries") is not None
        ):
            return _call(args)
        shapes = _loop_stream_shapes(args)
        if not shapes:
            return _call(args)
        # Gate on the VIDEO stream's frame count (the 5-dim shape); audio rides along.
        t_video = max((s[2] for s in shapes if len(s) == 5),
                      default=max((s[2] for s in shapes if len(s) >= 3), default=0))
        if t_video < LOOP_ROLL_MIN_FRAMES:
            return _call(args)

        state["count"] += 1
        frac = _van_der_corput(state["count"])
        if not state["logged"]:
            state["logged"] = True
            print(f"[FunPack] loop temporal style: Mobius latent roll active (T={t_video})")
        try:
            new_c = dict(c)
            for key in ("denoise_mask", "audio_denoise_mask"):
                if isinstance(new_c.get(key), torch.Tensor):
                    new_c[key] = _loop_roll_mask(new_c[key], frac, 1)
            rolled = dict(args)
            rolled["input"] = _loop_roll_packed(args["input"], shapes, frac, 1)
            rolled["c"] = new_c
            out = _call(rolled)
            return _loop_roll_packed(out, shapes, frac, -1)
        except Exception as e:
            if state.get("roll_error") is None:
                state["roll_error"] = True
                print(f"[FunPack] loop roll failed, running canonical: {e}")
            return _call(args)

    return _loop_wrapper


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


def _attn_weights_temp_path(refinement_key):
    return os.path.join(_maps_dir(), f"attn_weights_temp_{_safe_key(refinement_key)}.pt")


def _attn_weights_blessed_path(refinement_key):
    return os.path.join(_maps_dir(), f"attn_weights_blessed_{_safe_key(refinement_key)}.pt")


def _load_blessed_attn_weights(refinement_key):
    path = _attn_weights_blessed_path(refinement_key)
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None


def bless_attn_weights(refinement_key):
    """EMA-merge temp attention weights into blessed. Call on liked ratings."""
    src = _attn_weights_temp_path(refinement_key)
    dst = _attn_weights_blessed_path(refinement_key)
    if not os.path.exists(src):
        return False
    try:
        new_data = torch.load(src, map_location="cpu", weights_only=True)
        if os.path.exists(dst):
            old_data = torch.load(dst, map_location="cpu", weights_only=True)
            merged = {}
            for k in set(old_data) | set(new_data):
                o, n = old_data.get(k), new_data.get(k)
                if o is not None and n is not None and o.shape == n.shape:
                    merged[k] = (0.8 * o.float() + 0.2 * n.float()).half()
                else:
                    merged[k] = (n if n is not None else o)
            torch.save(merged, dst)
        else:
            torch.save({k: v.half() for k, v in new_data.items()}, dst)
        print(f"[FunPackEnhancements] Blessed attention weights for key '{refinement_key}'")
        return True
    except Exception as e:
        print(f"[FunPackEnhancements] Attn weights bless failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Raw K/V identity bank (BachVid-style)
#
# The blessed-attn store above keeps only a per-token IMPORTANCE weight (output
# norm). BachVid (arxiv 2510.21696) instead reuses the raw attention Keys and
# Values from a liked "identity" generation and re-injects them into later runs,
# so character appearance carries across scenes/sessions without a reference
# image or any training. We capture K/V at the IDENTITY_BLOCKS during the
# mid-sigma window and lerp the running gen's K/V toward the blessed K/V at the
# same token positions (the packed-safe, RoPE-safe analogue of the paper's
# concat-and-attend; same seq positions => no RoPE surgery, no mask changes).
# Stored as {block_idx: {"k": [seq, D] half, "v": [seq, D] half}}, batch-mean.
# ---------------------------------------------------------------------------

def _kv_temp_path(refinement_key):
    return os.path.join(_maps_dir(), f"kv_temp_{_safe_key(refinement_key)}.pt")


def _kv_blessed_path(refinement_key):
    return os.path.join(_maps_dir(), f"kv_blessed_{_safe_key(refinement_key)}.pt")


def _load_blessed_kv(refinement_key):
    path = _kv_blessed_path(refinement_key)
    if not os.path.exists(path):
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(data, dict) or not data:
            return None
        return data
    except Exception:
        return None


def bless_kv(refinement_key):
    """EMA-merge temp K/V identity bank into blessed. Call on liked ratings."""
    src = _kv_temp_path(refinement_key)
    dst = _kv_blessed_path(refinement_key)
    if not os.path.exists(src):
        return False
    try:
        new_data = torch.load(src, map_location="cpu", weights_only=True)
        if os.path.exists(dst):
            old_data = torch.load(dst, map_location="cpu", weights_only=True)
            merged = {}
            for b in set(old_data) | set(new_data):
                o, n = old_data.get(b), new_data.get(b)
                if not isinstance(n, dict):
                    merged[b] = o
                    continue
                if not isinstance(o, dict):
                    merged[b] = n
                    continue
                entry = {}
                for tk in ("k", "v"):
                    ot, nt = o.get(tk), n.get(tk)
                    if isinstance(ot, torch.Tensor) and isinstance(nt, torch.Tensor) and ot.shape == nt.shape:
                        entry[tk] = (0.8 * ot.float() + 0.2 * nt.float()).half()
                    else:
                        entry[tk] = nt if isinstance(nt, torch.Tensor) else ot
                merged[b] = entry
            torch.save(merged, dst)
        else:
            torch.save(new_data, dst)
        print(f"[FunPackEnhancements] Blessed K/V identity bank for key '{refinement_key}'")
        return True
    except Exception as e:
        print(f"[FunPackEnhancements] K/V bless failed: {e}")
        return False


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
        # Gate by noise_mask so i2v reference frames (mask≈0.3) aren't perturbed
        if explicit_latent is not None:
            nm = explicit_latent.get("noise_mask")
            if isinstance(nm, torch.Tensor) and nm.shape[2] == noise_scale.shape[2]:
                noise_scale = noise_scale * nm.float().to(noise_scale.device)
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
    """Remove all enhancement files for a key. Called on reset_session.

    A keyless run stores its maps/velocity under the "default" bucket (see the
    `refinement_key or "default"` fallback on the write side, e.g. samplers.py
    _velocity_store_path). Normalize the same way here so a keyless Session Reset
    actually wipes that bucket instead of no-opping on an empty key."""
    norm = str(refinement_key or "default").strip() or "default"
    for path in (
        _temp_maps_path(norm),
        _blessed_maps_path(norm),
        _creativity_latent_path(norm),
        _attn_weights_temp_path(norm),
        _attn_weights_blessed_path(norm),
        _kv_temp_path(norm),
        _kv_blessed_path(norm),
    ):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"[FunPackEnhancements] Cleanup failed for {path}: {e}")
    # Velocity-bias / rescue trajectory memory is in-process (not a file), so clear
    # it here too — otherwise a session reset leaves stale good-trajectory memory that
    # rescue and velocity bias would keep steering toward.
    try:
        try:
            from .samplers import clear_velocity_bias_memory
        except ImportError:
            from samplers import clear_velocity_bias_memory
        clear_velocity_bias_memory(norm)
    except Exception as e:
        print(f"[FunPackEnhancements] Velocity memory cleanup failed: {e}")


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

def _derive_temperature_map(rating_profile, reward, vf_score=None):
    """
    Returns {block_idx: temperature} for blocks that should deviate from 1.0.
      temperature < 1.0  sharper/colder - confident, focused decisions
      temperature > 1.0  softer/warmer  - more varied, exploratory

    When vf_score is provided (value function predicted reward for current conditioning),
    it continuously modulates temperature independent of the rating signal:
      vf_score near +1 → sharpen (model is confident this direction is good)
      vf_score near -1 → warm (explore more, current direction looks bad)
    """
    quality = float(rating_profile.get("quality_signal", 0.0))
    concept = float(rating_profile.get("concept_signal", 0.0))
    detail = float(rating_profile.get("detail_signal", 0.0))

    temps = {}

    # Rating-driven temperatures (only when previous result was poor)
    if reward < 0.8:
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

    # Value-function-driven temperature: continuous, score-based modulation
    # Runs on every generation when VF is ready, regardless of rating
    if vf_score is not None:
        # vf_score ∈ [-1, 1]: map to temperature factor
        # +1 → 0.80 (sharpen — VF predicts this is a good direction)
        # 0  → 1.00 (neutral)
        # -1 → 1.20 (warm — VF predicts poor outcome, explore more)
        vf_temp = max(0.80, min(1.20, 1.0 - vf_score * 0.20))
        for b in list(ANCHOR_BLOCKS) + list(_ZONE_CONCEPT):
            if b in temps:
                # Blend: rating temp × VF temp, weighted toward VF
                temps[b] = temps[b] * 0.5 + vf_temp * 0.5
            else:
                temps[b] = vf_temp

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


def _extend_v_pe(kwargs, n_ref):
    """Extend v_pe in kwargs by prepending n_ref neutral-rotation entries (cos=1, sin=0)."""
    v_pe = kwargs.get("v_pe")
    if v_pe is None:
        return kwargs
    try:
        cos, sin = v_pe[0], v_pe[1]
        dev, dt = cos.device, cos.dtype
        ndim = cos.ndim
        if ndim == 4:        # [B, H, T, d]
            r = (cos.shape[0], cos.shape[1], n_ref, cos.shape[3])
            dim = 2
        elif ndim == 3:      # [B, T, d]
            r = (cos.shape[0], n_ref, cos.shape[2])
            dim = 1
        elif ndim == 2:      # [T, d]
            r = (n_ref, cos.shape[1])
            dim = 0
        else:
            return kwargs
        ref_cos = torch.ones(r, device=dev, dtype=dt)
        ref_sin = torch.zeros(r, device=dev, dtype=dt)
        ext_cos = torch.cat([ref_cos, cos], dim=dim)
        ext_sin = torch.cat([ref_sin, sin], dim=dim)
        tail = tuple(v_pe[2:]) if len(v_pe) > 2 else ()
        new_kwargs = dict(kwargs)
        new_kwargs["v_pe"] = (ext_cos, ext_sin) + tail
        return new_kwargs
    except Exception:
        return kwargs


def _build_block_replacement(block_idx, temp_scale, capture_buf, inject_tensor, inject_strength,
                              sigma_state=None, sigma_gate=(0.2, 0.5),
                              attn_capture_buf=None, attn_inject=None, attn_inject_strength=0.0,
                              attn_sigma_gate=(0.35, 0.85),
                              kv_capture_buf=None, kv_inject=None, kv_inject_strength=0.0,
                              kv_sigma_gate=(0.35, 0.85), kvlock_scale=None):
    """
    Builds a single patches_replace["dit"] replacement function that applies:
      - attention temperature (temp_scale != 1.0)
      - attention weight capture (attn_capture_buf) — output norm proxy per token position
      - attention weight inject (attn_inject) — V scaling by accumulated importance
      - raw K/V capture (kv_capture_buf) — BachVid identity bank, batch-mean per position
      - raw K/V inject (kv_inject) — lerp K/V toward blessed identity at matched positions;
        injection strength is multiplied by kvlock_scale[0] (KV-Lock variance schedule, 1.0 = off)
      - hidden state capture (capture_buf)
      - hidden state injection (inject_tensor), sigma-gated
    All in one pass through the block.
    """
    import comfy.ldm.modules.attention as attn_mod

    do_temperature = temp_scale is not None and abs(temp_scale - 1.0) > 0.001
    do_capture = capture_buf is not None
    do_inject = inject_tensor is not None
    do_attn_capture = attn_capture_buf is not None
    do_attn_inject = attn_inject is not None and attn_inject_strength > 0
    do_kv_capture = kv_capture_buf is not None
    do_kv_inject = kv_inject is not None and kv_inject_strength > 0
    do_attn_patch = do_temperature or do_attn_capture or do_attn_inject or do_kv_capture or do_kv_inject

    if not do_temperature and not do_capture and not do_inject and not do_attn_patch:
        return None

    def replacement(args, extra):
        # --- Attention patch (temperature + attn capture/inject + K/V capture/inject) ---
        if do_attn_patch:
            s = temp_scale if do_temperature else 1.0
            orig_attn = attn_mod.optimized_attention
            orig_attn_masked = attn_mod.optimized_attention_masked

            def _attn_body(q, k, v, orig_fn, extra_args, extra_kwargs):
                v_use = v
                k_use = k
                sigma = sigma_state[0] if sigma_state is not None else 0.0
                # --- BachVid K/V inject: lerp toward blessed identity at matched positions ---
                if do_kv_inject and kv_sigma_gate[0] <= sigma <= kv_sigma_gate[1]:
                    try:
                        scale_mul = float(kvlock_scale[0]) if kvlock_scale is not None else 1.0
                        alpha = max(0.0, min(0.6, kv_inject_strength * scale_mul))
                        if alpha > 0.0:
                            k_id = kv_inject.get("k")
                            v_id = kv_inject.get("v")
                            if isinstance(k_id, torch.Tensor) and k_id.shape[0] == k.shape[1] and k_id.shape[1] == k.shape[2]:
                                k_id = k_id.to(device=k.device, dtype=k.dtype)
                                k_use = k.lerp(k_id.unsqueeze(0).expand_as(k), alpha)
                            if isinstance(v_id, torch.Tensor) and v_id.shape[0] == v.shape[1] and v_id.shape[1] == v.shape[2]:
                                v_id = v_id.to(device=v.device, dtype=v.dtype)
                                v_use = v.lerp(v_id.unsqueeze(0).expand_as(v), alpha)
                    except Exception:
                        k_use, v_use = k, v
                if do_attn_inject and attn_inject.shape[0] == v.shape[1]:
                    if attn_sigma_gate[0] <= sigma <= attn_sigma_gate[1]:
                        imp = attn_inject.to(device=v.device, dtype=torch.float32)
                        imp = imp / imp.mean().clamp(min=1e-6)
                        scale = (1.0 + attn_inject_strength * (imp - 1.0)).to(v.dtype)
                        v_use = v_use * scale.unsqueeze(0).unsqueeze(-1)
                result = orig_fn(q * s, k_use * s, v_use, *extra_args, **extra_kwargs)
                # --- BachVid K/V capture: batch-mean K/V at this position, last-in-window wins ---
                if do_kv_capture and kv_sigma_gate[0] <= sigma <= kv_sigma_gate[1]:
                    try:
                        kv_capture_buf[block_idx] = {
                            "k": k.detach().float().mean(dim=0).half().cpu(),
                            "v": v.detach().float().mean(dim=0).half().cpu(),
                        }
                    except Exception:
                        pass
                if do_attn_capture:
                    try:
                        imp_cap = result.detach().float().norm(dim=-1).mean(dim=0).cpu()
                        if block_idx in attn_capture_buf:
                            attn_capture_buf[block_idx] = 0.7 * attn_capture_buf[block_idx] + 0.3 * imp_cap
                        else:
                            attn_capture_buf[block_idx] = imp_cap
                    except Exception:
                        pass
                return result

            def _scaled(q, k, v, heads, *a, **kw):
                return _attn_body(q, k, v, lambda q_, k_, v_, *a_, **kw_: orig_attn(q_, k_, v_, heads, *a_, **kw_), a, kw)

            def _scaled_masked(q, k, v, heads, mask, *a, **kw):
                return _attn_body(q, k, v, lambda q_, k_, v_, *a_, **kw_: orig_attn_masked(q_, k_, v_, heads, mask, *a_, **kw_), a, kw)

            attn_mod.optimized_attention = _scaled
            attn_mod.optimized_attention_masked = _scaled_masked
            try:
                out = extra["original_block"](args)
            finally:
                attn_mod.optimized_attention = orig_attn
                attn_mod.optimized_attention_masked = orig_attn_masked
        else:
            out = extra["original_block"](args)

        # --- Hidden state capture ---
        if do_capture:
            try:
                hidden = out["img"].detach().cpu()
                capture_buf[block_idx] = hidden
            except Exception:
                pass

        # --- Hidden state inject ---
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
# Block-hook leak guard
# ---------------------------------------------------------------------------
# Technique-5 forward hooks (see _register_hooks) attach directly to the SHARED
# diffusion blocks. They are removed on a scene-transition sigma jump, but NOT at
# end-of-sampling, so a single-scene run (or the final scene of a multi-scene run)
# leaves them installed. Since the blocks are shared across ModelPatcher clones,
# the next generation stacks another set — over many gens this compounds into
# progressive output corruption that survives Session Reset and deleting
# refinements/ (the hooks live in-process, not on disk). We tag every installed
# hook and strip any leftovers at the start of each run so accumulation is bounded
# to a single set and removed before the next sampling pass.
_FUNPACK_HOOK_TAG = "_funpack_enh_hook"


def _funpack_locate_blocks(model):
    diff = getattr(getattr(model, "model", None), "diffusion_model", None)
    if diff is None:
        return None
    for attr in ("transformer_blocks", "blocks", "joint_blocks", "layers"):
        candidate = getattr(diff, attr, None)
        if isinstance(candidate, torch.nn.ModuleList) and len(candidate) >= 28:
            return candidate
    return None


def strip_funpack_block_hooks(model):
    """Remove forward / forward-pre hooks this module installed on the shared
    diffusion blocks in a previous run. Idempotent; safe to call every run."""
    bl = _funpack_locate_blocks(model)
    if bl is None:
        return 0
    removed = 0
    for block in bl:
        for store in (getattr(block, "_forward_hooks", None),
                      getattr(block, "_forward_pre_hooks", None)):
            if not store:
                continue
            for hid in [hid for hid, fn in list(store.items())
                        if getattr(fn, _FUNPACK_HOOK_TAG, False)]:
                store.pop(hid, None)
                removed += 1
    if removed:
        print(f"[FunPackEnhancements] Stripped {removed} leaked block hook(s) from a previous run")
    return removed


# ---------------------------------------------------------------------------
# Main enhancement builder
# ---------------------------------------------------------------------------

def build_enhancements(model, rating_profile, temporal_style, refinement_key, reward, reference_latent=None, conditioning=None):
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

    # Strip any block hooks leaked by a previous run before installing fresh ones,
    # so they cannot stack across generations (see _FUNPACK_HOOK_TAG note above).
    strip_funpack_block_hooks(model)

    # --- VF score for temperature modulation ---
    vf_score = None
    if conditioning is not None and refinement_key:
        try:
            try:
                from .value_function import OnlineValueFunction
                from .conditioning import refinement_state_path
            except ImportError:
                from value_function import OnlineValueFunction
                from conditioning import refinement_state_path
            _vf_path = refinement_state_path(refinement_key, "value_fn", prefix="refine_v2", extension="pt")
            if os.path.exists(_vf_path):
                with torch.inference_mode(False):
                    _vf = OnlineValueFunction.load(_vf_path)
                if _vf.is_ready():
                    _cond = conditioning[0][0] if isinstance(conditioning, (list, tuple)) else conditioning
                    if isinstance(_cond, torch.Tensor):
                        with torch.inference_mode(False), torch.no_grad():
                            vf_score = float(_vf.forward(_vf.compress(_cond.float()).unsqueeze(0)).item())
        except Exception:
            pass

    # --- Attention weight accumulation ---
    attn_capture_buf = {} if refinement_key else None
    blessed_attn = _load_blessed_attn_weights(refinement_key) if refinement_key else None
    attn_inject_strength = max(0.04, min(0.18, reward * 0.18)) if blessed_attn else 0.0

    # --- BachVid raw K/V identity bank ---
    # Capture K/V on every keyed run (temp bank, promoted on Perfect); inject the
    # blessed identity K/V when one exists. Strength scales with reward like the
    # importance-weight inject above. kvlock_scale is the shared multiplier the
    # KV-Lock variance scheduler (Phase 2) writes per step; 1.0 = passthrough.
    kv_capture_buf = {} if refinement_key else None
    blessed_kv = _load_blessed_kv(refinement_key) if refinement_key else None
    kv_inject_strength = max(0.04, min(0.22, reward * 0.22)) if blessed_kv else 0.0
    kvlock_scale = [1.0]

    # --- Technique 2: temperature map ---
    temperature_map = _derive_temperature_map(rating_profile, reward, vf_score=vf_score)
    # temperature → q/k scale: scale = 1/sqrt(temp)
    temp_scales = {
        b: (1.0 / math.sqrt(max(0.1, t))) if abs(t - 1.0) > 0.01 else None
        for b, t in temperature_map.items()
    }

    # --- Technique 5: injection data ---
    # Clear the temp maps file so this generation's capture starts fresh.
    # Without this, maps from previous runs merge in and compound across
    # sessions — each "loved it" cycle injects tainted states from the
    # previous round, producing progressively worse color bias.
    if refinement_key:
        temp_path = _temp_maps_path(refinement_key)
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

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
        identity_strength = 1.0  # in-context gate: inject when sigma in gate range
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
    kv_active = kv_capture_buf is not None or bool(blessed_kv)
    attn_active_blocks = set(IDENTITY_BLOCKS) if (attn_capture_buf is not None or blessed_attn or kv_active) else set()
    all_blocks = set(temperature_map.keys()) | (all_injection_blocks if (capture_buf is not None or needs_injection) else set()) | attn_active_blocks

    if all_blocks:
        to = model.model_options.setdefault("transformer_options", {})
        # Expose the KV-Lock multiplier so the sampler's variance scheduler (Phase 2)
        # can drive the BachVid injection strength step-by-step via this same list.
        if kv_active:
            to["funpack_kvlock_scale"] = kvlock_scale
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

            attn_inj = blessed_attn.get(block_idx) if blessed_attn else None
            kv_cap = kv_capture_buf if (kv_capture_buf is not None and is_identity) else None
            kv_inj = blessed_kv.get(block_idx) if (blessed_kv and is_identity) else None
            replacement = _build_block_replacement(
                block_idx, t_scale, cap, inj_tensor, inj_strength,
                sigma_state=sigma_state if (inj_tensor is not None or attn_inj is not None or kv_inj is not None or kv_cap is not None) else None,
                sigma_gate=inj_gate,
                attn_capture_buf=attn_capture_buf,
                attn_inject=attn_inj,
                attn_inject_strength=attn_inject_strength,
                kv_capture_buf=kv_cap,
                kv_inject=kv_inj if isinstance(kv_inj, dict) else None,
                kv_inject_strength=kv_inject_strength,
                kvlock_scale=kvlock_scale,
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

    # --- Technique 5 (forward hooks): lazy per-step registration ---
    # Hooks are only registered for the 2-3 steps where they actually fire.
    # High-sigma steps (1-5 of 8) have zero hook overhead.
    _block_list = None
    _hook_block_params = {}  # {idx: (buf, lazy, strength, gate)}
    hook_handles = []
    if capture_buf is not None and has_i2v:
        try:
            diff = getattr(getattr(model, "model", None), "diffusion_model", None)
            if diff is not None:
                for attr in ["transformer_blocks", "blocks", "joint_blocks", "layers"]:
                    candidate = getattr(diff, attr, None)
                    if isinstance(candidate, torch.nn.ModuleList) and len(candidate) >= 28:
                        _block_list = candidate
                        break
                if _block_list is not None:
                    _block_list = weakref.ref(_block_list)  # avoid strong ref cycle through wrapper
                    blocks_to_hook = set(lazy_injects.keys()) | (set(ANCHOR_BLOCKS) if not has_i2v else set())
                    _bl = _block_list()
                    for idx in blocks_to_hook:
                        if _bl is not None and idx < len(_bl):
                            is_anchor = idx in ANCHOR_BLOCKS
                            _hook_block_params[idx] = (
                                capture_buf,
                                lazy_injects.get(idx),
                                anchor_strength if is_anchor else identity_strength,
                                (0.2, 0.5) if is_anchor else (0.35, 0.80),
                            )
                else:
                    print("[FunPackEnhancements] Could not find transformer block list for hooks")
        except Exception as e:
            print(f"[FunPackEnhancements] Hook setup failed: {e}")

    # --- Technique 3: temporal RoPE via model_function_wrapper ---
    # "auto" and "pulse" are per-scene and owned by the chain sampler (auto bakes a
    # per-scene multiplier; pulse installs a segmented ease-down wrapper per scene).
    # The global path here fires for concrete manual styles and pulse on single-scene
    # workflows. "loop" = Mobius latent roll (not a frame_rate mult); installed here it
    # sits innermost relative to the chain sampler's per-scene wrappers, which is
    # required — prediction-modifying wrappers must see canonical orientation.
    if temporal_style == "pulse":
        wrapper = make_pulse_temporal_wrapper(model.model_options.get("model_function_wrapper"))
        if wrapper is not None:
            model.model_options["model_function_wrapper"] = wrapper
    elif temporal_style == "loop":
        model.model_options["model_function_wrapper"] = make_loop_temporal_wrapper(
            model.model_options.get("model_function_wrapper"))
    elif temporal_style not in ("natural", "auto"):
        mult = TEMPORAL_STYLE_MULT.get(temporal_style, 1.0)
        wrapper = make_temporal_wrapper(model.model_options.get("model_function_wrapper"), mult)
        if wrapper is not None:
            model.model_options["model_function_wrapper"] = wrapper

    # --- Technique 5 (sigma tracking + i2v reference extraction) ---
    # Updates sigma_state before each model call so injection gating has the current
    # timestep. Also fires reference map extraction on the first call when i2v is used.
    if needs_injection or has_i2v:
        _existing_for_sigma = model.model_options.get("model_function_wrapper")
        _state = sigma_state
        _ref_extracted = [False]
        _ref_x = ref_x
        _lazy = lazy_injects

        _prev_sigma_track = [1.0]
        _active_handles = []

        def _register_hooks():
            bl = _block_list() if _block_list is not None else None
            if not bl or not _hook_block_params:
                return

            def _extract_tensor(out):
                if isinstance(out, dict):
                    return out.get("img") or out.get("hidden_states") or next(iter(out.values()), None), out
                elif isinstance(out, tuple):
                    return out[0], out
                return out, out

            def _make_hook(block_idx, buf, lazy_ref, strength, gate, s_state):
                def _hook(module, inp, out):
                    if 0.85 <= s_state[0] <= 0.95:
                        try:
                            t, _ = _extract_tensor(out)
                            if isinstance(t, torch.Tensor):
                                buf[block_idx] = t.detach()
                        except Exception:
                            pass
                    if lazy_ref is None or lazy_ref.tensor is None:
                        return None
                    try:
                        effective = _sigma_gated_strength(strength, s_state[0], gate[1], gate[0])
                        if effective <= 0.0:
                            return None
                        t, _ = _extract_tensor(out)
                        if not isinstance(t, torch.Tensor):
                            return None
                        b = lazy_ref.tensor.to(device=t.device, dtype=t.dtype, non_blocking=True)
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
                setattr(_hook, _FUNPACK_HOOK_TAG, True)
                return _hook

            def _make_incontext_hooks(block_idx, buf, lazy_ref, strength, gate, s_state):
                """Pre+post hook pair for in-context conditioning.

                Pre-hook: captures block input at sigma (0.85-0.95); prepends ref tokens
                          to vx and extends v_pe with neutral-rotation entries.
                Post-hook: crops the ref-token prefix from the block output's vx.
                """
                _n_prepended = [0]

                def _pre(module, args, kwargs):
                    sigma = s_state[0]
                    # Capture block INPUT during mid-sigma window
                    if 0.85 <= sigma <= 0.95:
                        try:
                            vx = args[0][0]
                            buf[block_idx] = vx.detach().cpu()
                        except Exception:
                            pass
                    # Inject: prepend ref tokens to extend self-attention context
                    if lazy_ref is None or lazy_ref.tensor is None:
                        _n_prepended[0] = 0
                        return None
                    effective = _sigma_gated_strength(strength, sigma, gate[1], gate[0])
                    if effective <= 0.0:
                        _n_prepended[0] = 0
                        return None
                    try:
                        vx, ax = args[0]
                        ref = lazy_ref.tensor.to(vx.device, vx.dtype)
                        ref_exp = ref.expand(vx.shape[0], -1, -1)
                        new_vx = torch.cat([ref_exp, vx], dim=1)
                        n_ref = ref.shape[1]
                        _n_prepended[0] = n_ref
                        new_kwargs = _extend_v_pe(kwargs, n_ref)
                        return ((new_vx, ax),) + args[1:], new_kwargs
                    except Exception:
                        _n_prepended[0] = 0
                        return None

                def _post(module, inp, out):
                    n = _n_prepended[0]
                    if n == 0:
                        return None
                    try:
                        vx, ax = out
                        return (vx[:, n:], ax)
                    except Exception:
                        return None

                setattr(_pre, _FUNPACK_HOOK_TAG, True)
                setattr(_post, _FUNPACK_HOOK_TAG, True)
                return _pre, _post

            for idx, (buf, lazy_ref, strength, gate) in _hook_block_params.items():
                is_identity = idx in IDENTITY_BLOCKS
                if is_identity and has_i2v:
                    # In-context conditioning: video tokens attend to reference tokens
                    pre, post = _make_incontext_hooks(idx, buf, lazy_ref, strength, gate, _state)
                    h1 = bl[idx].register_forward_pre_hook(pre, with_kwargs=True)
                    h2 = bl[idx].register_forward_hook(post)
                    _active_handles.append(h1)
                    _active_handles.append(h2)
                else:
                    h = bl[idx].register_forward_hook(
                        _make_hook(idx, buf, lazy_ref, strength, gate, _state)
                    )
                    _active_handles.append(h)

        def _remove_hooks():
            for h in _active_handles:
                try: h.remove()
                except Exception: pass
            _active_handles.clear()

        def _sigma_tracker(apply_fn, args, _ew=_existing_for_sigma, _state=_state,
                           _ref_extracted=_ref_extracted, _ref_x=_ref_x,
                           _lazy=_lazy, _cap_buf=capture_buf,
                           _prev=_prev_sigma_track, _active=_active_handles,
                           _reg=_register_hooks, _rem=_remove_hooks):
            ts = args.get("timestep")
            try:
                sigma = float(ts.max().item()) if ts is not None else 1.0
            except Exception:
                sigma = 1.0

            # Scene transition: sigma jumped back up → unregister hooks for clean state
            if sigma > _prev[0] + 0.05 and _active:
                _rem()
            _prev[0] = sigma
            _state[0] = sigma

            # Register hooks lazily when sigma enters the active zone
            if not _active and sigma < 0.95 and _ref_x is not None:
                _reg()

            if _ew is not None:
                result = _ew(apply_fn, args)
            else:
                result = apply_fn(args["input"], args["timestep"], **args.get("c", {}))

            # Snapshot into lazy injects at first mid-sigma crossing
            if _ref_x is not None and not _ref_extracted[0] and _cap_buf and _lazy:
                if sigma < 0.95 and len(_cap_buf) >= len(_lazy):
                    _ref_extracted[0] = True
                    for idx, lazy in _lazy.items():
                        if idx in _cap_buf:
                            lazy.set(_cap_buf[idx])

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

    # --- Attn weight capture finalizer: save attn_capture_buf to temp file ---
    if attn_capture_buf is not None and refinement_key:
        _attn_buf = attn_capture_buf
        _attn_rk = refinement_key
        _attn_saved = [False]
        existing_wrapper = model.model_options.get("model_function_wrapper")

        def _attn_finalizer(apply_fn, args, _ew=existing_wrapper, _buf=_attn_buf,
                             _rk=_attn_rk, _saved=_attn_saved):
            if _ew is not None:
                result = _ew(apply_fn, args)
            else:
                result = apply_fn(args["input"], args["timestep"], **args.get("c", {}))
            if not _saved[0] and len(_buf) >= len(IDENTITY_BLOCKS):
                try:
                    torch.save({k: v.half() for k, v in _buf.items()},
                               _attn_weights_temp_path(_rk))
                    _saved[0] = True
                except Exception as e:
                    print(f"[FunPackEnhancements] Attn weights save failed: {e}")
            return result

        model.model_options["model_function_wrapper"] = _attn_finalizer

    # --- K/V identity bank finalizer: save kv_capture_buf to temp file ---
    if kv_capture_buf is not None and refinement_key:
        _kv_buf = kv_capture_buf
        _kv_rk = refinement_key
        _kv_saved = [False]
        existing_wrapper = model.model_options.get("model_function_wrapper")

        def _kv_finalizer(apply_fn, args, _ew=existing_wrapper, _buf=_kv_buf,
                          _rk=_kv_rk, _saved=_kv_saved):
            if _ew is not None:
                result = _ew(apply_fn, args)
            else:
                result = apply_fn(args["input"], args["timestep"], **args.get("c", {}))
            # Overwrite each call so the last (cleanest) mid-window capture wins.
            if len(_buf) >= len(IDENTITY_BLOCKS):
                try:
                    torch.save(dict(_buf), _kv_temp_path(_rk))
                    _saved[0] = True
                except Exception as e:
                    print(f"[FunPackEnhancements] K/V bank save failed: {e}")
            return result

        model.model_options["model_function_wrapper"] = _kv_finalizer

    # Hook cleanup is handled by _sigma_tracker's scene-transition logic.
    # _active_handles is cleared on each sigma reset and at the end.

    return model
