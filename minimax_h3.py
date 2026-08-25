"""MiniMax-H3 adapter — the one place that knows how H3 differs from LTXAV.

FunPack was built against LTX2.3/LTXAV, and supports BOTH families side by side:
nothing here changes an LTX run, and every entry point answers "not H3" when the
model in hand is an LTX one. H3 (merged as ComfyUI 57500fc5, PR #15224, shipped
in ComfyUI v0.30.0) is close enough that most of the stack works untouched, and
different in four ways that break silently if nobody accounts for them:

1. **Audio latent axis.** LTXAV's audio latent is ``[B, C, T, freq]`` — time on
   dim 2, the same axis as the video latent's frames, which is why the Chain
   Sampler could slice every stream at ``[:, :, a:b]``. H3's audio latent is
   ``[B, 32, stereo=2, T]`` — dim 2 is the STEREO CHANNEL. Slicing it as time
   silently crops one speaker instead of trimming the tail. Both are 4-D, so the
   tensor alone cannot tell you which; it has to come from the model.

2. **Frame grid.** LTXAV is ``8k+1`` pixel frames -> ``k+1`` latent frames. H3 is
   ``17k+5`` -> ``5k+2`` (and the VAE reports ``downscale_index_formula = (4,16,16)``,
   which is the *index* map, not the count map — reading a frame count off it is
   off by ~20%). The count always lives in ``vae.downscale_ratio[0]``.

3. **No cross-attention.** H3 is a single packed self-attention stream
   ``[text | cond/ref | audio | video]``. There is no ``attn2``, no ``audio_attn2``,
   and no audio text-channel half — so the LTXAV conditioning channel split is a
   no-op here (correctly: the whole tensor conditions both streams).

4. **Batch size 1.** ``MiniMaxH3Model._forward`` raises on ``x.shape[0] != 1``,
   and comfy batches cond+uncond into one forward whenever ``cfg != 1.0``. Left
   alone that makes every CFG generation crash; :func:`install_batch_split`
   splits the batch back apart around the model call.

Nothing here imports the H3 core modules at module scope: an older ComfyUI (or an
LTX-only install) has none of them, and every entry point has to degrade to "not
H3" rather than raise on import.

WEIGHTS COME IN TWO CHECKPOINTS, and they are not interchangeable:
``minimax_h3_fl2va_*`` is the t2v / first-last-frame model (keyframe pins) and
``minimax_h3_ref2va_*`` is the reference model (reference blocks). Both load
through the same detection and produce a structurally valid graph either way, so
nothing crashes when the wrong one is loaded — it just conditions badly. The DiT
state dict carries no variant marker, so :func:`checkpoint_mode_note` reports
which mode a run is actually using and lets the user match the file.
"""
from __future__ import annotations

import math

import torch

# ── the model's own constants (comfy/ldm/minimax/model.py, nodes_minimax_h3.py) ──
IMAGE_MODEL = "minimax_h3"
FPS = 24                     # H3 generates at a fixed 24 fps
FRAME_GRID = 17              # pixel frames per grid step ...
FRAME_BASE = 5               # ... on top of this base (frame_count % 17 == 5)
LATENT_GRID = 5              # latent frames per grid step ...
LATENT_BASE = 2              # ... on top of this base
AUDIO_LATENT_FPS = 40        # audio latent frames per second
SPATIAL_DOWNSCALE = 16
VIDEO_LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32
CANVAS_MULTIPLE = 32         # width/height must be a multiple of this
REF_IMAGE_SHORT_EDGE = 2048  # the reference pipeline's own "max" sizing
DEFAULT_SHIFT_VIDEO = 12.0
DEFAULT_SHIFT_AUDIO = 3.0

# transformer_options keys MiniMaxH3SigmaShift sets. The DiT inverts the VIDEO shift to
# its base grid to derive the audio timestep AND the audio velocity slope, so a schedule
# built on a different shift has to say so here or the audio stream is integrated against
# the wrong clock. Absent, the DiT falls back to its own 12.0 / 3.0.
SHIFT_VIDEO_OPTION = "minimax_h3_sigma_shift_video"
SHIFT_AUDIO_OPTION = "minimax_h3_sigma_shift_audio"

# Audio rows are [B, 32, stereo, T]: time is the LAST axis, not dim 2.
AUDIO_TIME_DIM = -1
VIDEO_TIME_DIM = 2


# ── the audio clock ──────────────────────────────────────────────────────────
# H3 denoises video and audio on two different flow schedules (shift 12 / shift 3).
# Only ONE sigma grid reaches a sampler, so comfy reconciles them inside the DiT: it
# returns the audio velocity pre-multiplied by d(sigma_a)/d(sigma_v), which makes the
# flat ODE a sampler integrates on the video grid equal to the audio stream's true ODE
# on its own grid (comfy/ldm/minimax/model.py, final lines of _forward).
#
# That identity holds in the CONTINUOUS limit. The derivative is the tangent at the
# start of the step, and the two schedules diverge hard, so on a big step the tangent
# badly overshoots the chord it is standing in for. On a 4-step shift-12 schedule the
# video grid ~[1, .973, .923, .8, 0] maps to audio ~[1, .9, .75, .5, 0]; the final step's
# tangent is 1.56 where the true chord is 0.63 — the audio is driven ~2.5x past its
# target and comes out distorted. At 20+ steps the two agree to within a percent, which
# is why this only shows up on distilled / turbo few-step schedules.
#
# The fix is to integrate the audio on its own grid: swap the tangent for the chord that
# actually spans the step. One scalar per step, no extra model evaluation.

_AUDIO_CLOCK_MAX_FACTOR = 16.0


def time_shift_sigma(sigma, from_shift, to_shift):
    """Re-express a sigma from one flow schedule on another. Mirrors the DiT's own."""
    sigma = float(sigma)
    from_shift = float(from_shift)
    to_shift = float(to_shift)
    denom = from_shift + sigma * (1.0 - from_shift)
    if abs(denom) < 1e-12:
        return sigma
    base = sigma / denom
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def time_shift_slope(sigma, from_shift, to_shift):
    """d(sigma_to)/d(sigma_from) at the same base-grid point. Mirrors the DiT's own."""
    sigma = float(sigma)
    from_shift = float(from_shift)
    to_shift = float(to_shift)
    denom = from_shift + sigma * (1.0 - from_shift)
    if abs(denom) < 1e-12 or abs(from_shift) < 1e-12:
        return 1.0
    base = sigma / denom
    lower = from_shift * (1.0 + (to_shift - 1.0) * base) ** 2
    if abs(lower) < 1e-12:
        return 1.0
    return (to_shift * (1.0 + (from_shift - 1.0) * base) ** 2) / lower


def audio_clock_factors(sigmas, shift_video=DEFAULT_SHIFT_VIDEO, shift_audio=DEFAULT_SHIFT_AUDIO):
    """Per-step correction for the audio stream's displacement, one float per step.

    The sampler moves audio by ``slope(sigma_v) * v_a * (sigma_v_next - sigma_v)``.
    The audio's own schedule wants ``v_a * (sigma_a_next - sigma_a)``. Multiplying the
    audio part of the step by

        (sigma_a_next - sigma_a) / (slope(sigma_v) * (sigma_v_next - sigma_v))

    turns the first into the second exactly, for any sampler that moves the stream by a
    velocity times the step. Returns ``len(sigmas) - 1`` factors; 1.0 wherever the
    correction is undefined (zero-length step, degenerate slope), so it no-ops rather
    than dividing by ~0.

    Every factor is 1.0 when the two shifts are equal — the schedules coincide and
    there is nothing to correct.
    """
    out = []
    values = [float(s) for s in sigmas]
    for i in range(max(0, len(values) - 1)):
        sv, sv_next = values[i], values[i + 1]
        dv = sv_next - sv
        slope = time_shift_slope(sv, shift_video, shift_audio)
        if abs(dv) < 1e-9 or abs(slope) < 1e-9:
            out.append(1.0)
            continue
        da = (time_shift_sigma(sv_next, shift_video, shift_audio)
              - time_shift_sigma(sv, shift_video, shift_audio))
        factor = da / (slope * dv)
        # With the stock shifts (video 12 > audio 3) the map is convex, the tangent is
        # taken at the step's high-sigma end, and the factor lands in (0, 1] — it always
        # SHORTENS the audio step. Shifts the other way round invert that and the factor
        # exceeds 1, which is correct and must not be clamped away. The bound here is a
        # numerical sanity rail only, wide enough never to bind on a real schedule.
        out.append(float(min(_AUDIO_CLOCK_MAX_FACTOR, max(0.0, factor))))
    return out


def resolve_sigma_shifts(model_options):
    """The (video, audio) flow shifts in force, from a model_options dict.

    MiniMaxH3SigmaShift writes both into transformer_options; without that node the DiT
    falls back to its own defaults, and so do we.
    """
    to = {}
    if isinstance(model_options, dict):
        candidate = model_options.get("transformer_options")
        if isinstance(candidate, dict):
            to = candidate
    try:
        shift_v = float(to.get(SHIFT_VIDEO_OPTION, DEFAULT_SHIFT_VIDEO))
    except (TypeError, ValueError):
        shift_v = DEFAULT_SHIFT_VIDEO
    try:
        shift_a = float(to.get(SHIFT_AUDIO_OPTION, DEFAULT_SHIFT_AUDIO))
    except (TypeError, ValueError):
        shift_a = DEFAULT_SHIFT_AUDIO
    return shift_v, shift_a


# ── detection ────────────────────────────────────────────────────────────────
# All three of these have to work when the H3 PR is not installed at all, so they
# only ever read attributes that any comfy object has.

def _unet_config(model):
    inner = getattr(model, "model", None)
    cfg = getattr(inner, "model_config", None)
    return getattr(cfg, "unet_config", {}) or {}


def is_h3_model(model) -> bool:
    """True for a ModelPatcher (or BaseModel) wrapping the MiniMax H3 DiT."""
    if model is None:
        return False
    try:
        inner = getattr(model, "model", None)
        if type(inner).__name__ == "MiniMaxH3":
            return True
        if type(model).__name__ == "MiniMaxH3":
            return True
        if str(_unet_config(model).get("image_model", "")).lower() == IMAGE_MODEL:
            return True
        diff = getattr(inner, "diffusion_model", None)
        if type(diff).__name__ == "MiniMaxH3Model":
            return True
    except Exception:
        pass
    return False


def is_h3_clip(clip) -> bool:
    """True for the Qwen3-VL-32B conditioning encoder H3 ships with."""
    if clip is None:
        return False
    try:
        tok = getattr(clip, "tokenizer", None)
        if type(tok).__name__ == "MiniMaxH3Tokenizer":
            return True
        # SD1Tokenizer names its sub-tokenizer after the TE; H3's is unique.
        if tok is not None and hasattr(tok, "qwen3vl_32b"):
            return True
        cond_stage = getattr(clip, "cond_stage_model", None)
        if type(cond_stage).__name__ in ("MiniMaxH3TEModel", "MiniMaxH3TEModel_"):
            return True
    except Exception:
        pass
    return False


def is_h3_video_vae(vae) -> bool:
    if vae is None:
        return False
    try:
        return type(getattr(vae, "first_stage_model", None)).__name__ == "MiniMaxH3VideoVAE"
    except Exception:
        return False


def is_h3_audio_vae(vae) -> bool:
    if vae is None:
        return False
    try:
        return type(getattr(vae, "first_stage_model", None)).__name__ == "MiniMaxH3AudioVAE"
    except Exception:
        return False


# ── frame-grid geometry ──────────────────────────────────────────────────────

def align_frame_count(length: int) -> int:
    """Snap a pixel frame count UP to the model's 17k+5 grid (the node's rule)."""
    n = max(FRAME_BASE, int(length))
    while n % FRAME_GRID != FRAME_BASE:
        n += 1
    return n


def video_latent_frames(frame_count: int) -> int:
    """Pixel frames -> video latent frames, for an already-aligned count."""
    n = int(frame_count)
    if n <= FRAME_BASE:
        return LATENT_BASE
    return ((n - FRAME_BASE) // FRAME_GRID) * LATENT_GRID + LATENT_BASE


def audio_latent_frames(frame_count: int) -> int:
    """Pixel frames -> audio latent frames (40 Hz against a 24 fps video clock)."""
    return int(round(int(frame_count) / float(FPS) * AUDIO_LATENT_FPS))


def temporal_shape(length: int):
    """(aligned pixel frames, video latent frames, audio latent frames)."""
    frames = align_frame_count(length)
    return frames, video_latent_frames(frames), audio_latent_frames(frames)


def latent_frames_from_vae(vae, pixel_frames: int):
    """Video latent frame count for `pixel_frames`, asked of the VAE itself.

    ``vae.downscale_ratio[0]`` is the count map every video VAE in comfy carries
    (LTX: ``ceil(n/8)``; H3: ``(n-5)//17*5+2``). Reading it beats hardcoding a
    per-family formula and beats ``downscale_index_formula``, which is the index
    map and gives the wrong count on H3. Returns None when the VAE doesn't
    expose one, so callers can keep their old fallback.
    """
    try:
        ratio = getattr(vae, "downscale_ratio", None)
        if isinstance(ratio, (list, tuple)) and ratio and callable(ratio[0]):
            return max(1, int(ratio[0](max(1, int(pixel_frames)))))
        if isinstance(ratio, (int, float)) and ratio:
            return ((max(1, int(pixel_frames)) - 1) // max(1, int(ratio))) + 1
    except Exception:
        pass
    return None


def adapt_canvas(width: int, height: int, base_short_edge=768, max_pixels=768 * 1344):
    """H3's reference canvas rule: 768 short edge, area-capped, rounded to 32."""
    width = max(1, int(width))
    height = max(1, int(height))
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = base_short_edge * ratio, float(base_short_edge)
    else:
        nom_w, nom_h = float(base_short_edge), base_short_edge / ratio
    if nom_w * nom_h > max_pixels:
        s = math.sqrt(max_pixels / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    snap = lambda v: max(CANVAS_MULTIPLE, int(round(v / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)
    return snap(nom_w), snap(nom_h)


# ── per-stream time axes ─────────────────────────────────────────────────────

def stream_time_dims(stream_count: int, h3: bool):
    """Time axis per latent stream, in comfy's pack order (video first).

    LTXAV: every stream carries time on dim 2. H3: video on dim 2, audio on the
    last axis (dim 2 there is the stereo channel). Returned as a plain list so
    callers can index it by stream position without re-deciding the family.
    """
    dims = [VIDEO_TIME_DIM] * max(1, int(stream_count))
    if h3:
        for i in range(1, len(dims)):
            dims[i] = AUDIO_TIME_DIM
    return dims


def time_len(tensor, dim: int) -> int:
    return int(tensor.shape[dim])


def time_slice(tensor, start, end, dim: int):
    idx = [slice(None)] * tensor.dim()
    idx[dim] = slice(start, end)
    return tensor[tuple(idx)]


def set_time_slice(tensor, start, end, value, dim: int):
    idx = [slice(None)] * tensor.dim()
    idx[dim] = slice(start, end)
    tensor[tuple(idx)] = value
    return tensor


def time_cat(tensors, dim: int):
    return torch.cat(tensors, dim=dim if dim >= 0 else tensors[0].dim() + dim)


# ── attention bridge ─────────────────────────────────────────────────────────
# LTX calls ``comfy.ldm.modules.attention.optimized_attention(...)`` through the
# module, so FunPack's temperature / capture / K-V-lock patches take effect by
# rebinding that one symbol. H3 does ``from ... import optimized_attention`` at
# module load, which snapshots the function — rebinding the source module does
# nothing. Any patch that wants to reach H3 has to rebind H3's own copy too.

def attention_patch_targets():
    """Modules whose ``optimized_attention`` symbol must be rebound together.

    Always includes comfy's attention module; includes the H3 DiT module only
    when the PR is installed. Callers save/restore whatever they find.
    """
    targets = []
    try:
        import comfy.ldm.modules.attention as attn_mod
        targets.append(attn_mod)
    except Exception:
        return targets
    try:
        import comfy.ldm.minimax.model as h3_mod
        if hasattr(h3_mod, "optimized_attention"):
            targets.append(h3_mod)
    except Exception:
        pass
    return targets


# H3 calls ``optimized_attention`` with ``skip_reshape=True`` and q/k/v already at
# ``[1, heads, S, head_dim]``. A patch that reads ``v.shape[1]`` as a token count (the
# LTX layout) is reading the HEAD count instead, so anything indexed by token position
# has to check that kwarg before assuming a layout. Temperature is layout agnostic.


# ── batch-size-1 constraint ──────────────────────────────────────────────────

def _split_cond_batch(value, index, batch):
    """Slice one entry of the ``c`` dict down to sample `index` of `batch`."""
    if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == batch:
        return value[index:index + 1]
    if getattr(value, "is_nested", False):
        try:
            import comfy.nested_tensor
            parts = [t[index:index + 1] if t.shape[0] == batch else t for t in value.unbind()]
            return comfy.nested_tensor.NestedTensor(parts)
        except Exception:
            return value
    return value


def install_batch_split(model):
    """Run the H3 DiT one sample at a time when comfy hands it a batched forward.

    ``MiniMaxH3Model._forward`` raises on ``x.shape[0] != 1``. In practice comfy does
    NOT batch the positive and negative conds together on H3 — each carries its own
    ``minimax_payload`` (different token tags, different packed layout), and an unequal
    ``CONDConstant`` blocks the concat — so the cfg>1 path survives on its own. What
    does reach the DiT batched is a latent whose batch size is >1 (several videos in one
    request). Splitting here keeps that working at exactly the cost the batch already
    implies, and stays correct if comfy's batching rules change.

    Returns the previous ``model_function_wrapper`` so the caller can restore it.
    """
    old_wrapper = model.model_options.get("model_function_wrapper")

    def _call(apply_fn, args):
        if old_wrapper is not None:
            return old_wrapper(apply_fn, args)
        return apply_fn(args["input"], args["timestep"], **args.get("c", {}))

    def _batch_split_wrapper(apply_fn, args, _old=old_wrapper):
        x = args.get("input")
        try:
            batch = int(x.shape[0]) if isinstance(x, torch.Tensor) else 1
        except Exception:
            batch = 1
        if batch <= 1:
            return _call(apply_fn, args)
        outs = []
        ts = args.get("timestep")
        c = args.get("c") or {}
        for i in range(batch):
            sub = dict(args)
            sub["input"] = x[i:i + 1]
            if isinstance(ts, torch.Tensor) and ts.shape[0] == batch:
                sub["timestep"] = ts[i:i + 1]
            sub["c"] = {k: _split_cond_batch(v, i, batch) for k, v in c.items()}
            outs.append(_call(apply_fn, sub))
        return torch.cat(outs, dim=0)

    model.model_options["model_function_wrapper"] = _batch_split_wrapper
    return old_wrapper, _batch_split_wrapper


# ── conditioning payload ─────────────────────────────────────────────────────
# H3 carries its visual conditioning on the CONDITIONING dict, not in the latent:
#   minimax_keyframes  — fl2va first/last frame pins (never denoised)
#   minimax_frame_count— pixel length, so the layout can place the last-frame pin
#   minimax_refs       — ref2va identity/style/audio reference blocks
# model_base.MiniMaxH3.extra_conds turns these into the DiT payload.

def _set_values(conditioning, values):
    out = []
    for entry in conditioning or []:
        cond, meta = entry[0], (entry[1] if len(entry) > 1 else {})
        new_meta = dict(meta) if isinstance(meta, dict) else {}
        for key, value in values.items():
            if value is None:
                new_meta.pop(key, None)
            else:
                new_meta[key] = value
        out.append([cond, new_meta])
    return out


def set_keyframes(conditioning, keyframes, frame_count):
    """Attach fl2va keyframe pins. `keyframes` = [{resolved_frame_index, latent}]."""
    if not keyframes:
        return conditioning
    return _set_values(conditioning, {
        "minimax_keyframes": list(keyframes),
        "minimax_frame_count": int(frame_count),
    })


def set_refs(conditioning, ref_blocks):
    """Attach ref2va reference blocks (images / videos / audio)."""
    if not ref_blocks:
        return conditioning
    return _set_values(conditioning, {"minimax_refs": list(ref_blocks)})


def checkpoint_mode_note(has_keyframes: bool, has_refs: bool):
    """One line naming the checkpoint this run's conditioning actually needs, or None.

    H3 ships as two separate DiTs — ``minimax_h3_fl2va_*`` (t2v / first-last frame) and
    ``minimax_h3_ref2va_*`` (references). Both load through the same detection and neither
    rejects the other's conditioning, so the only failure signal is a bad generation. There
    is no variant marker in the state dict to check against, so the run says what it is
    doing and leaves matching the file to the user.
    """
    if has_keyframes and has_refs:
        return ("this run uses BOTH keyframe pins and reference blocks, which live in "
                "DIFFERENT checkpoints (fl2va pins / ref2va references) — whichever model "
                "is loaded, one of the two is untrained conditioning. Drop the anchor image "
                "or the references, or run them as separate scenes.")
    if has_keyframes:
        return ("keyframe pin conditioning (fl2va) — this needs a minimax_h3_fl2va_* "
                "checkpoint; ref2va weights will read it, but were not trained on it.")
    if has_refs:
        return ("reference conditioning (ref2va) — this needs a minimax_h3_ref2va_* "
                "checkpoint; fl2va weights will read it, but were not trained on it.")
    return None


# ── interior keyframe pins ────────────────────────────────────────────────────
#
# A pin is a block of condition rows carrying ONE time coordinate (``cond_t``) on the
# same spatial grid as the target video, never denoised (``img_update=False``). Upstream
# ``PackedLayout`` computes that coordinate in two hard-coded branches and raises for
# anything else:
#
#     pixel_index == 0                -> cond_t = text_len
#     pixel_index == frame_count - 1  -> cond_t = text_len + sum(_video_t_spans(latent_t))
#                                                 - FRAME_RESCALE
#
# Those are not two special cases. They are the two ENDPOINTS of one straight line: the
# video time axis advances exactly ``FRAME_RESCALE`` per PIXEL frame, so
#
#     cond_t = text_len + FRAME_RESCALE * pixel_index
#
# reproduces both, exactly, at every clip length — including the irregular
# ``FRAME_PER_TOKEN = (1, 4, 4, 4, 4)`` latent grouping, because the grouping changes how
# many pixel frames a LATENT frame covers without changing the per-pixel-frame rate.
# :func:`_linear_rule_matches_upstream` re-derives that equality from upstream's own
# constants before anything is patched, so an upstream change to the grid turns this
# feature off instead of silently mis-placing pins.
#
# What this does NOT establish is that the WEIGHTS were trained on interior pins. fl2va
# saw condition rows at t=0 and t=end and nowhere between. MM-RoPE is continuous in t, so
# an interior coordinate is representable rather than undefined, and the pin reaches the
# target through ordinary self-attention with no special-cased path — but "representable"
# is not "learned". Treat an interior pin as experimental until a run shows the frame
# actually landing.
KEYFRAME_T_PER_FRAME = 5.0 / 3.0     # upstream FRAME_RESCALE, re-checked before patching

_INTERIOR_PINS = {"state": None}     # None = not tried, True/False = patch verdict


def keyframe_cond_t(text_len, pixel_index):
    """The packed-sequence time coordinate of a pin at ``pixel_index``."""
    return float(text_len) + KEYFRAME_T_PER_FRAME * float(int(pixel_index))


def _linear_rule_matches_upstream():
    """True when :func:`keyframe_cond_t` reproduces upstream's own two branches.

    Checked against ``_video_t_spans`` and ``FRAME_RESCALE`` themselves rather than
    against remembered numbers, so a changed frame grid is caught here instead of
    becoming pins placed at the wrong instant.
    """
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE, _video_t_spans

    if abs(float(FRAME_RESCALE) - KEYFRAME_T_PER_FRAME) > 1e-9:
        return False
    text_len = 7                                     # any value; it cancels
    for latent_t in (1, 2, 5, 6, 11, 24, 25, 26):
        spans = _video_t_spans(latent_t)
        frame_count = sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))
        first = float(text_len)
        last = float(text_len) + sum(spans) - FRAME_RESCALE
        if abs(first - keyframe_cond_t(text_len, 0)) > 1e-9:
            return False
        if abs(last - keyframe_cond_t(text_len, frame_count - 1)) > 1e-9:
            return False
    return True


def install_interior_keyframes():
    """Let a keyframe pin sit at any pixel frame, not only the first or the last.

    Deliberately NOT a re-implementation of ``PackedLayout.__init__``. Upstream builds the
    layout exactly as it always has — with every interior pin declared at index 0, which is
    always legal — and this rewrites the ``t`` column of the resulting condition rows to the
    coordinate the pin actually wants. Everything else about the layout (row order, latent
    placement, ``img_update``, the segment table, refs, audio) is upstream's, so upstream
    changes are inherited rather than frozen into a copy here.

    A run whose pins are all endpoints never enters the rewrite at all: it calls straight
    through to the original, so existing behaviour is bit-identical.

    Returns True when interior pins are available.
    """
    if _INTERIOR_PINS["state"] is not None:
        return _INTERIOR_PINS["state"]
    try:
        from comfy.ldm.minimax.model import PackedLayout
    except Exception as error:                                    # no H3 in this ComfyUI
        _INTERIOR_PINS["state"] = False
        print(f"[FunPack H3] interior keyframe pins unavailable ({error}).")
        return False
    if not _linear_rule_matches_upstream():
        _INTERIOR_PINS["state"] = False
        print("[FunPack H3] interior keyframe pins DISABLED: this ComfyUI's video time grid "
              "no longer matches the rule FunPack derives pin coordinates from. Pins stay "
              "first/last only. (Upstream comfy/ldm/minimax/model.py changed.)")
        return False
    if getattr(PackedLayout.__init__, "_funpack_interior_pins", False):
        _INTERIOR_PINS["state"] = True
        return True

    original = PackedLayout.__init__

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t,
                 keyframes=None, refs=None, frame_count=None):
        pins = list(keyframes or ())
        interior = [kf for kf in pins
                    if not keyframe_is_endpoint(kf.get("resolved_frame_index"), frame_count)]
        if not interior:
            original(self, text_len, latent_t, latent_h, latent_w, audio_t,
                     keyframes=keyframes, refs=refs, frame_count=frame_count)
            return
        # Declared at 0 so upstream's branch accepts them; the coordinate is corrected below.
        safe = [dict(kf, resolved_frame_index=0) for kf in pins]
        original(self, text_len, latent_t, latent_h, latent_w, audio_t,
                 keyframes=safe, refs=refs, frame_count=frame_count)
        # Condition segments are emitted in `keyframes` order, immediately after text.
        spans = [(a, b) for a, b, kind in self.segments if kind == "cond"]
        for (start, stop), kf in zip(spans, pins):
            self.position_ids[start:stop, 0] = keyframe_cond_t(
                text_len, kf.get("resolved_frame_index", 0))

    __init__._funpack_interior_pins = True
    PackedLayout.__init__ = __init__
    _INTERIOR_PINS["state"] = True
    return True


def keyframe_is_endpoint(frame_index, frame_count):
    """True for the first or last pixel frame — the two pins upstream places unaided."""
    if frame_index is None:
        return False
    idx = int(frame_index)
    if idx == 0:
        return True
    return frame_count is not None and idx == int(frame_count) - 1


def keyframe_indices_supported(frame_index, frame_count):
    """Can a pin be placed at this pixel frame?

    The first and last frames always can. Anything between them needs
    :func:`install_interior_keyframes`, which is attempted here rather than assumed, so a
    ComfyUI whose frame grid has moved refuses the pin instead of mis-placing it.
    """
    idx = int(frame_index)
    if idx < 0 or idx > int(frame_count) - 1:
        return False
    if keyframe_is_endpoint(idx, frame_count):
        return True
    return install_interior_keyframes()


def encode_keyframe(vae, image, width, height, frame_index, crop="disabled"):
    """VAE-encode one pixel frame into an H3 keyframe pin entry.

    `crop` mirrors the reference node: the frame-0 anchor is stretched to the
    canvas (it defines the geometry), a last-frame follower is cover-cropped.
    """
    import comfy.utils
    # one frame only: a pin is a single condition frame, and an IMAGE batch here would
    # otherwise encode into a latent with a batch dimension the packed layout cannot place.
    samples = image[:1, ..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, int(width), int(height), "lanczos", crop)
    pixels = samples.movedim(1, -1)
    return {"resolved_frame_index": int(frame_index), "latent": vae.encode(pixels)}


def image_ref_block(vae, image, width, height, size_mode="match"):
    """Encode a pixel image into a ref2va IMAGE block (identity / style reference).

    `size_mode` mirrors the reference node's own control: "match" scales the reference
    (down only, keeping aspect) to the generation's pixel area; "max" uses the reference
    pipeline's 2048px short edge for the best identity fidelity. Reference rows ride
    through every sampling step, so "max" costs real time on every step of every scene.
    """
    import comfy.utils
    h, w = int(image.shape[1]), int(image.shape[2])
    if str(size_mode).lower() == "max":
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / max(1, min(w, h)))
    else:
        scale = min(1.0, math.sqrt((int(width) * int(height)) / max(1, w * h)))
    snap = lambda v: max(CANVAS_MULTIPLE, int(round(v / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)
    tw, th = snap(w * scale), snap(h * scale)
    samples = image[:1, ..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, tw, th, "lanczos", "disabled")
    resized = samples.movedim(1, -1)
    return ({"type": "image", "data": resized},
            {"kind": "image", "latent_h": th // SPATIAL_DOWNSCALE,
             "latent_w": tw // SPATIAL_DOWNSCALE, "latent": vae.encode(resized)})


def audio_ref_block(audio_vae, audio):
    """Encode an AUDIO dict into a ref2va AUDIO block."""
    import torchaudio
    waveform = audio["waveform"]
    sr = int(audio.get("sample_rate", 32000))
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return ({"type": "audio"},
            {"kind": "audio", "ref_audio_t": int(z.shape[-1]), "audio_latent": z})


# ── ref2va: one ordered spec, two consumers ──────────────────────────────────
# A reference has to reach the model twice and in the SAME order both times:
#   * as PRESENTATION — "<Picture 1>: <vision block>" spliced into the Qwen prompt, so
#     the text encoder has seen it and the prompt can refer to "<Picture 1>";
#   * as a LATENT BLOCK — packed into the DiT sequence, never denoised.
# Those two encodes happen in different nodes (Studio owns the CLIP, the Chain Sampler
# owns the VAE), so the ordered spec travels between them on the conditioning and both
# sides walk it with the helpers below. Ordinals are 1-based per type, exactly as the
# reference node numbers them, so a drift in order silently re-points "<Picture 2>".

REF_KINDS = ("image", "audio", "video")


def normalize_ref_spec(raw):
    """Coerce a user/JSON reference list into an ordered list of {kind, filename, ...}.

    Unknown kinds and entries without a filename are dropped rather than guessed at.
    Returns [] for anything unusable, so callers can treat "no references" and "bad
    references" the same way.
    """
    if isinstance(raw, str):
        import json
        try:
            raw = json.loads(raw or "[]")
        except Exception:
            return []
    if isinstance(raw, dict):
        raw = raw.get("references") or raw.get("refs") or []
    if not isinstance(raw, (list, tuple)):
        return []
    out = []
    for item in raw:
        if isinstance(item, str):
            item = {"kind": "image", "filename": item}
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or item.get("type") or "image").strip().lower()
        filename = str(item.get("filename") or item.get("file") or "").strip()
        if kind not in REF_KINDS or not filename:
            continue
        entry = {"kind": kind, "filename": filename}
        if kind == "video" and item.get("audio"):
            entry["audio"] = str(item["audio"]).strip()
        if kind == "image":
            size = str(item.get("size") or "match").strip().lower()
            entry["size"] = size if size in ("match", "max") else "match"
        out.append(entry)
    return out


def load_input_image(filename):
    """Load an image from ComfyUI's input directory as [1, H, W, 3] in 0..1."""
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
        arr = np.array(img).astype("float32") / 255.0
        return torch.from_numpy(arr)[None,]
    except Exception:
        return None


def load_input_audio(filename):
    """Load an audio file from ComfyUI's input directory as comfy's AUDIO dict."""
    import os
    try:
        import folder_paths
        import torchaudio
    except ImportError:
        return None
    path = os.path.join(folder_paths.get_input_directory(), filename)
    if not os.path.isfile(path):
        return None
    try:
        waveform, sample_rate = torchaudio.load(path)
        return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}
    except Exception:
        return None


def load_input_video(filename, fps=FPS, max_seconds=15.0):
    """Decode a video from ComfyUI's input directory to frames [T, H, W, 3] in 0..1.

    ffmpeg, which FunPack already requires for preview/export, rather than a node
    dependency — the reference pipeline wants frames at the model's own 24 fps and the
    tokenizer wants every second frame of that, so decoding once here keeps both views
    of the same clip consistent. Capped at `max_seconds` because reference tokens ride
    through every sampling step: a long clip is a silent, permanent slowdown.
    """
    import os
    import shutil
    import subprocess
    try:
        import folder_paths
        import numpy as np
    except ImportError:
        return None
    path = os.path.join(folder_paths.get_input_directory(), filename)
    if not os.path.isfile(path):
        return None
    ff = shutil.which("ffmpeg")
    if ff is None:
        return None
    # rgb24 rawvideo on stdout: no temp files, no container parsing on our side. The
    # scale filter pins the frame size so the byte stream is a flat T x H x W x 3 block.
    probe = shutil.which("ffprobe")
    width = height = None
    if probe is not None:
        try:
            out = subprocess.run(
                [probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height", "-of", "csv=p=0:s=x", path],
                capture_output=True, text=True, timeout=30)
            width, height = (int(v) for v in out.stdout.strip().split("x")[:2])
        except Exception:
            width = height = None
    if not width or not height:
        return None
    # even dimensions keep the decode simple and match the model's 32-multiple canvas rule
    width -= width % 2
    height -= height % 2
    try:
        proc = subprocess.run(
            [ff, "-v", "error", "-i", path, "-t", str(float(max_seconds)),
             "-vf", f"fps={int(fps)},scale={width}:{height}",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, timeout=300)
    except Exception:
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    frame_bytes = width * height * 3
    count = len(proc.stdout) // frame_bytes
    if count < 5:      # the model's floor: reference videos need at least 5 frames
        return None
    arr = np.frombuffer(proc.stdout[:count * frame_bytes], dtype=np.uint8)
    arr = arr.reshape(count, height, width, 3).astype("float32") / 255.0
    return torch.from_numpy(arr.copy())


def trim_video_to_grid(frames):
    """Trim reference video frames DOWN to the model's 17k+5 grid (never up).

    Trims rather than pads because padding a reference invents motion that was never in
    the source. Returns None when there is not a single whole grid step left, which is
    the same floor the model enforces. Deliberately independent of the generation length:
    Studio and the Chain Sampler both trim this way, so the clip the text encoder saw and
    the clip that was encoded to latents are always the same frames.
    """
    n = int(frames.shape[0])
    while n > 0 and n % FRAME_GRID != FRAME_BASE:
        n -= 1
    if n < FRAME_BASE:
        return None
    return frames[:n]


def video_presentation_frames(frames):
    """The 2 fps view Qwen sees, with its timestamps — the tokenizer's own sampling rate.

    Returns (frames_at_2fps, timestamps_in_seconds). The tokenizer repeat-pads an odd
    count into its 2-frame temporal patch, so an odd number here is fine.
    """
    step = max(1, FPS // 2)
    idx = list(range(0, int(frames.shape[0]), step))
    return frames[idx], [i / 2.0 for i in range(len(idx))]


def video_ref_block(vae, frames, audio_vae=None, audio=None):
    """Encode reference video frames (plus an optional soundtrack) into a ref2va block.

    The block's audio rows pack immediately before its video rows and share a cursor
    origin, which is what ties a soundtrack to its own clip rather than to the target.
    """
    h, w = int(frames.shape[1]), int(frames.shape[2])
    cw, ch = adapt_canvas(w, h)
    if w * h < cw * ch:
        # never upscale a reference: keep its own size, snapped to the canvas multiple
        snap = lambda v: max(CANVAS_MULTIPLE, int(round(v / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)
        cw, ch = snap(w), snap(h)
    import comfy.utils
    samples = frames[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, cw, ch, "lanczos", "disabled")
    resized = samples.movedim(1, -1)

    z = vae.encode(resized)
    audio_latent, ref_audio_t = None, 0
    if audio is not None and audio_vae is not None:
        _item, audio_block = audio_ref_block(audio_vae, audio)
        audio_latent = audio_block["audio_latent"]
        ref_audio_t = audio_block["ref_audio_t"]

    qwen_frames, timestamps = video_presentation_frames(resized)
    item = {"type": "video", "data": qwen_frames, "timestamps": timestamps}
    block = {"kind": "video_audio" if ref_audio_t else "video",
             "latent_t": int(z.shape[2]), "latent_h": ch // SPATIAL_DOWNSCALE,
             "latent_w": cw // SPATIAL_DOWNSCALE, "ref_audio_t": ref_audio_t,
             "latent": z, "audio_latent": audio_latent}
    return item, block


def ref_items_from_spec(spec):
    """Spec -> the tokenizer's `minimax_ref_items` (presentation only, no VAE needed).

    Only images carry pixels into Qwen; audio contributes a bare "<Audio j>: " label
    (the reference node does the same — audio never enters the text encoder). Entries
    whose file will not load are dropped from BOTH lists by the caller pairing this
    with :func:`ref_blocks_from_spec` over the same resolved spec.
    """
    items = []
    for entry in spec:
        kind = entry["kind"]
        if kind == "image":
            image = entry.get("image")
            if image is None:
                continue
            items.append({"type": "image", "data": image})
        elif kind == "audio":
            items.append({"type": "audio"})
        elif kind == "video":
            frames = entry.get("frames")
            if frames is None:
                continue
            # a soundtrack gets its own "<Audio j>" label, emitted BEFORE its "<Video k>",
            # which is how the model learns the two belong to the same clip
            if entry.get("audio_data") is not None:
                items.append({"type": "audio"})
            qwen_frames, timestamps = video_presentation_frames(frames)
            items.append({"type": "video", "data": qwen_frames, "timestamps": timestamps})
    return items


def resolve_ref_spec(spec, load_image=None, load_audio=None, load_video=None):
    """Load every reference's media once, dropping the ones that fail.

    Returns (resolved, skipped) where `resolved` entries carry the loaded media under
    "image" / "audio_data" / "frames". Both the presentation and the latent blocks are
    built from this single resolved list, so the two can never fall out of order.
    """
    load_image = load_image or load_input_image
    load_audio = load_audio or load_input_audio
    load_video = load_video or load_input_video
    resolved, skipped = [], []
    for entry in spec:
        kind = entry["kind"]
        if kind == "image":
            image = load_image(entry["filename"])
            if image is None:
                skipped.append((entry["filename"], "image could not be loaded"))
                continue
            resolved.append({**entry, "image": image})
        elif kind == "audio":
            audio = load_audio(entry["filename"])
            if audio is None:
                skipped.append((entry["filename"], "audio could not be loaded"))
                continue
            resolved.append({**entry, "audio_data": audio})
        elif kind == "video":
            frames = load_video(entry["filename"])
            if frames is None:
                skipped.append((entry["filename"],
                                "video could not be decoded, or is shorter than 5 frames "
                                "(~0.2s) — ffmpeg must be on PATH"))
                continue
            trimmed = trim_video_to_grid(frames)
            if trimmed is None:
                skipped.append((entry["filename"],
                                "video is shorter than one 17-frame grid step (~0.9s at 24 fps)"))
                continue
            out = {**entry, "frames": trimmed}
            # an index-paired soundtrack: the clip's own audio, not a standalone reference
            if entry.get("audio"):
                track = load_audio(entry["audio"])
                if track is None:
                    skipped.append((entry["audio"],
                                    "soundtrack could not be loaded — the video reference "
                                    "is still used, silently"))
                else:
                    out["audio_data"] = track
            resolved.append(out)
    return resolved, skipped


def ref_blocks_from_spec(resolved, vae, width, height, audio_vae=None):
    """Resolved spec -> the DiT's `minimax_refs` blocks, in the same order.

    `vae` encodes image references; `audio_vae` encodes audio ones. An audio reference
    with no audio VAE connected cannot produce a block, so it is dropped AND returned in
    `skipped` — the caller must say so out loud, because the presentation was already
    baked at encode time and every later "<Audio j>" in the prompt now points one
    reference earlier than the user wrote it.

    Reference videos are capped by :func:`load_input_video`'s own `max_seconds`, NOT by
    the generation's length (which the reference node uses): the presentation is baked by
    Studio, which never sees a frame count, so a sampler-side cap would silently show Qwen
    more of the clip than the packed rows contain.
    """
    blocks, skipped = [], []
    for entry in resolved:
        kind = entry["kind"]
        if kind == "image":
            _item, block = image_ref_block(vae, entry["image"], width, height,
                                           size_mode=entry.get("size", "match"))
            blocks.append(block)
        elif kind == "audio":
            if audio_vae is None:
                skipped.append((entry["filename"],
                                "audio reference needs an audio_vae connected"))
                continue
            _item, block = audio_ref_block(audio_vae, entry["audio_data"])
            blocks.append(block)
        elif kind == "video":
            track = entry.get("audio_data")
            if track is not None and audio_vae is None:
                # the clip still goes in, just mute — better than dropping the whole
                # reference over its soundtrack, and the caller reports it either way
                skipped.append((entry["filename"],
                                "video reference used WITHOUT its soundtrack — encoding "
                                "the audio needs an audio_vae connected"))
                track = None
            _item, block = video_ref_block(vae, entry["frames"],
                                           audio_vae=audio_vae, audio=track)
            blocks.append(block)
    return blocks, skipped


def token_tags_length(meta):
    """Length of the ``minimax_token_tags`` run this conditioning carries, or None.

    The DiT walks the tag vector position-by-position across the text span, so a
    conditioning whose token count no longer matches its tags is a hard failure
    (IndexError mid-forward), not a soft one. Anything that changes the token
    count — appending identity tokens, truncating, concatenating prompts — must
    either update the tags or drop the whole feature for H3.
    """
    if not isinstance(meta, dict):
        return None
    tags = meta.get("minimax_token_tags")
    if tags is None:
        return None
    try:
        return int(tags.reshape(-1).shape[0])
    except Exception:
        return None


def tags_match(cond, meta) -> bool:
    """True when `meta`'s token tags still line up with `cond`'s token count."""
    n = token_tags_length(meta)
    if n is None:
        return True  # no tags to invalidate
    try:
        return int(cond.shape[1]) == n
    except Exception:
        return False


def trim_token_tags(meta, want):
    """Cut the tag vector down to `want` positions, keeping the head aligned.

    Tags are built from the tokenizer's INPUT sequence (`embeds.shape[1]` in comfy's
    MiniMaxQwen3VL.forward) while the conditioning is what the encoder returned, so an
    image prompt normally comes back one tag long — the vision block's expansion does not
    survive the encode 1:1. Position i still means position i in both, so trimming the tail
    is exact; the alternative (dropping the tags) would throw away the vision span's
    modality-0 marks and hand the image tokens to the DiT's text modulation.
    """
    n = token_tags_length(meta)
    if n is None or want < 0 or want >= n:
        return meta
    out = dict(meta)
    out["minimax_token_tags"] = meta["minimax_token_tags"].reshape(-1)[:int(want)]
    return out


def extend_token_tags(meta, added_tokens, tag=1):
    """Grow the tag vector by `added_tokens` positions so appended tokens stay legal.

    Tag 1 is the text modality (0 = video, 2 = audio) — appended context tokens
    are text-like, so they take the text adaLN row.
    """
    n = token_tags_length(meta)
    if n is None or added_tokens <= 0:
        return meta
    out = dict(meta)
    tags = meta["minimax_token_tags"].reshape(-1)
    pad = torch.full((int(added_tokens),), int(tag), dtype=tags.dtype, device=tags.device)
    out["minimax_token_tags"] = torch.cat([tags, pad], dim=0)
    return out
