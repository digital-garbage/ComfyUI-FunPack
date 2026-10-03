"""Loop: seamless looping by rolling the latent in time (Mobius, arXiv:2502.20307).

On eligible denoise steps the latent is cyclically rolled along its time axis before the forward pass
and the prediction rolled back after, so the model meets the clip's seam (last frame | first frame) as an
interior pair at a step-varying position and smooths it like any other cut. No training, no extra
forward passes.
"""

import torch

from ..._core import log

# Seamless looping without training or extra forward passes: on eligible denoise steps
# the latent is cyclically rolled along its temporal dim before the forward and the
# prediction unrolled after. The model then repeatedly sees the video's seam
# (last frame | first frame) as an INTERIOR frame pair at a step-varying position and
# smooths it like any other cut — after enough steps the sequence is consistent under
# rotation, i.e. it loops. WanVideoWrapper's "Loop Args" is the same trick for WAN.
#
# Roll starts below the near-noise plateau: content is ~pure noise up there, so there is
# nothing to smooth. Appended guide frames (carry_i2v_guides / mid-scene guide / JoyAI memory /
# custom stacks) live at the TAIL of the same T range, pinned to absolute positions by
# keyframe_idxs — so only the content region [0, T - tail) rolls and the tail (plus
# keyframe_idxs and attention entries, which reference tail tokens) stays canonical.
# Guides keep informing every rolled position (their influence smears across the cycle),
# which is exactly what identity/style references should do in a loop; they are never
# rendered as frames themselves.
LOOP_ROLL_MAX_SIGMA = 0.95
LOOP_ROLL_MIN_FRAMES = 4  # below this many CONTENT latent frames there is nothing to loop


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


def _loop_content_shift(frac, t, tail):
    """Frame shift for a stream with `t` total frames of which the last `tail` are pinned
    guide frames. 0 when the content region is too small to roll."""
    tc = int(t) - max(0, int(tail))
    if tc < 2:
        return 0, tc
    return int(round(frac * tc)) % tc, tc


def _loop_roll_stream(stream, frac, direction, tail):
    """Roll only the content region [0, T-tail) of one stream's temporal dim (dim 2);
    the pinned guide tail stays canonical. Returns the input unchanged when there is
    nothing to roll."""
    shift, tc = _loop_content_shift(frac, stream.shape[2], tail)
    if not shift:
        return stream
    out = stream.clone()
    out[:, :, :tc] = torch.roll(stream[:, :, :tc], shifts=direction * shift, dims=2)
    return out


def _loop_roll_packed(x, shapes, frac, direction, tails=None):
    """Cyclically roll every stream's content region of a packed [B,1,N] tensor by
    round(frac * T_content) frames (video and audio each by their own content length, so
    the two stay time-aligned). `tails` gives per-stream pinned tail-frame counts aligned
    with `shapes` (appended guides / audio memory — left canonical). direction=+1 rolls,
    -1 unrolls. Returns a new tensor."""
    import math as _math
    if x.ndim == 5:  # unpacked single stream
        return _loop_roll_stream(x, frac, direction, (tails or [0])[0])
    out = x.clone()
    off = 0
    for i, dims in enumerate(shapes):
        sz = _math.prod(dims[1:])
        if len(dims) >= 3 and dims[2] >= 2:
            stream = x[..., off:off + sz].reshape([x.shape[0]] + list(dims[1:]))
            rolled = _loop_roll_stream(stream, frac, direction, (tails or [])[i] if tails else 0)
            if rolled is not stream:
                out[..., off:off + sz] = rolled.reshape(x.shape[0], 1, sz)
        off += sz
    return out


def _loop_roll_mask(mask, frac, direction, tail=0):
    """Roll a denoise mask ([B,1,T,H,W] video / [B,C,T,F] audio) in step with its stream,
    leaving the pinned tail region canonical like the stream itself."""
    if not isinstance(mask, torch.Tensor) or mask.ndim < 3 or mask.shape[2] < 2:
        return mask
    return _loop_roll_stream(mask, frac, direction, tail)


def _loop_video_tail_frames(c, video_shape):
    """Number of appended guide latent frames pinned at the video stream's tail. 0 when
    the call carries no guides; None when a guide context exists but can't be parsed
    (caller then leaves the call canonical rather than risking a corrupting roll).
    keyframe_idxs is per-TOKEN — frames = tokens / (H*W), same math as comfy's
    get_keyframe_idxs."""
    kf = c.get("keyframe_idxs")
    if kf is None:
        # Attention entries without keyframe_idxs shouldn't happen; treat as unparseable.
        return None if c.get("guide_attention_entries") else 0
    try:
        if len(video_shape) != 5 or not hasattr(kf, "shape") or len(kf.shape) < 3:
            return None
        tokens_per_frame = int(video_shape[-2]) * int(video_shape[-1])
        n = int(kf.shape[2]) // max(1, tokens_per_frame)
        return n if 0 < n < int(video_shape[2]) else None
    except Exception:
        return None


def _loop_audio_tail_frames(mask):
    """Appended audio-memory frames at the tail of the audio stream = the trailing run of
    all-zero frames in its denoise mask (JoyAI paired audio memory rides mask=0; genuine
    audio content is never fully masked). 0 when there is no mask or no zero tail."""
    if not isinstance(mask, torch.Tensor) or mask.ndim < 3 or mask.shape[2] < 2:
        return 0
    try:
        other_dims = [d for d in range(mask.ndim) if d != 2]
        per_frame = mask.float().abs().amax(dim=other_dims)  # [T]
        tail = 0
        for v in reversed(per_frame.tolist()):
            if v > 1e-4:
                break
            tail += 1
        return tail if tail < mask.shape[2] else 0
    except Exception:
        return 0


def _roll_failed(e, call, args):
    log.once("temporal_style:loop_failed", log.ALERT, "FunPack Temporal style",
             f"the loop roll failed ({type(e).__name__}: {e}); the calls it failed on ran without it")
    return call(args) if call else None


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
            state["count"], state["step_sigma"] = 0, None  # sigma went back up: new scene/run
        state["last_sigma"] = sigma

        c = args.get("c") or {}
        if sigma > LOOP_ROLL_MAX_SIGMA:
            return _call(args)
        shapes = _loop_stream_shapes(args)
        if not shapes:
            return _call(args)
        # Per-stream pinned tails: appended guide frames (video, counted from the
        # per-token keyframe_idxs) and appended audio memory (trailing mask=0 run).
        # Only the content region in front of each tail rolls; the tail — and the
        # keyframe_idxs / attention entries that reference it — stays canonical.
        video_shape = next((s for s in shapes if len(s) == 5), shapes[0])
        v_tail = _loop_video_tail_frames(c, video_shape)
        if v_tail is None:
            return _call(args)  # guide context we can't parse: never risk a bad roll
        tails = [
            (v_tail if len(s) == 5 else _loop_audio_tail_frames(c.get("audio_denoise_mask")))
            for s in shapes
        ]
        t_content = int(video_shape[2]) - v_tail if len(video_shape) >= 3 else 0
        if t_content < LOOP_ROLL_MIN_FRAMES:
            return _call(args)

        if state.get("step_sigma") is None or abs(sigma - state["step_sigma"]) > 1e-6:
            state["count"] += 1                       # a new step; cond and uncond calls of one step share a roll
            state["step_sigma"] = sigma
            state["frac"] = _van_der_corput(state["count"])
        frac = state["frac"]
        if not state["logged"]:
            state["logged"] = True
            tail_note = f", {v_tail} guide frame(s) pinned" if v_tail else ""
            log.once("temporal_style:loop", log.INFO, "FunPack Temporal style", f"loop: Mobius latent roll active (T={t_content}{tail_note})")
        try:
            new_c = dict(c)
            if isinstance(new_c.get("denoise_mask"), torch.Tensor):
                new_c["denoise_mask"] = _loop_roll_mask(new_c["denoise_mask"], frac, 1, tail=v_tail)
            if isinstance(new_c.get("audio_denoise_mask"), torch.Tensor):
                new_c["audio_denoise_mask"] = _loop_roll_mask(
                    new_c["audio_denoise_mask"], frac, 1,
                    tail=_loop_audio_tail_frames(new_c["audio_denoise_mask"]))
            rolled = dict(args)
            rolled["input"] = _loop_roll_packed(args["input"], shapes, frac, 1, tails=tails)
            rolled["c"] = new_c
        except Exception as e:                         # noqa: BLE001 -- only the roll is ours to catch
            return _roll_failed(e, _call, args)
        out = _call(rolled)                            # the model's errors are the model's: not caught here
        try:
            return _loop_roll_packed(out, shapes, frac, -1, tails=tails)
        except Exception as e:                         # noqa: BLE001
            _roll_failed(e, None, None)
            return out

    return _loop_wrapper


