"""Block Steering (EXPERIMENTAL): rating-learned per-block residual gains for LTX-AV.

RLHF at the MODEL level instead of the conditioning level. The rating system already
attributes quality to sampling settings and conditioning; this adds the missing
coordinate system - which REGIONS OF THE MODEL were doing the work when a generation
came out good or bad - and then steers per-block application accordingly.

Three stages, all riding existing FunPack infra:

1. MEASURE - forward hooks on the transformer blocks record, per model call, each
   block's relative residual delta on the VIDEO stream: ||vx_out - vx_in|| / ||vx_in||.
   A block that barely moves the hidden state is coasting; one with a large relative
   delta is doing the work for this prompt at this noise level. Hooks are tagged with
   ltx_enhancements' _FUNPACK_HOOK_TAG so the existing leak-strip covers them.

2. ATTRIBUTE - two signals, folded into a per-key `block_profile.pt` sidecar:
   * run-level: the end-of-run mean activity fingerprint is staged (`block_snapshot.pt`,
     same stage-then-pair convention as x0_snapshot) and paired with the user's rating
     reward on the next Refiner cycle - liked / disliked / all-runs EMA profiles.
   * step-level: when output_guidance's value function is trained, a model-function
     wrapper scores each step's x0_hat (one tiny-MLP forward, no extra model pass) and
     credits blocks by (value delta this step) x (their activity this step, centered).
     Converges in a handful of runs - matters because keys are retrained per instance.

3. STEER - per-block gains on the video residual only: vx_out' = vx_in + g_b*(vx_out -
   vx_in), g_b = 1 + strength * score_b, scores zero-mean/unit-max so emphasis is
   REDISTRIBUTED rather than globally inflated, clamped to +/-MAX_GAIN_DELTA. The audio
   stream is never touched (residual-gain on the video half is the intervention family
   that works on LTX-AV; hidden-state injection through joint attention is the one that
   corrupts audio). Exact no-op until the profile has enough paired data.

Everything is per-refinement-key and rebuilt from ratings in a few generations - a
mechanism upgrade, not a data store to migrate.
"""

from __future__ import annotations

import os

import torch

try:
    from .ltx_enhancements import _FUNPACK_HOOK_TAG, _funpack_locate_blocks
except ImportError:
    from ltx_enhancements import _FUNPACK_HOOK_TAG, _funpack_locate_blocks

# Steering never moves a block's residual contribution by more than this, regardless of
# strength widget or how lopsided the learned scores are.
MAX_GAIN_DELTA = 0.10
# Profile readiness: at least this many rating-paired runs (liked+disliked pools)…
MIN_RATED_RUNS = 2
# …or this many value-credit runs, before scores are considered meaningful.
MIN_CREDIT_RUNS = 2
# Blend weights when both attribution signals are available.
RATING_WEIGHT = 0.6
CREDIT_WEIGHT = 0.4
_EPS = 1e-8


def _video_stream(x):
    """The video tensor of a block input/output: (vx, ax) tuple on LTX-AV, plain tensor
    on video-only LTXV."""
    if isinstance(x, (tuple, list)) and len(x) >= 1:
        return x[0]
    return x if isinstance(x, torch.Tensor) else None


# ---------------------------------------------------------------------------
# Stage 1 - measurement
# ---------------------------------------------------------------------------

class BlockActivityRecorder:
    """Accumulates per-block relative residual deltas (video stream) across a run.

    `cur` always holds the most recent model call's per-block activity so the value-
    credit wrapper (which runs after all blocks in the same call) can read it.
    """

    def __init__(self, n_blocks: int):
        self.n_blocks = int(n_blocks)
        self.sums = torch.zeros(self.n_blocks, dtype=torch.float32)
        self.counts = torch.zeros(self.n_blocks, dtype=torch.float32)
        self.cur = torch.zeros(self.n_blocks, dtype=torch.float32)
        # value-credit accumulation (filled by the credit wrapper, if installed)
        self.credit = torch.zeros(self.n_blocks, dtype=torch.float32)
        self.credit_steps = 0
        self._prev_score = None

    def record(self, idx: int, rel_delta: float):
        if 0 <= idx < self.n_blocks:
            self.sums[idx] += rel_delta
            self.counts[idx] += 1.0
            self.cur[idx] = rel_delta

    def fingerprint(self):
        """Mean per-block activity over the run, or None if nothing was recorded."""
        if float(self.counts.max()) <= 0:
            return None
        return self.sums / self.counts.clamp(min=1.0)

    def credit_step(self, score: float):
        """Fold one model call's value score into per-block credit: blocks that were
        MORE active than average during a step where predicted quality rose (fell) get
        positive (negative) credit."""
        if self._prev_score is not None:
            delta = float(score) - self._prev_score
            centered = self.cur - self.cur.mean()
            self.credit += delta * centered
            self.credit_steps += 1
        self._prev_score = float(score)

    def credit_vector(self):
        if self.credit_steps <= 0:
            return None
        return self.credit / float(self.credit_steps)


def install_recorder(model, recorder: BlockActivityRecorder):
    """Tagged forward hooks on every transformer block; returns handles (possibly [])."""
    blocks = _funpack_locate_blocks(model)
    if blocks is None:
        return []
    handles = []
    for i, blk in enumerate(blocks):
        def _hook(_module, args, output, _idx=i, _rec=recorder):
            try:
                v_in = _video_stream(args[0] if args else None)
                v_out = _video_stream(output)
                if v_in is None or v_out is None or v_in.shape != v_out.shape:
                    return output
                with torch.no_grad():
                    denom = float(v_in.norm()) + _EPS
                    _rec.record(_idx, float((v_out - v_in).norm()) / denom)
            except Exception:
                pass
            return output
        setattr(_hook, _FUNPACK_HOOK_TAG, True)
        handles.append(blk.register_forward_hook(_hook))
    return handles


def make_credit_wrapper(old_wrapper, recorder: BlockActivityRecorder, value_fn):
    """model_function_wrapper that scores each call's x0_hat with the (already trained)
    output-space value function and folds the step-to-step delta into block credit.
    Installed INNERMOST so it scores the base model's own prediction, before
    embed_guidance / output_guidance corrections layer on top."""

    def _call(apply_fn, a):
        if old_wrapper is not None:
            return old_wrapper(apply_fn, a)
        return apply_fn(a["input"], a["timestep"], **a.get("c", {}))

    def _wrapper(apply_fn, args, _rec=recorder, _vf=value_fn):
        denoised = _call(apply_fn, args)
        try:
            with torch.inference_mode(False), torch.no_grad():
                c = _vf.compress(denoised.detach().float())
                score = float(_vf(c).mean())
            _rec.credit_step(score)
        except Exception:
            pass
        return denoised

    return _wrapper


# ---------------------------------------------------------------------------
# Stage 3 - steering
# ---------------------------------------------------------------------------

def install_steer(model, gains):
    """Tagged forward hooks scaling each block's VIDEO residual by its gain:
    vx' = vx_in + g*(vx_out - vx_in). Audio passes through untouched. Returns handles;
    blocks whose gain is ~1.0 get no hook at all (byte-identical native path)."""
    blocks = _funpack_locate_blocks(model)
    if blocks is None:
        return []
    handles = []
    for i, blk in enumerate(blocks):
        g = float(gains[i]) if i < len(gains) else 1.0
        if abs(g - 1.0) < 1e-4:
            continue
        def _hook(_module, args, output, _g=g):
            try:
                v_in = _video_stream(args[0] if args else None)
                if isinstance(output, tuple) and len(output) == 2:
                    v_out, a_out = output
                    if v_in is None or v_in.shape != v_out.shape:
                        return output
                    return (v_in + _g * (v_out - v_in), a_out)
                if isinstance(output, torch.Tensor) and v_in is not None and v_in.shape == output.shape:
                    return v_in + _g * (output - v_in)
            except Exception:
                pass
            return output
        setattr(_hook, _FUNPACK_HOOK_TAG, True)
        handles.append(blk.register_forward_hook(_hook))
    return handles


def remove_handles(handles):
    for h in handles or []:
        try:
            h.remove()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Stage 2 - persistence + attribution
# ---------------------------------------------------------------------------

def _state_path(refinement_key, name):
    try:
        from .conditioning import refinement_state_path
    except ImportError:
        from conditioning import refinement_state_path
    return refinement_state_path(refinement_key, name, prefix="refine_v2", extension="pt")


def save_run_snapshot(refinement_key, recorder: BlockActivityRecorder):
    """End of run: stage this run's fingerprint (+ optional value credit) so the NEXT
    rating cycle can pair it with a reward. Atomic write - a rating read never sees a
    partial file. Same convention as the sampler's x0_snapshot."""
    if not refinement_key:
        return False
    fp = recorder.fingerprint()
    if fp is None:
        return False
    try:
        path = _state_path(refinement_key, "block_snapshot")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        torch.save({
            "activity": fp.cpu(),
            "credit": (recorder.credit_vector().cpu() if recorder.credit_vector() is not None else None),
            "n_blocks": recorder.n_blocks,
        }, tmp)
        os.replace(tmp, path)
        return True
    except Exception as e:
        print(f"[FunPackBlockSteer] snapshot save failed: {e}")
        return False


def _ema(old, x, n):
    a = max(0.2, 1.0 / (float(n) + 1.0))
    return x if old is None else old * (1.0 - a) + x * a


def update_profile_with_rating(refinement_key, reward):
    """Pair the staged snapshot with a rating reward (called by the Refiner at the same
    site that trains the output value function). Folds:
      * every run into the all-runs mean (the contrast baseline),
      * reward >= 0.3 into the liked profile, reward <= 0.0 into the disliked profile
        (the mild-positive band is too ambiguous to move either pole - same rule as the
        Absolute store),
      * the run's value credit (rating-independent, self-supervised) into the credit EMA.
    Consumes the snapshot so one run is never paired twice. Returns the number of
    rating-paired runs, or None when there was nothing to pair."""
    if not refinement_key:
        return None
    try:
        snap_path = _state_path(refinement_key, "block_snapshot")
        if not os.path.exists(snap_path):
            return None
        snap = torch.load(snap_path, map_location="cpu", weights_only=False)
        activity = snap.get("activity")
        if not isinstance(activity, torch.Tensor):
            return None
        prof_path = _state_path(refinement_key, "block_profile")
        prof = {}
        if os.path.exists(prof_path):
            prof = torch.load(prof_path, map_location="cpu", weights_only=False)
            if int(prof.get("n_blocks", -1)) != activity.numel():
                prof = {}  # different model depth - start fresh
        prof["n_blocks"] = activity.numel()

        prof["all_mean"] = _ema(prof.get("all_mean"), activity, prof.get("all_n", 0))
        prof["all_n"] = int(prof.get("all_n", 0)) + 1
        r = float(reward)
        if r >= 0.3:
            prof["liked"] = _ema(prof.get("liked"), activity, prof.get("liked_n", 0))
            prof["liked_n"] = int(prof.get("liked_n", 0)) + 1
        elif r <= 0.0:
            prof["disliked"] = _ema(prof.get("disliked"), activity, prof.get("disliked_n", 0))
            prof["disliked_n"] = int(prof.get("disliked_n", 0)) + 1
        credit = snap.get("credit")
        if isinstance(credit, torch.Tensor) and credit.numel() == activity.numel():
            prof["credit"] = _ema(prof.get("credit"), credit, prof.get("credit_n", 0))
            prof["credit_n"] = int(prof.get("credit_n", 0)) + 1

        tmp = prof_path + ".tmp"
        torch.save(prof, tmp)
        os.replace(tmp, prof_path)
        os.remove(snap_path)  # consumed - never pair one run with two ratings
        return int(prof.get("liked_n", 0)) + int(prof.get("disliked_n", 0))
    except Exception as e:
        print(f"[FunPackBlockSteer] profile update failed: {e}")
        return None


def _normalize(v):
    v = v - v.mean()
    m = float(v.abs().max())
    return v / m if m > _EPS else None


def load_block_scores(refinement_key):
    """Combined, normalized per-block scores in [-1, 1] (zero-mean, unit max-abs), or
    None while the profile lacks enough paired data. Rating contrast is liked-vs-
    disliked when both poles exist, else the available pole against the all-runs mean."""
    if not refinement_key:
        return None
    try:
        prof_path = _state_path(refinement_key, "block_profile")
        if not os.path.exists(prof_path):
            return None
        prof = torch.load(prof_path, map_location="cpu", weights_only=False)
        liked, disliked = prof.get("liked"), prof.get("disliked")
        liked_n, disliked_n = int(prof.get("liked_n", 0)), int(prof.get("disliked_n", 0))
        all_mean, all_n = prof.get("all_mean"), int(prof.get("all_n", 0))
        credit, credit_n = prof.get("credit"), int(prof.get("credit_n", 0))

        rating_part = None
        if liked_n + disliked_n >= MIN_RATED_RUNS:
            if liked is not None and disliked is not None:
                rating_part = _normalize(liked - disliked)
            elif liked is not None and all_mean is not None and all_n >= 2:
                rating_part = _normalize(liked - all_mean)
            elif disliked is not None and all_mean is not None and all_n >= 2:
                rating_part = _normalize(-(disliked - all_mean))
        credit_part = _normalize(credit) if (credit is not None and credit_n >= MIN_CREDIT_RUNS) else None

        if rating_part is not None and credit_part is not None:
            combined = RATING_WEIGHT * rating_part + CREDIT_WEIGHT * credit_part
        else:
            combined = rating_part if rating_part is not None else credit_part
        if combined is None:
            return None
        return _normalize(combined)
    except Exception as e:
        print(f"[FunPackBlockSteer] score load failed: {e}")
        return None


def gains_from_scores(scores, strength):
    """Per-block gains 1 + strength*score, hard-clamped to 1 +/- MAX_GAIN_DELTA."""
    s = max(0.0, float(strength))
    g = 1.0 + (s * scores).clamp(-MAX_GAIN_DELTA, MAX_GAIN_DELTA)
    return [float(x) for x in g]
