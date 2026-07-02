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
# Profile readiness: at least this many rating-paired runs (with 2, per-block correlation
# is degenerate — every block correlates ±1 trivially)…
MIN_RATED_RUNS = 3
# …with at least this much reward spread. Attribution is the per-block CORRELATION between
# reward and activity across runs — if every run is rated the same there is nothing to
# attribute, no matter how many runs were paired (the v1 liked/disliked-EMA design failed
# exactly there: 16 same-pole ratings produced zero contrast and looked "not ready").
# Correlation, not raw covariance: run-mean activity is nearly deterministic (averaged
# over ~1200 steps it varies in the 4th decimal run-to-run), so raw covariance is
# numerically dead ("flat activity across runs") — but a tiny variation that CONSISTENTLY
# tracks reward is exactly the signal we want, and correlation is scale-free.
MIN_REWARD_STD = 0.05
# Confidence ramp: scores scale by min(1, (runs - 2) / 6) so a young profile steers
# gently (correlation over 3 runs is noisy) and reaches full strength around 8 runs.
CONFIDENCE_FULL_AT = 8
# …or this many value-credit runs, before scores are considered meaningful.
MIN_CREDIT_RUNS = 2
# Rolling history of (activity fingerprint, reward) pairs kept per key.
HISTORY_MAX = 32
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
    # Store RELATIVE emphasis (mean-normalized): run-global scale differences (step
    # counts, schedulers, scene counts) must not masquerade as per-block signal.
    fp = fp / (fp.mean() + _EPS)
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
    site that trains the output value function). Appends (activity fingerprint, reward)
    to a rolling per-key history — EVERY admissible reward contributes with its actual
    value (no liked/disliked pole thresholds, no dead band) — and folds the run's value
    credit (rating-independent, self-supervised) into the credit EMA. Consumes the
    snapshot so one run is never paired twice. Returns the number of rating-paired runs
    in the history, or None when there was nothing to pair."""
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
            # Start fresh on: different model depth, a v1 pole-EMA profile, or a v2
            # profile whose fingerprints predate mean-normalization (raw scales would
            # poison the correlation).
            if (int(prof.get("n_blocks", -1)) != activity.numel()
                    or ("history_act" not in prof and "liked_n" in prof)
                    or int(prof.get("fp_version", 1)) < 3):
                prof = {}
        prof["n_blocks"] = activity.numel()
        prof["fp_version"] = 3

        row = activity.float().reshape(1, -1)
        rw = torch.tensor([float(reward)], dtype=torch.float32)
        ha, hr = prof.get("history_act"), prof.get("history_rw")
        prof["history_act"] = row if ha is None else torch.cat([ha, row], dim=0)[-HISTORY_MAX:]
        prof["history_rw"] = rw if hr is None else torch.cat([hr, rw], dim=0)[-HISTORY_MAX:]

        credit = snap.get("credit")
        if isinstance(credit, torch.Tensor) and credit.numel() == activity.numel():
            prof["credit"] = _ema(prof.get("credit"), credit, prof.get("credit_n", 0))
            prof["credit_n"] = int(prof.get("credit_n", 0)) + 1

        tmp = prof_path + ".tmp"
        torch.save(prof, tmp)
        os.replace(tmp, prof_path)
        os.remove(snap_path)  # consumed - never pair one run with two ratings
        return int(prof["history_rw"].numel())
    except Exception as e:
        print(f"[FunPackBlockSteer] profile update failed: {e}")
        return None


def _normalize(v):
    v = v - v.mean()
    m = float(v.abs().max())
    return v / m if m > _EPS else None


def block_scores_with_status(refinement_key):
    """(scores, status): combined normalized per-block scores in [-1, 1] (zero-mean,
    unit max-abs), or (None, why-not). Rating attribution is the per-block covariance
    between reward and activity across the run history; value credit blends in when
    trained. `status` is a user-facing sentence for the sampler log."""
    if not refinement_key:
        return None, "no refinement key wired"
    try:
        prof_path = _state_path(refinement_key, "block_profile")
        if not os.path.exists(prof_path):
            return None, "no profile yet — rate a generation made with block_steer on"
        prof = torch.load(prof_path, map_location="cpu", weights_only=False)
        ha, hr = prof.get("history_act"), prof.get("history_rw")
        credit, credit_n = prof.get("credit"), int(prof.get("credit_n", 0))

        rating_part = None
        peak_corr = 0.0
        n = int(hr.numel()) if isinstance(hr, torch.Tensor) else 0
        conf = min(1.0, max(0.0, (n - (MIN_RATED_RUNS - 1)) / float(CONFIDENCE_FULL_AT - (MIN_RATED_RUNS - 1))))
        if not isinstance(ha, torch.Tensor) or n < MIN_RATED_RUNS:
            rating_status = f"needs {MIN_RATED_RUNS}+ rated runs for this key (have {n})"
        elif float(hr.std()) < MIN_REWARD_STD:
            rating_status = (f"{n} rated runs but all rated alike — rate both good AND bad "
                             "runs so blocks have something to contrast")
        else:
            # Per-block Pearson correlation between reward and activity. Scale-free:
            # run-mean activity varies only in the 4th decimal run-to-run, so raw
            # covariance is numerically dead — but consistency with reward isn't.
            rc = hr - hr.mean()
            ac = ha - ha.mean(dim=0)
            corr = (rc.unsqueeze(1) * ac).sum(dim=0) / (
                (rc.square().sum().sqrt() * ac.square().sum(dim=0).sqrt()) + _EPS)
            corr = corr.clamp(-1.0, 1.0)
            peak_corr = float(corr.abs().max())
            rating_part = _normalize(corr)
            rating_status = ("no block's activity varies at all across runs"
                             if rating_part is None else "ready")

        credit_part = _normalize(credit) if (credit is not None and credit_n >= MIN_CREDIT_RUNS) else None

        if rating_part is not None and credit_part is not None:
            combined = RATING_WEIGHT * rating_part + CREDIT_WEIGHT * credit_part
        else:
            combined = rating_part if rating_part is not None else credit_part
        if combined is None:
            return None, rating_status
        scores = _normalize(combined)
        if scores is None:
            return None, "flat combined scores"
        # Confidence ramp: correlation over few runs is noisy — steer gently until the
        # history has substance. Applied AFTER normalization so it survives into gains.
        scores = scores * conf
        return scores, (f"ready ({n} runs, peak |corr| {peak_corr:.2f}, "
                        f"confidence {conf:.2f})")
    except Exception as e:
        return None, f"score load failed: {e}"


def load_block_scores(refinement_key):
    """Scores only (None while not ready) — see block_scores_with_status."""
    return block_scores_with_status(refinement_key)[0]


def gains_from_scores(scores, strength):
    """Per-block gains 1 + strength*score, hard-clamped to 1 +/- MAX_GAIN_DELTA."""
    s = max(0.0, float(strength))
    g = 1.0 + (s * scores).clamp(-MAX_GAIN_DELTA, MAX_GAIN_DELTA)
    return [float(x) for x in g]
