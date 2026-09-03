"""REINS-style representation steering for MiniMax H3.

Everything else this project has built for ratings acts OUTSIDE the model: an attention
bias on the prompt's own tokens (h3_token_weights), or a push on the latent between denoiser
calls (output_guidance, trajectory_guidance). Neither changes what the model itself computes.

This does. It captures the video rows of one block's hidden state per generation, keeps a
running direction from how each generation was rated, and adds that direction back into the
same block's output on the next run -- reaching inside the forward pass instead of around it.
Training-free: no gradient, no MLP, a mean difference of vectors is the whole "model".
("Pulling The REINS", arXiv:2606.17257 -- same idea, applied here via a two-group mean
difference instead of their supervised-PCA.)

Liked vs disliked by the rating's SIGN only, not weighted by magnitude. A magnitude-weighted
version (a 9/10 pulling harder than a 6/10) was tried first and is the statistically better
estimator in the limit -- but it needs enough data for a weakly-rated row's noise to average
out, and at the handful of ratings a key actually has, a mediocre "6" just gets a vote and
dilutes the direction instead of being excluded by it. Binary spends what little data exists
on the clearest examples only; see direction()'s docstring and the 2026-09-04 session before
reintroducing weighting. The weight is still the rating's own `reward` -- already
admissibility-filtered by the caller (conditioning.py only commits inside the same
`_v2_reward_admissible` gate the value functions use), so a Wrong-* identity-mismatch rating
never reaches here at all, not because its number was excluded but because it was never a
quality verdict to begin with.

Masked to VIDEO rows only, using the model's own `mod_segments` tags (0=video, 1=text,
2=audio) -- never audio or text rows. The one documented failure of this class of
intervention (`project_self_consistency_failed.md`) happened on a DIFFERENT model (LTXAV)
by overwriting rows fed back through joint audio-video attention; H3 has no separate
audio-attention path to spare, so touching audio rows here risks the same corruption for a
worse reason. This is the mitigation, not a guarantee -- unvalidated until it's actually run.
"""
from __future__ import annotations

import os

import torch

# Roughly half of H3's 50 blocks -- REINS reports this FRACTION as their sweet spot (their
# own notation: block l in {0..L-1}, effect peaking ~50% of L), on whatever model they tested
# -- not H3, and their own paper picks the actual best layer per model by sweep rather than
# trusting 50% universally. This is that same sweep, run on H3 with real rated data instead
# of guessed from someone else's percentage. Only DEFAULT_BLOCK ever STEERS; the rest of
# CANDIDATE_BLOCKS are captured read-only alongside it so block_sweep() has something to rank.
DEFAULT_BLOCK = 25
CANDIDATE_BLOCKS = sorted({5, 10, 15, 20, 25, 30, 35, 40, 45, DEFAULT_BLOCK})

MIN_PER_GROUP = 2  # need 2+ POSITIVE-weight and 2+ NEGATIVE-weight rows. Was 3 (parity with
# absolute/taste steering's floor, not derived from anything specific to this mechanism).
# 1 would mean the "direction" is literally one liked descriptor minus one disliked one --
# no averaging, as vulnerable to a single noisy capture as anything else this project has
# learned not to trust from one data point. 2 still averages a pair on each side while
# halving the wait (4 rated runs instead of 6) -- lower this only if you want to see raw
# per-pair noise, not a real direction. A weight near zero still gets logged and still
# contributes, just barely -- it does not count toward this floor.


def _log():
    try:
        from . import funpack_log as fl
    except ImportError:
        import funpack_log as fl
    return fl


def state_path(refinement_key):
    try:
        from .conditioning import refinement_state_path
    except ImportError:
        from conditioning import refinement_state_path
    return refinement_state_path(refinement_key, "repr_steer", prefix="refine_v2", extension="pt")


def video_mask_from_mod_segments(mod_segments, seq_len, device):
    """mod_segments (from the H3 block hook's own args) -> a [seq_len] bool mask, True on
    VIDEO rows. Each entry is (a, b, row) where row is either a scalar `t_row*3 + tag` or,
    for masked/multi-timestep rows, a tensor of them -- tag is `row % 3`, and 0 is video by
    construction (`seg_tag` in model.py: video/cond/ref_img all tag 0). Returns None if
    nothing matches, which happens on a shape this has never seen -- callers no-op rather
    than guess."""
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    found = False
    for a, b, row in mod_segments or ():
        if torch.is_tensor(row):
            tags = (row % 3) == 0
            if tags.any():
                mask[a:b][tags] = True
                found = True
        elif int(row) % 3 == 0:
            mask[a:b] = True
            found = True
    return mask if found else None


def capture(hidden_state, video_mask):
    """[seq_len, hidden] + a video-row mask -> a [hidden] descriptor, or None.

    Mean over rows, nothing more -- hidden_size is architecturally fixed (H3 is one width
    throughout), so there is no varying dimension to pool away the way a latent's spatial
    size needs adaptive_avg_pool1d for."""
    if video_mask is None or not bool(video_mask.any()):
        return None
    return hidden_state[video_mask].detach().float().mean(dim=0)


# --- persistence -------------------------------------------------------------------------
#
# One row per generation: {"desc": {block: tensor}, "weight": float, "prompt_hash": str|None}.
# Kept as a log and the direction recomputed from it on demand -- same reasoning
# trajectory_guidance's train_from_rows gives: this is cheap enough that being able to
# re-derive it (a log carried from another box, a fixed weighting formula) is worth more than
# the saved cycles of an online update. `desc` captures EVERY candidate block, not just the
# one that steers, so block_sweep() can be answered from data already being collected instead
# of needing its own separate collection pass.

def _load(refinement_key):
    path = state_path(refinement_key)
    if not os.path.exists(path):
        return {"rows": [], "pending": None, "enabled": True}
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001
        return {"rows": [], "pending": None, "enabled": True}
    if not isinstance(data, dict):
        return {"rows": [], "pending": None, "enabled": True}
    data.setdefault("rows", [])
    data.setdefault("pending", None)
    data.setdefault("enabled", True)
    return data


def capture_enabled(refinement_key):
    """Whether this key is recording at all -- independent of the sampler's own
    h3_repr_steering toggle, which also controls whether the learned direction gets APPLIED.
    A user who wants to pause data collection (to test something else without polluting the
    log, or because a sweep came back null and they want to think before adding more rows)
    still needs the widget on for `block_sweep`/status to mean anything, so the pause lives
    here instead of overloading that widget."""
    if not refinement_key:
        return True
    return bool(_load(refinement_key).get("enabled", True))


def set_capture_enabled(refinement_key, enabled):
    if not refinement_key:
        return
    data = _load(refinement_key)
    data["enabled"] = bool(enabled)
    try:
        _save(refinement_key, data)
    except OSError as e:
        _log().failed("H3 representation steering", "set capture enabled", e,
                       "the pause/resume did not persist")


def _save(refinement_key, data):
    path = state_path(refinement_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, path)


def save_pending(refinement_key, descriptors):
    """Written on EVERY captured run, win or lose -- same fix trajectory_probe needed: a
    pending capture must not survive to be paired with the NEXT run's rating instead of its
    own, and the only way to guarantee that is to overwrite it every time, not just when a
    rating happens to follow. `descriptors` is {block: tensor}, whatever candidate blocks the
    sampler actually captured this run -- not required to cover every entry in
    CANDIDATE_BLOCKS (a shape the hook has never seen just means that block is missing)."""
    if not refinement_key or not descriptors:
        return
    data = _load(refinement_key)
    if not data.get("enabled", True):
        return
    data["pending"] = {int(b): d.detach().float().cpu() for b, d in descriptors.items()}
    try:
        _save(refinement_key, data)
    except OSError as e:
        _log().failed("H3 representation steering", "save pending", e,
                       "this run's capture is not kept")


def commit(refinement_key, reward, prompt_hash=None):
    """Pairs the pending capture with a rating's WEIGHT (and the prompt it was rated under,
    for block_sweep's cross-prompt guard) and appends it to the log. `reward` is the caller's
    already-resolved, already-admissibility-filtered reward -- this function trusts it rather
    than re-deriving a verdict, same as the value functions do with the same number. No
    pending capture (nothing generated since the last rating) is silently a no-op."""
    if not refinement_key:
        return
    data = _load(refinement_key)
    pending = data.get("pending")
    data["pending"] = None
    if pending is not None:
        try:
            data["rows"].append({"desc": pending, "weight": float(reward),
                                 "prompt_hash": prompt_hash})
        except (TypeError, ValueError):
            pass
    try:
        _save(refinement_key, data)
    except OSError as e:
        _log().failed("H3 representation steering", "commit", e, "rating not recorded")


def clear_all(refinement_key):
    if not refinement_key:
        return
    try:
        os.remove(state_path(refinement_key))
    except FileNotFoundError:
        pass


def direction(refinement_key, block=None):
    """-> (unit_vector, n_positive, n_negative), or (None, n_positive, n_negative) if either
    side is under MIN_PER_GROUP. One Awful and one Perfect is not a direction, it is two
    points -- doesn't matter how many barely-positive rows sit between them.

    Plain mean(liked) - mean(disliked), liked/disliked by weight SIGN only -- not weighted by
    magnitude. Weighting by magnitude (a 9/10 pulling harder than a 6/10) is the statistically
    better estimator, but only once there is enough data for a weakly-rated row's noise to
    average out; at the handful of ratings this actually runs on, a mediocre "6" still gets a
    vote and dilutes the direction with an ambiguous case instead of being excluded by it. A
    binary split spends what little data exists on the clearest examples only. Revisit
    weighting once a key has real volume, not before -- see the 2026-09-04 dev session.

    Raw descriptors, not per-vector normalised: a hidden-state row's magnitude carries real
    content, and normalising it away is the trajectory_probe failure this deliberately avoids.

    `block` defaults to DEFAULT_BLOCK -- the one that actually steers. Rows captured before
    that block was in CANDIDATE_BLOCKS (or missing it for any other reason) are skipped, not
    treated as zero."""
    block = DEFAULT_BLOCK if block is None else int(block)
    data = _load(refinement_key)
    rows = [r for r in data["rows"]
            if isinstance(r, dict) and "weight" in r and isinstance(r.get("desc"), dict)
            and block in r["desc"]]
    liked = [r["desc"][block] for r in rows if r["weight"] > 0]
    disliked = [r["desc"][block] for r in rows if r["weight"] < 0]
    n_pos, n_neg = len(liked), len(disliked)
    if n_pos < MIN_PER_GROUP or n_neg < MIN_PER_GROUP:
        return None, n_pos, n_neg
    diff = torch.stack(liked).mean(dim=0) - torch.stack(disliked).mean(dim=0)
    norm = diff.norm()
    if not torch.isfinite(norm) or norm <= 1e-8:
        return None, n_pos, n_neg
    return diff / norm, n_pos, n_neg


def _trajectory_probe():
    try:
        from . import trajectory_probe as tp
    except ImportError:
        import trajectory_probe as tp
    return tp


def block_sweep(refinement_key, trials=2000):
    """-> {block: {"separation":..., "p_value":..., "noise_floor":..., "n":...}}, ranked by
    which of CANDIDATE_BLOCKS actually separates liked from disliked in YOUR rated data --
    reusing trajectory_probe's own permutation test (the one that correctly found real signal
    in the early-schedule work this session, after the centring fix) instead of trusting
    REINS' 50%-depth heuristic or porting a diagnostic built for a different question (LTXAV's
    identity-block search measured cross-scene consistency, not liked-vs-disliked separation).

    Grouped by prompt_hash so "these are different prompts" cannot masquerade as "these are
    different ratings" -- the same confound that made the first cross-prompt read of this
    project's OTHER value function come out backwards. Rows with weight == 0 (a rating that
    landed exactly at the scale's neutral midpoint) are dropped -- neither liked nor disliked.
    """
    tp = _trajectory_probe()
    data = _load(refinement_key)
    rows = [r for r in data["rows"]
            if isinstance(r, dict) and isinstance(r.get("desc"), dict)
            and r.get("weight", 0) != 0]
    out = {}
    for block in CANDIDATE_BLOCKS:
        have = [r for r in rows if block in r["desc"]]
        if len(have) < 2 * MIN_PER_GROUP:
            continue
        descriptors = [r["desc"][block] for r in have]
        labels = [r["weight"] > 0 for r in have]
        groups = [r.get("prompt_hash") for r in have]
        if any(g is None for g in groups):
            groups = None  # can't stratify by a hash some rows never recorded
        result = tp.permutation_test(descriptors, labels, groups=groups, trials=trials)
        if result is not None:
            out[block] = result
    return out
