"""REINS-style representation steering for MiniMax H3.

Everything else this project has built for ratings acts OUTSIDE the model: an attention
bias on the prompt's own tokens (h3_token_weights), or a push on the latent between denoiser
calls (output_guidance, trajectory_guidance). Neither changes what the model itself computes.

This does. It captures the video rows of one block's hidden state per generation, keeps a
running direction weighted by how each generation was actually rated, and adds that direction
back into the same block's output on the next run -- reaching inside the forward pass instead
of around it. Training-free: no gradient, no MLP, a weighted covariance of vectors is the
whole "model". ("Pulling The REINS", arXiv:2606.17257 -- same idea, applied here via a
weighted direction instead of their supervised-PCA.)

Weighted, not a hard liked/disliked split: a 9/10 pulls harder than a 6/10, a 2/10 pulls
moderately, a 5/10 contributes almost nothing. `direction proportional to sum((w_i - mean(w))
* desc_i)` -- centring the WEIGHTS (not the descriptors) makes shared content cancel exactly
regardless of how unbalanced the weights are (sum(w_i - mean(w)) is 0 by construction, so
whatever every descriptor has in common multiplies out to zero), and reduces to the old plain
mean(liked) - mean(disliked) in the equal-and-opposite case. The weight is the rating's own
`reward` -- already admissibility-filtered by the caller (conditioning.py only commits inside
the same `_v2_reward_admissible` gate the value functions use), so a Wrong-* identity-mismatch
rating never reaches here at all, not because its number was excluded but because it was
never a quality verdict to begin with.

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

# Roughly half of H3's 50 blocks -- REINS measured this depth as the sweet spot: early
# enough that the rest of the stack can still act on the change, late enough that the
# concept the direction encodes has actually formed.
DEFAULT_BLOCK = 25

MIN_PER_GROUP = 3  # need 3+ POSITIVE-weight and 3+ NEGATIVE-weight rows -- same floor
# absolute/taste steering uses. A weight near zero (Missing action's +0.05) still gets
# logged and still contributes, just barely -- it does not count toward this floor.


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
# One row per generation: {"desc": tensor, "weight": float}. Kept as a log and the direction
# recomputed from it on demand -- same reasoning trajectory_guidance's train_from_rows gives:
# this is cheap enough that being able to re-derive it (a log carried from another box, a
# fixed weighting formula) is worth more than the saved cycles of an online update.

def _load(refinement_key):
    path = state_path(refinement_key)
    if not os.path.exists(path):
        return {"rows": [], "pending": None}
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001
        return {"rows": [], "pending": None}
    if not isinstance(data, dict):
        return {"rows": [], "pending": None}
    data.setdefault("rows", [])
    data.setdefault("pending", None)
    return data


def _save(refinement_key, data):
    path = state_path(refinement_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, path)


def save_pending(refinement_key, descriptor):
    """Written on EVERY captured run, win or lose -- same fix trajectory_probe needed: a
    pending capture must not survive to be paired with the NEXT run's rating instead of its
    own, and the only way to guarantee that is to overwrite it every time, not just when a
    rating happens to follow."""
    if not refinement_key or descriptor is None:
        return
    data = _load(refinement_key)
    data["pending"] = descriptor.detach().float().cpu()
    try:
        _save(refinement_key, data)
    except OSError as e:
        _log().failed("H3 representation steering", "save pending", e,
                       "this run's capture is not kept")


def commit(refinement_key, reward):
    """Pairs the pending capture with a rating's WEIGHT and appends it to the log. `reward`
    is the caller's already-resolved, already-admissibility-filtered reward -- this function
    trusts it rather than re-deriving a verdict, same as the value functions do with the same
    number. No pending capture (nothing generated since the last rating) is silently a no-op."""
    if not refinement_key:
        return
    data = _load(refinement_key)
    pending = data.get("pending")
    data["pending"] = None
    if pending is not None:
        try:
            data["rows"].append({"desc": pending, "weight": float(reward)})
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


def direction(refinement_key):
    """-> (unit_vector, n_positive, n_negative), or (None, n_positive, n_negative) if either
    side is under MIN_PER_GROUP. One Awful and one Perfect is not a direction, it is two
    points -- doesn't matter how many barely-positive rows sit between them.

    Weighted covariance between rating and descriptor, not a two-group mean difference: each
    row pulls proportionally to how strongly it was rated, so a 9/10 counts for more than a
    6/10 instead of both counting as one full "liked" vote. Centring the WEIGHTS (not the
    descriptors) before multiplying is what makes shared content cancel exactly regardless of
    how unbalanced the weights are -- see the module docstring for the identity. Raw
    descriptors, not per-vector normalised: a hidden-state row's magnitude carries real
    content, and normalising it away is the trajectory_probe failure this deliberately avoids."""
    data = _load(refinement_key)
    rows = [r for r in data["rows"] if isinstance(r, dict) and "weight" in r]
    n_pos = sum(1 for r in rows if r["weight"] > 0)
    n_neg = sum(1 for r in rows if r["weight"] < 0)
    if n_pos < MIN_PER_GROUP or n_neg < MIN_PER_GROUP:
        return None, n_pos, n_neg
    weights = torch.tensor([r["weight"] for r in rows], dtype=torch.float32)
    descs = torch.stack([r["desc"] for r in rows])
    centred = weights - weights.mean()
    diff = (centred.unsqueeze(1) * descs).sum(dim=0)
    norm = diff.norm()
    if not torch.isfinite(norm) or norm <= 1e-8:
        return None, n_pos, n_neg
    return diff / norm, n_pos, n_neg
