"""REINS-style representation steering for MiniMax H3.

Everything else this project has built for ratings acts OUTSIDE the model: an attention
bias on the prompt's own tokens (h3_token_weights), or a push on the latent between denoiser
calls (output_guidance, trajectory_guidance). Neither changes what the model itself computes.

This does. It captures the video rows of one block's hidden state per generation, keeps a
running direction (liked runs' mean minus disliked runs' mean), and adds that direction back
into the same block's output on the next run -- reaching inside the forward pass instead of
around it. Training-free: no gradient, no MLP, a mean-difference of vectors is the whole
"model". ("Pulling The REINS", arXiv:2606.17257 -- same idea, applied here via difference-of-
means instead of their supervised-PCA, because a two-group direction is what a mean
difference already gives you exactts.)

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

MIN_PER_GROUP = 3  # "need 3+ liked generations" -- same floor absolute/taste steering uses.

# Rating KEYS this mechanism will learn from -- not the blended scalar reward every other
# mechanism reads. That scalar is a quality-LANDSCAPE score (Missing action = +0.05: "the
# frame was fine, just no motion" -- a near-miss worth a gentle push in the value functions'
# ascent, not a failure) and reusing it here means a near-miss counts as "liked", diluting a
# direction that needs to be a clean liked/disliked split. Everything not listed -- Wrong
# action, Wrong appearance, Missing action, all the near-misses -- is EXCLUDED, not counted
# toward either side: they are genuinely ambiguous for "was this a good generation", and
# excluding beats guessing.
LIKED_KEYS = {"like", "nailed_it", "loved_it"}
DISLIKED_KEYS = {"awful", "missing_quality", "missing_details_quality", "missing_action_quality"}


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
# One row per generation: {"desc": tensor, "reward": float}. Kept as a log and the direction
# recomputed from it on demand -- same reasoning trajectory_guidance's train_from_rows gives:
# a mean-difference is cheap enough that being able to re-derive it (a log carried from
# another box, a bad reward-sign fix) is worth more than the saved cycles of updating two
# running sums in place.

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


def commit(refinement_key, rating_key):
    """Pairs the pending capture with a rating's KEY (not the blended scalar reward every
    other mechanism reads -- see LIKED_KEYS/DISLIKED_KEYS) and appends it to the log if the
    key lands in either group. A key that lands in neither (Missing action, Wrong appearance,
    every near-miss) discards the pending capture without logging a row -- ambiguous evidence
    is excluded, not counted toward the side its reward sign happened to fall on."""
    if not refinement_key:
        return
    data = _load(refinement_key)
    pending = data.get("pending")
    data["pending"] = None
    key = str(rating_key or "")
    if pending is not None and (key in LIKED_KEYS or key in DISLIKED_KEYS):
        data["rows"].append({"desc": pending, "liked": key in LIKED_KEYS})
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
    """-> (unit_vector, n_liked, n_disliked), or (None, n_liked, n_disliked) if either group
    is under MIN_PER_GROUP. One Awful and one Perfect is not a direction, it is two points.

    Raw means, not per-vector normalised: a hidden-state row's magnitude carries real content
    (this is NOT the trajectory_probe failure mode). Shared content sits in BOTH groups'
    means equally and cancels in the subtraction by construction -- centring only matters
    when a vector is compared to itself normalised, which nothing here does."""
    data = _load(refinement_key)
    liked = [r["desc"] for r in data["rows"] if isinstance(r, dict) and r.get("liked") is True]
    disliked = [r["desc"] for r in data["rows"] if isinstance(r, dict) and r.get("liked") is False]
    if len(liked) < MIN_PER_GROUP or len(disliked) < MIN_PER_GROUP:
        return None, len(liked), len(disliked)
    diff = torch.stack(liked).mean(dim=0) - torch.stack(disliked).mean(dim=0)
    norm = diff.norm()
    if not torch.isfinite(norm) or norm <= 1e-8:
        return None, len(liked), len(disliked)
    return diff / norm, len(liked), len(disliked)
