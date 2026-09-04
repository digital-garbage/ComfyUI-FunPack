"""Per-block influence probe for MiniMax H3.

Answers one thing nothing else here measures: how much does each of H3's blocks actually
move the VIDEO rows of the hidden stream, and does that profile differ between generations
you rated liked and ones you rated disliked.

Each block of a residual transformer contributes `x_{i+1} = x_i + f_i(x_i)`. This records
||f_i(x_i)|| / ||x_i|| on video rows only, averaged over every sampling step of a run. Free
in the sense that matters: two norms per block per step, no extra forward pass, no second
model call. Row-subsampled (see samplers._install_block_influence) so the cost does not
scale with H3's 17k+ sequence length.

WHAT THIS PROVES, AND WHAT IT DOES NOT. A large residual delta means the block moved the
stream a lot at that point -- NOT that the movement survived to the final video. The blocks
behind it, and the final layer, can normalise an early push away entirely, and a small late
delta can reach the output untouched. The causal version of this question is ablation: zero
block i, re-run, measure how far the output latent moved. That is ~50 single-step forwards
per probe -- an offline calibration pass, cached per model+LoRA stack (a LoRA changes the
weights, so it changes the map), not something to run per generation. This module is the
free correlational map, and its job is to answer the question that gates the expensive one:
is the depth profile flat, or is it structured? A flat profile means rating-driven per-block
weighting has nothing to grip.

PRIOR ART IN THIS REPO, read before trusting a positive result: per-block residual steering
from ratings was built and REMOVED (b7ed4f9) -- block activity was found to vary in the 4th
decimal with no extractable rating signal. That attempt had no per-step resolution, which
its own removal note named as the missing piece; this one keeps every step separate before
averaging. So this either reproduces that null definitively, or it doesn't. Both outcomes
are worth having, and a null here is the cheaper answer.

H3 is 50 dense uniform blocks with no MoE and no router (see the H3 intervention map), so
there is nothing to REROUTE -- the only levers a block exposes are skip, scale, and add.
This measures which of those would be worth aiming at.
"""
from __future__ import annotations

import os

import torch

# Match h3_repr_steering's floor: a liked-vs-disliked profile difference needs 2+ rated runs
# on each side before it is a difference rather than a pair of points.
MIN_PER_GROUP = 2

_ENV_SWITCH = "FUNPACK_BLOCK_INFLUENCE"
_ENABLED_FILE = "enabled"


def _switch_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "refinements", "block_influence")


def _switch_path():
    return os.path.join(_switch_dir(), _ENABLED_FILE)


def collection_enabled():
    """True when the probe should record. OFF by default -- this is research data, opted
    into from Settings > Refinement & Taste, not something every generation pays for.

    Same two-layer switch trajectory_probe uses, for the same reason: the environment
    variable is live (the sampler runs in this process, so the toggle reaches the very next
    generation with nothing to restart) and wins when set, while the on-disk copy means a
    restarted ComfyUI -- a fresh rental -- comes back recording instead of having quietly
    stopped mid-measurement."""
    raw = os.environ.get(_ENV_SWITCH, "").strip().lower()
    if raw:
        return raw in ("1", "true", "yes", "on")
    try:
        with open(_switch_path(), "r", encoding="utf-8") as fh:
            return fh.read().strip() == "1"
    except (OSError, ValueError):
        return False


def set_collection_enabled(on):
    """Set the switch for this process AND the next one. -> the new state."""
    on = bool(on)
    os.environ[_ENV_SWITCH] = "1" if on else "0"
    try:
        os.makedirs(_switch_dir(), exist_ok=True)
        tmp = _switch_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("1" if on else "0")
        os.replace(tmp, _switch_path())
    except OSError as e:
        _log().failed("H3 block influence", "switch save", e,
                      "recording is set for this session only and reverts after a restart")
    return on


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
    return refinement_state_path(refinement_key, "block_influence", prefix="refine_v2",
                                 extension="pt")


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


def save_pending(refinement_key, profile):
    """Store this run's per-block profile as the pending candidate the next rating scores.
    `profile` is {block_index: mean relative delta}. Overwritten every run, same reason
    h3_repr_steering.save_pending overwrites: a pending capture must never survive to be
    paired with a LATER run's rating."""
    if not refinement_key or not profile:
        return
    data = _load(refinement_key)
    data["pending"] = {int(b): float(v) for b, v in profile.items()}
    try:
        _save(refinement_key, data)
    except OSError as e:
        _log().failed("H3 block influence", "save pending", e, "this run's profile is not kept")


def commit(refinement_key, reward):
    """Pair the pending profile with the rating that scored its run.
    -> "recorded" | "no_pending" | "no_key", same contract h3_repr_steering.commit uses so
    the caller can tell a real commit from a reward that had nothing to attach to."""
    if not refinement_key:
        return "no_key"
    data = _load(refinement_key)
    pending = data.get("pending")
    data["pending"] = None
    if pending is None:
        try:
            _save(refinement_key, data)
        except OSError:
            pass
        return "no_pending"
    try:
        data["rows"].append({"profile": pending, "weight": float(reward)})
    except (TypeError, ValueError):
        try:
            _save(refinement_key, data)
        except OSError:
            pass
        return "no_pending"
    try:
        _save(refinement_key, data)
    except OSError as e:
        _log().failed("H3 block influence", "commit", e, "rating not recorded")
        return "no_pending"
    return "recorded"


def clear_all(refinement_key):
    if not refinement_key:
        return
    try:
        os.remove(state_path(refinement_key))
    except FileNotFoundError:
        pass


def _blocks_in(rows):
    blocks = set()
    for r in rows:
        prof = r.get("profile")
        if isinstance(prof, dict):
            blocks.update(int(b) for b in prof)
    return sorted(blocks)


def profile(refinement_key):
    """-> {"overall": {block: mean}, "liked": {...}|None, "disliked": {...}|None,
           "difference": {block: liked-disliked}|None, "n_liked": int, "n_disliked": int,
           "flatness": float|None}

    `overall` is the depth profile itself -- where this model does its work, regardless of
    ratings. `difference` is the rating-driven signal: blocks whose activity runs higher on
    runs you liked. `flatness` is the spread of `overall` as a fraction of its own mean
    (coefficient of variation): near 0 means every block moves the stream by the same amount
    and there is nothing to aim at; a large value means the depth profile is structured.

    Rows missing a block are skipped for that block rather than counted as zero -- a run that
    hooked fewer blocks must not drag a block's mean down."""
    data = _load(refinement_key)
    rows = [r for r in data["rows"]
            if isinstance(r, dict) and isinstance(r.get("profile"), dict) and "weight" in r]
    blocks = _blocks_in(rows)
    if not rows or not blocks:
        return {"overall": {}, "liked": None, "disliked": None, "difference": None,
                "n_liked": 0, "n_disliked": 0, "flatness": None}

    def _mean(subset):
        out = {}
        for b in blocks:
            vals = [float(r["profile"][b]) for r in subset if b in r["profile"]]
            if vals:
                out[b] = sum(vals) / len(vals)
        return out

    liked_rows = [r for r in rows if r["weight"] > 0]
    disliked_rows = [r for r in rows if r["weight"] < 0]
    overall = _mean(rows)
    flatness = None
    if overall:
        vals = torch.tensor(list(overall.values()), dtype=torch.float32)
        m = float(vals.mean())
        flatness = float(vals.std(unbiased=False) / m) if abs(m) > 1e-12 else None

    liked = disliked = difference = None
    if len(liked_rows) >= MIN_PER_GROUP and len(disliked_rows) >= MIN_PER_GROUP:
        liked, disliked = _mean(liked_rows), _mean(disliked_rows)
        difference = {b: liked[b] - disliked[b] for b in liked if b in disliked}
    return {"overall": overall, "liked": liked, "disliked": disliked,
            "difference": difference, "n_liked": len(liked_rows),
            "n_disliked": len(disliked_rows), "flatness": flatness}
