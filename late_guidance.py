"""Late-branch guidance: push each step's picture away from a deliberately weakened copy.

STG-style skip guidance (the weak copy skips one block), made cheap enough for CFG=1:
both copies share every block before the branch, so the weak one only re-runs the tail.
Branching at 43 of H3's 50 blocks costs ~7 blocks per step (~15%), not a second forward.

    guided = normal + w * (normal - weak)      picture only; sound is the normal pass

One number per run, learned from ratings the same way as decisiveness (tsr.py): each run
tries a strength near the key's current value, and each rating pulls the value toward the
strength that was liked or away from the one that was disliked.
"""

import json
import os
import random

START = 0.5
W_MAX = 1.5
EXPLORE = 0.15
LR = 0.5
MODES = ("off", "learned", "manual")
DEFAULT_BLOCK = 43


def _path(refinement_key):
    try:
        from .conditioning import refinement_state_path
    except ImportError:
        from conditioning import refinement_state_path
    return refinement_state_path(refinement_key, "late_guidance", prefix="refine_v2",
                                 extension="json")


def _read(refinement_key):
    try:
        with open(_path(refinement_key), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError):
        return {}


def _write(refinement_key, data):
    path = _path(refinement_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def learned_w(history):
    m = START
    for h in history:
        s = (h["reward"] > 0) - (h["reward"] < 0)
        m += LR * s * (float(h["w"]) - m)
        m = min(max(m, 0.0), W_MAX)
    return m


def choose_w(refinement_key, mode, manual_w=START, rng=random):
    """-> (w, note). A learned w only becomes pending once the run finishes (save_pending)."""
    if mode == "manual":
        return max(0.0, float(manual_w)), f"manual strength {float(manual_w):.2f}"
    if mode != "learned" or not refinement_key:
        return 0.0, ""
    history = _read(refinement_key).get("history") or []
    centre = learned_w(history)
    w = min(max(centre + rng.gauss(0.0, EXPLORE), 0.0), W_MAX)
    return w, f"learned strength {centre:.2f} from {len(history)} rating(s), trying {w:.2f}"


def mix(normal, weak, w):
    return normal + w * (normal - weak)


def save_pending(refinement_key, w):
    if not refinement_key:
        return
    data = _read(refinement_key)
    data["pending"] = float(w)
    _write(refinement_key, data)


def clear_pending(refinement_key):
    """A keyed run not learning w: its rating must not score an older run's w."""
    if refinement_key and "pending" in _read(refinement_key):
        data = _read(refinement_key)
        data.pop("pending", None)
        _write(refinement_key, data)


def commit(refinement_key, reward):
    """Pair the pending w with the rating of its run. -> ratings held, or None."""
    if not refinement_key:
        return None
    data = _read(refinement_key)
    w = data.pop("pending", None)
    if w is None:
        return None
    if reward:
        data["history"] = ((data.get("history") or []) + [{"w": float(w), "reward": float(reward)}])[-200:]
    _write(refinement_key, data)
    return len(data["history"]) if reward else None
