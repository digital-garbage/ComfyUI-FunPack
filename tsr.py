"""Decisiveness: a temperature for what the model draws (Temporal Score Rescaling).

Xu et al., arXiv 2510.01184. Each step's prediction is split into "the picture"
and "the noise it thinks is left"; scaling the noise part by

    r = (eta * s^2 + 1) / (eta * s^2 / k + 1),   eta = ((1 - sigma) / sigma)^2

samples from a sharper (k > 1: more committed, less varied) or broader (k < 1)
version of what the model learned. k = 1 is exactly off. Near pure noise r -> 1,
so it cannot disturb the first steps; `s` sets how early it takes hold.

One number per run, learned from ratings: every run tries a slightly different k
around the key's current value, and each rating pulls the value toward the k that
was liked or away from the one that was disliked. Edits the picture only (the sound can react);
no extra model call.
"""

import json
import math
import os
import random

# The paper's image setting (SD3). ponytail: fixed; learn it too if k alone plateaus.
S = 3.0
LOGK_MAX = 0.3          # k within ~0.74..1.35
EXPLORE = 0.05
LR = 0.5
MODES = ("off", "learned", "manual")


def factor(sigma, k, s=S):
    sigma = min(max(float(sigma), 1e-6), 1.0)
    eta = ((1.0 - sigma) / sigma) ** 2
    return (eta * s * s + 1.0) / (eta * s * s / k + 1.0)


def rescale_x0(x, x0, sigma, k):
    """A rectified-flow x0 prediction at `sigma` with its noise part scaled by r."""
    a = 1.0 - float(sigma)
    if a < 1e-3 or float(sigma) < 1e-4 or k == 1.0:
        return x0
    r = factor(sigma, k)
    eps = (x - a * x0) / float(sigma)
    return (x - float(sigma) * r * eps) / a


def _path(refinement_key):
    try:
        from .conditioning import refinement_state_path
    except ImportError:
        from conditioning import refinement_state_path
    return refinement_state_path(refinement_key, "tsr", prefix="refine_v2", extension="json")


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


def learned_logk(history):
    m = 0.0
    for h in history:
        s = (h["reward"] > 0) - (h["reward"] < 0)
        m += LR * s * (float(h["logk"]) - m)
        m = min(max(m, -LOGK_MAX), LOGK_MAX)
    return m


def choose_k(refinement_key, mode, manual_k=1.0, rng=random):
    """-> (k, note) for this run. A learned k is only pending once the run finishes
    (save_pending), so an interrupted run can never take a finished run's rating."""
    if mode == "manual":
        return float(manual_k), f"manual k {float(manual_k):.3f}"
    if mode != "learned" or not refinement_key:
        return 1.0, ""
    history = _read(refinement_key).get("history") or []
    centre = learned_logk(history)
    logk = min(max(centre + rng.gauss(0.0, EXPLORE), -LOGK_MAX), LOGK_MAX)
    return math.exp(logk), (f"learned k {math.exp(centre):.3f} from {len(history)} rating(s), "
                            f"trying {math.exp(logk):.3f}")


def save_pending(refinement_key, k):
    if not refinement_key:
        return
    data = _read(refinement_key)
    data["pending"] = math.log(k)
    _write(refinement_key, data)


def clear_pending(refinement_key):
    """A keyed run not learning k: its rating must not score an older run's k."""
    if refinement_key and "pending" in _read(refinement_key):
        data = _read(refinement_key)
        data.pop("pending", None)
        _write(refinement_key, data)


def commit(refinement_key, reward):
    """Pair the pending k with the rating of its run. -> ratings held, or None."""
    if not refinement_key:
        return None
    data = _read(refinement_key)
    logk = data.pop("pending", None)
    if logk is None:
        return None
    if reward:
        history = (data.get("history") or []) + [{"logk": float(logk), "reward": float(reward)}]
        data["history"] = history[-200:]
    _write(refinement_key, data)
    return len(data["history"]) if reward else None
