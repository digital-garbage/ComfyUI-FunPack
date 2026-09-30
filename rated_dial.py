"""One number per run, learned from Liked/Disliked ratings.

Each run tries a value near the key's current centre; the rating of that run pulls the
centre toward the tried value (liked) or pushes it away (disliked). The tried value is
only pending once the run finishes, so an interrupted run never takes a finished run's
rating, and a keyed run with the feature off clears it. Stored as
`<key>.<name>.json`: {"history": [{"v", "reward"}], "pending": v}.
"""

import json
import os
import random

MODES = ("off", "learned", "manual")


MIN_EFFECTS = 3         # runs of history before an effect is compared with the key's own


def unusualness(effect, past, lo=0.25, hi=2.0):
    """Weight of one rating: this run's effect over the key's median past effect, clamped.
    1.0 until enough runs exist to know what usual is."""
    if not effect or len(past) < MIN_EFFECTS:
        return 1.0
    med = sorted(past)[len(past) // 2]
    return min(max(effect / med, lo), hi) if med > 0 else 1.0


class Dial:
    def __init__(self, name, start, lo, hi, explore, lr=0.5):
        self.name, self.start, self.lo, self.hi = name, start, lo, hi
        self.explore, self.lr = explore, lr

    def path(self, refinement_key):
        try:
            from .conditioning import refinement_state_path
        except ImportError:
            from conditioning import refinement_state_path
        return refinement_state_path(refinement_key, self.name, prefix="refine_v2",
                                     extension="json")

    def read(self, refinement_key):
        try:
            with open(self.path(refinement_key), "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, ValueError):
            return {}

    def _write(self, refinement_key, data):
        path = self.path(refinement_key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def learned(self, history):
        m = self.start
        for h in history:
            s = (h["reward"] > 0) - (h["reward"] < 0)
            m += self.lr * s * h.get("w", 1.0) * (float(h["v"]) - m)
            m = min(max(m, self.lo), self.hi)
        return m

    def choose(self, refinement_key, mode, manual, rng=random):
        """-> (value, note); `None` value = off this run."""
        if mode == "manual":
            return float(manual), f"manual {float(manual):.2f}"
        if mode != "learned" or not refinement_key:
            return None, ""
        history = self.read(refinement_key).get("history") or []
        centre = self.learned(history)
        v = min(max(centre + rng.gauss(0.0, self.explore), self.lo), self.hi)
        return v, f"learned {centre:.2f} from {len(history)} rating(s), trying {v:.2f}"

    def save_pending(self, refinement_key, v, effect=None):
        """`effect`: how hard the run's steering actually hit (None = not measured)."""
        if refinement_key:
            data = self.read(refinement_key)
            data["pending"] = float(v)
            data["effect"] = float(effect) if effect else None
            self._write(refinement_key, data)

    def clear_pending(self, refinement_key):
        """A keyed run not learning this: its rating must not score an older run's value."""
        if refinement_key and "pending" in self.read(refinement_key):
            data = self.read(refinement_key)
            data.pop("pending", None)
            data.pop("effect", None)
            self._write(refinement_key, data)

    def commit(self, refinement_key, reward):
        """Pair the pending value with its run's rating. -> ratings held, or None.
        The rating counts more when the run's measured effect was unusually strong for this
        key (and less when unusually weak): a verdict on a run that barely felt the feature
        says little about it."""
        if not refinement_key:
            return None
        data = self.read(refinement_key)
        v = data.pop("pending", None)
        effect = data.pop("effect", None)
        if v is None:
            return None
        weight = unusualness(effect, [h["e"] for h in data.get("history") or [] if h.get("e")])
        if reward:
            data["history"] = ((data.get("history") or [])
                               + [{"v": float(v), "reward": float(reward),
                                                          "w": weight, "e": effect}])[-200:]
        self._write(refinement_key, data)
        return len(data["history"]) if reward else None
