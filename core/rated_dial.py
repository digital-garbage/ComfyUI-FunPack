"""One number per run, learned from Liked/Disliked ratings.

Each run tries a value near the key's current centre; the rating of that run pulls the
centre toward the tried value (liked) or pushes it away (disliked). A rating counts more
when the run's measured effect was unusually strong for this key and less when unusually
weak: a verdict on a run that barely felt the feature says little about it.

Pure arithmetic over rated rows. The caller keeps the rows (the taste store) and fills a
capture `{"v": tensor, "e": tensor}` during the run; this file holds no state.
"""

import random

MIN_EFFECTS = 3         # runs of history before an effect is compared with the key's own


def history(rows, setup=None):
    """Taste-store rows -> [(value, reward, effect | None)], oldest first.

    `setup` is what gives the value its meaning ({"b": block}): a rating made with another
    setup says nothing about this one and is left out. Rows from before a setup was
    recorded are left out too."""
    out = []
    for r in rows:
        p = r["rows"]
        if "v" in p and all(k in p and int(p[k]) == int(want) for k, want in (setup or {}).items()):
            e = float(p["e"]) if "e" in p and float(p["e"]) > 0 else None
            out.append((float(p["v"]), float(r["reward"]), e))
    return out


def unusualness(effect, past, lo=0.25, hi=2.0):
    """Weight of one rating: this run's effect over the key's median past effect, clamped.
    1.0 until enough runs exist to know what usual is."""
    if not effect or len(past) < MIN_EFFECTS:
        return 1.0
    med = sorted(past)[len(past) // 2]
    return min(max(effect / med, lo), hi) if med > 0 else 1.0


class Dial:
    def __init__(self, start, lo, hi, explore, lr=0.5):
        self.start, self.lo, self.hi, self.explore, self.lr = start, lo, hi, explore, lr

    def centre(self, hist):
        m, seen = self.start, []
        for v, reward, effect in hist:
            sign = (reward > 0) - (reward < 0)
            m += self.lr * sign * unusualness(effect, seen) * (v - m)
            m = min(max(m, self.lo), self.hi)
            if effect:
                seen.append(effect)
        return m

    def pick(self, hist, rng=random):
        """-> (value to try, centre, ratings held)."""
        centre = self.centre(hist)
        v = min(max(centre + rng.gauss(0.0, self.explore), self.lo), self.hi)
        return v, centre, len(hist)
