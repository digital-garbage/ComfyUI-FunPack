"""Shot memory: the starting noise remembers the shots you liked.

With the prompt fixed, what a like or dislike reacts to is mostly the SHOT -- where
things sit, how the frame is split. The model decides that from the coarse part of
the starting noise, so a rating here lands on the thing that caused it, not on an
averaged direction in a space thousands of dimensions wide.

Each run's coarse noise (one value per 4x4-latent cell per channel, averaged over
the whole clip) is recorded; the rating that scores the run keeps it or not. The
next run then either:

* reuses a liked shot -- its coarse noise is blended into the fresh noise by
  `amount`, and the fine noise is always fresh, so the details are new; or
* starts fresh.

Both choices learn from ratings (Thompson draws, the path planner's pattern):
whether reusing beats fresh for this key, which liked shot to reuse, and how
strongly (`amount`, one number, nudged toward what was liked). A liked child is a
shot in its own right, so good shots breed.

The noise stays exactly unit Gaussian: a block mean of Gaussian noise is
independent of what is left once it is removed, so swapping one block mean for
another of the same variance changes nothing the model can tell apart from
ordinary noise. Zero model calls. Audio noise is never touched.
"""

import math
import os
import random

import torch
import torch.nn.functional as F

# Latent pixels per coarse cell: 64 image px on H3's 16x VAE.
# ponytail: fixed; learn it per key if one cell size stops fitting every resolution.
CELL = 4
MAX_ROWS = 64
START_AMOUNT = 0.7
EXPLORE = 0.15
LR = 0.5
AMOUNT_MIN, AMOUNT_MAX = 0.2, 0.95
MODES = ("off", "learned", "manual")


def _path(refinement_key, mode):
    try:
        from .conditioning import refinement_state_path
    except ImportError:
        from conditioning import refinement_state_path
    return refinement_state_path(refinement_key, mode, prefix="refine_v2", extension="pt")


def _load(path, default):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        return default
    except Exception as e:  # noqa: BLE001
        print(f"[FunPack Shot memory] could not read {os.path.basename(path)}: {e}")
        return default


def _save(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_rows(refinement_key):
    """Rated shots, oldest first: {id, parent, amount, reward, coarse [C,h,w], cond [D]|None}."""
    if not refinement_key:
        return []
    return list(_load(_path(refinement_key, "shot_memory"), {"rows": []}).get("rows") or [])


def _sign(x):
    return (x > 0) - (x < 0)


def coarse_of(video):
    """[1, C, T, H, W] noise -> its coarse part [C, h, w], unit variance, or None."""
    if video.dim() != 5 or video.shape[0] != 1:
        return None
    _b, c, t, H, W = video.shape
    h, w = H // CELL, W // CELL
    if h < 1 or w < 1:
        return None
    blocks = video[0, :, :, :h * CELL, :w * CELL].float().reshape(c, t, h, CELL, w, CELL)
    return blocks.mean(dim=(1, 3, 5)) * math.sqrt(t * CELL * CELL)


def with_coarse(video, coarse):
    """`video` with its coarse part replaced by `coarse` [C, h, w]; the fine part kept."""
    _b, c, t, _H, _W = video.shape
    h, w = coarse.shape[-2:]
    core = video[0, :, :, :h * CELL, :w * CELL].float().reshape(c, t, h, CELL, w, CELL)
    delta = coarse.to(core) / math.sqrt(t * CELL * CELL) - core.mean(dim=(1, 3, 5))
    core = core + delta[:, None, :, None, :, None]
    out = video.clone()
    out[0, :, :, :h * CELL, :w * CELL] = core.reshape(c, t, h * CELL, w * CELL).to(video.dtype)
    return out


def _fit(coarse, h, w):
    """A stored coarse map on this run's grid. Another resolution is resampled and
    re-standardised, which keeps the layout but is no longer exactly Gaussian."""
    coarse = coarse.float()
    if tuple(coarse.shape[-2:]) == (h, w):
        return coarse
    x = F.interpolate(coarse[None], size=(h, w), mode="bilinear", align_corners=False)[0]
    return (x - x.mean()) / x.std().clamp(min=1e-6)


def pooled(cond):
    if not isinstance(cond, torch.Tensor):
        return None
    p = cond.detach().float()
    while p.dim() > 1:
        p = p.mean(dim=0)
    return p.cpu()


def learned_amount(rows):
    """How strongly to reuse a liked shot, replayed from the rated reuses in order:
    each liked one pulls the value toward the amount it used, each disliked one
    pushes it away. Recent ratings weigh more because they move it last."""
    m = START_AMOUNT
    for r in rows:
        if int(r.get("parent", -1)) < 0:
            continue
        m += LR * _sign(float(r.get("reward", 0.0))) * (float(r.get("amount", m)) - m)
        m = min(max(m, AMOUNT_MIN), AMOUNT_MAX)
    return m


def choose_parent(rows, cond, channels, rng=random):
    """-> (liked row to reuse | None, why). Two Thompson draws: reuse-vs-fresh from
    how reuses and fresh shots have been rated, then which liked shot, weighted by
    how alike its prompt was."""
    by_id = {int(r["id"]): r for r in rows}
    reuse_ab, fresh_ab = [1.0, 1.0], [1.0, 1.0]
    children = {}
    for r in rows:
        s = _sign(float(r.get("reward", 0.0)))
        if not s:
            continue
        parent = int(r.get("parent", -1))
        ab = fresh_ab if parent < 0 else reuse_ab
        ab[0 if s > 0 else 1] += 1.0
        if parent in by_id:
            children.setdefault(parent, [1.0, 1.0])[0 if s > 0 else 1] += 1.0
    liked = [r for r in rows if float(r.get("reward", 0.0)) > 0
             and int(r["coarse"].shape[0]) == int(channels)]
    if not liked:
        return None, "fresh: no liked shot yet"
    reuse, fresh = rng.betavariate(*reuse_ab), rng.betavariate(*fresh_ab)
    if fresh >= reuse:
        return None, f"fresh (reuse {reuse:.2f} < fresh {fresh:.2f})"
    best, best_theta = None, -1.0
    for r in liked:
        a, b = children.get(int(r["id"]), [1.0, 1.0])
        theta = rng.betavariate(a + 1.0, b)          # +1: its own like
        stored = r.get("cond")
        if cond is not None and isinstance(stored, torch.Tensor) and stored.numel() == cond.numel():
            # ponytail: raw pooled-prompt cosine; sits high for most prompts under one key.
            theta *= float(F.cosine_similarity(cond, stored.float(), dim=0).clamp(0.0, 1.0))
        if theta > best_theta:
            best, best_theta = r, theta
    return best, f"reusing a liked shot of {len(liked)} (reuse {reuse:.2f} > fresh {fresh:.2f})"


class ShotMemory:
    """One run's decisions: which shot each scene reuses and how strongly, and the
    shots actually used, waiting for the rating."""

    def __init__(self, refinement_key, mode="learned", manual_amount=START_AMOUNT, rng=random):
        self.key = refinement_key
        self.mode = mode
        self.manual = min(max(float(manual_amount), 0.0), AMOUNT_MAX)
        self.rng = rng
        self.rows = load_rows(refinement_key)
        self.plans = {}
        self.used = []
        self.notes = []
        self.skipped = 0

    def _plan(self, cond, channels):
        sig = None if cond is None else (cond.numel(), round(float(cond.sum()), 3))
        if (sig, channels) not in self.plans:
            parent, why = choose_parent(self.rows, cond, channels, self.rng)
            amount = 0.0
            if parent is not None:
                if self.mode == "manual":
                    amount = self.manual
                else:
                    amount = learned_amount(self.rows) + self.rng.gauss(0.0, EXPLORE)
                    amount = min(max(amount, AMOUNT_MIN), AMOUNT_MAX)
                why += f", amount {amount:.2f}"
            self.plans[(sig, channels)] = (parent, amount)
            self.notes.append(why)
        return self.plans[(sig, channels)]

    def shape(self, noise, samples, cond, record=False):
        """The starting noise for a run from `samples`. Unchanged when the latent is
        not empty (a second pass, a latent anchor, a carried overlap): there the noise
        is not what decides the shot, and the run says so (`skipped`).

        Only the run's FIRST shot is recorded: one rating is one row, so a
        multi-scene run cannot count its single Like several times over. Later
        scenes still reuse liked shots; they just don't become shots themselves."""
        nested = getattr(noise, "is_nested", False)
        video = noise.unbind()[0] if nested else noise
        base = samples.unbind()[0] if getattr(samples, "is_nested", False) else samples
        if not isinstance(video, torch.Tensor) or video.dim() != 5:
            return noise
        if bool(torch.count_nonzero(base)):
            self.skipped += int(record)
            return noise
        own = coarse_of(video)
        if own is None:
            return noise
        cond = pooled(cond)
        parent, amount = self._plan(cond, int(own.shape[0]))
        coarse = own
        if parent is not None:
            coarse = amount * _fit(parent["coarse"], *own.shape[-2:]) + math.sqrt(1.0 - amount ** 2) * own
            video = with_coarse(video, coarse)
        if record and not self.used:
            self.used.append({
                "id": self.rng.getrandbits(62), "parent": int(parent["id"]) if parent else -1,
                "amount": float(amount), "coarse": coarse.half().cpu(),
                "cond": cond.half() if cond is not None else None,
            })
        if not nested:
            return video
        from comfy.nested_tensor import NestedTensor
        return NestedTensor([video, *noise.unbind()[1:]])

    def save_pending(self):
        """This run's shots, for the rating that scores it. One run pending per key."""
        if not self.key:
            return
        path = _path(self.key, "shot_memory_pending")
        if self.used:
            _save({"rows": self.used}, path)
        elif os.path.exists(path):
            os.remove(path)


def clear_pending(refinement_key):
    """A keyed run with shot memory off: its rating must not score an older run's shots."""
    if refinement_key:
        try:
            os.remove(_path(refinement_key, "shot_memory_pending"))
        except OSError:
            pass


def commit(refinement_key, reward):
    """Pair the pending shots with the rating that scores their run. -> rows kept, or
    None when nothing was pending. The pending file is removed either way."""
    if not refinement_key:
        return None
    path = _path(refinement_key, "shot_memory_pending")
    pending = _load(path, None)
    if pending is None:
        return None
    try:
        if not _sign(float(reward)):
            return None
        rows = load_rows(refinement_key)
        for r in pending.get("rows") or []:
            rows.append({**r, "reward": float(_sign(float(reward)))})
        rows = rows[-MAX_ROWS:]
        _save({"rows": rows}, _path(refinement_key, "shot_memory"))
        return len(rows)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
