"""The probe's arithmetic and its stored profile: no model, no routes."""

import torch

from ..._core import config, dit_hooks, registry

KIND = "block_influence"
MIN_PER_GROUP = 2           # liked AND disliked before a difference is one
SWITCH = config.ROOT / "block_influence.enabled"   # on disk: a fresh rental resumes recording


# (key, why) the last recording run stored nothing, for the panel; cleared by any later run.
problem = None


def enabled() -> bool:
    try:
        return SWITCH.read_text().strip() == "1"
    except OSError:
        return False


def set_enabled(on: bool) -> bool:
    SWITCH.parent.mkdir(parents=True, exist_ok=True)
    SWITCH.write_text("1" if on else "0")
    return bool(on)


class Tally:
    """One run's per-block sums, kept on the device until the run ends.

    No `.item()` per block per step (50 blocks x steps host syncs): sums are 0-dim
    tensors and are read once, after sampling. Rows are strided down to
    ~`max_rows` before the norms so the cost does not grow with H3's sequence length.
    """

    def __init__(self, n, max_rows=512):
        self.n, self.max_rows = n, max_rows
        self._sum = {k: [None] * n for k in ("raw", "ratio", "novelty")}
        self._count = {k: [0] * n for k in ("raw", "ratio", "novelty")}
        self._prev = (None, -1)         # previous block's push, for the novelty cosine
        self._spans = {}

    def _span(self, args, src):
        seq = int(src.shape[0])
        if seq not in self._spans:
            self._spans[seq] = dit_hooks.row_span(
                dit_hooks.target_rows(args.get("mod_segments"), seq, src.device, "video"))
        return self._spans[seq]

    def _add(self, kind, block, value):
        s = self._sum[kind][block]
        self._sum[kind][block] = value if s is None else s + value
        self._count[kind][block] += 1

    def measure(self, block, args, original):
        """Run the block and record how far it moved the picture rows."""
        src = args.get("img")
        sel = before = None
        if torch.is_tensor(src) and src.dim() == 2:
            span = self._span(args, src)
            if span is not None:
                sel = slice(span[0], span[1], max(1, (span[1] - span[0]) // self.max_rows))
                # A COPY, read before the block runs: a block that writes its residual in
                # place leaves `src` already holding the result, and every delta is zero.
                before = src[sel].detach().to(torch.float32, copy=True)
        out = original(args)
        if before is not None and torch.is_tensor(out.get("img")) \
                and out["img"].shape[0] == src.shape[0]:
            delta = out["img"][sel].detach().float() - before
            push = delta.norm()
            self._add("raw", block, push)
            self._add("ratio", block, push / before.norm().clamp(min=1e-8))
            last, last_block = self._prev
            # A block index that does not exceed the last one means a new step began:
            # pairing across it would compare the end of one step to the start of the next.
            if last is not None and block > last_block and last.shape == delta.shape:
                self._add("novelty", block, torch.nn.functional.cosine_similarity(
                    delta.flatten(), last.flatten(), dim=0))
            self._prev = (delta, block)
        return out

    def blocks_seen(self) -> int:
        return sum(1 for c in self._count["raw"] if c)

    def rows(self):
        """{name: [n_blocks] tensor} (NaN where a block saw nothing), or None when
        nothing was measured -- which must be said, not stored as a flat profile."""
        if not self.blocks_seen():
            return None
        rows = {}
        for name in ("ratio", "raw", "novelty"):
            have = [i for i, c in enumerate(self._count[name]) if c]
            vec = torch.full((self.n,), float("nan"))
            if have:
                means = torch.stack([self._sum[name][i].float() / self._count[name][i] for i in have])
                vec[have] = means.cpu()                      # the one host sync
            rows[name] = vec
        return rows


def profile(rows) -> dict:
    """Depth profile from a key's rated rows.

    `overall` is where the model does its work regardless of ratings; `share` is
    each block's RAW push as a fraction of all blocks' (rank by this: the ratio
    shrinks with depth by construction, since the stream grows); `difference` is
    liked minus disliked; `flatness` is the spread of `overall` as a fraction of
    its mean -- near 0 means every block moves the stream equally and there is
    nothing to aim at. `novelty` ~1: a block amplifies its predecessor, ~0 it
    adds something new, negative it partly undoes it.
    """
    rows = [r for r in rows if {"ratio", "raw", "novelty"} <= set(r["rows"])]
    empty = {"used": 0, "overall": {}, "difference": None, "share": {}, "novelty": {}, "mean_novelty": None,
             "flatness": None, "n_liked": 0, "n_disliked": 0}
    if not rows:
        return empty
    n = len(rows[-1]["rows"]["ratio"])
    rows = [r for r in rows if len(r["rows"]["ratio"]) == n]     # a different model's rows
    reward = torch.tensor([float(r["reward"]) for r in rows])
    stack = {k: torch.stack([r["rows"][k].float() for r in rows]) for k in ("ratio", "raw", "novelty")}

    def mean(name, mask=None):
        m = stack[name] if mask is None else stack[name][mask]
        return torch.nanmean(m, dim=0) if len(m) else torch.full((n,), float("nan"))

    def table(vec):
        return {str(i): float(v) for i, v in enumerate(vec) if torch.isfinite(v)}

    overall, raw, novelty = mean("ratio"), mean("raw"), mean("novelty")
    seen = torch.isfinite(raw)
    share = table(raw / raw[seen].sum()) if seen.any() and raw[seen].sum() > 1e-12 else {}
    liked, disliked = reward > 0, reward < 0
    difference = None
    if int(liked.sum()) >= MIN_PER_GROUP and int(disliked.sum()) >= MIN_PER_GROUP:
        difference = table(mean("ratio", liked) - mean("ratio", disliked))
    vals = overall[torch.isfinite(overall)]
    flat = float(vals.std(unbiased=False) / vals.mean()) if len(vals) and abs(float(vals.mean())) > 1e-12 else None
    nov = novelty[torch.isfinite(novelty)]
    return {"used": len(rows), "overall": table(overall), "difference": difference, "share": share,
            "novelty": table(novelty), "mean_novelty": float(nov.mean()) if len(nov) else None,
            "flatness": flat, "n_liked": int(liked.sum()), "n_disliked": int(disliked.sum())}


def _kind(key):
    kind = registry.current().ask("taste_kind", key, KIND)
    if kind is None:
        raise ValueError("the Taste key module is off, so there is nowhere to read profiles from")
    return kind


def resolve(key):
    """The key to show. The panel asks for a placeholder ("default") when it does not know
    which Taste key the runs write to, so a key with nothing recorded falls back to the key
    the latest capture went to -- and the answer says which one it is."""
    if _kind(key).path().exists():
        return key
    latest = registry.current().ask("taste_latest_key")
    return latest if latest and _kind(latest).path().exists() else key


def state(key, fallback=True) -> dict:
    """What Settings > Refinement & Taste shows for one key. `fallback` False answers for
    exactly that key (after a Clear, which must not show another key's data)."""
    key = resolve(key) if fallback else key
    rows = _kind(key).rows()
    prof = profile(rows)
    shown = problem[1] if problem and problem[0] == key and enabled() else None
    return {"key": key, "enabled": enabled(), "runs": len(rows), "problem": shown,
            "skipped": len(rows) - prof["used"], "min_per_group": MIN_PER_GROUP,
            **prof}


def clear(key):
    _kind(key).clear()


def path_of(key):
    return _kind(key).path()
