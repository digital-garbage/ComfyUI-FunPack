"""Where does the model READ a bracketed phrase, and where does the phrase CHANGE the stream?

Two maps per phrase, one number per block:

* read[b]     the share of attention the picture tokens spend on the phrase's tokens at block b
              (a strided sample of picture queries, averaged over steps). Free: computed from the
              q and k of the attention call that is happening anyway.
* response[b] how far block b's picture-row output moves when the phrase is MASKED from every
              picture query, relative to the unmasked output. Cumulative by nature (block b's
              input already differs), so the increment between blocks is where the reaction grows.
              Costs one extra full forward per phrase per model call: opt-in, for a look.

Anonymous by construction: phrases are numbered in prompt order and only their token COUNT is
kept -- never the text (a shortcut's trigger or replacement is the user's prompt, and the result
file gets pasted around).

Measurement only. The masked forwards are discarded; the unmasked forward is returned untouched.
"""

import json
import time

import torch

from ..._core import config, dit_hooks

MASKED_BIAS = -30.0
QUERY_SAMPLES = 32      # picture queries per block for the reading map
ROW_SAMPLES = 128       # picture rows per block kept from the visible pass for the response map
SWITCH = config.ROOT / "phrase_probe.enabled"
LATEST = config.ROOT / "phrase_probe.latest.json"

problem = None          # why nothing was measured, for the panel


def enabled() -> bool:
    try:
        return SWITCH.read_text().strip() == "1"
    except OSError:
        return False


def set_enabled(on: bool) -> bool:
    SWITCH.parent.mkdir(parents=True, exist_ok=True)
    SWITCH.write_text("1" if on else "0")
    return bool(on)


def phrases_of(published, cond_len=None):
    """Absolute key positions [(lo, hi)] of the phrases the markup node published, in prompt
    order, deduplicated; or []. Anything that does not fit inside the encoded prompt is dropped."""
    if not isinstance(published, dict):
        return []
    try:
        base, cond = int(published.get("base") or 0), int(published.get("cond_len") or cond_len or 0)
        return sorted({(base + int(a), base + int(b)) for a, b in published.get("spans") or []
                       if int(b) > int(a) and base + int(b) <= cond})
    except (TypeError, ValueError):
        return []


class State:
    """One run's tallies. Shared by the attention override, the block hooks and the model wrapper."""

    def __init__(self, phrases, cond_len):
        self.phrases, self.cond_len = list(phrases), int(cond_len)
        self.mode = None            # None: not inside a probed call; -1: the visible pass; i: phrase i hidden
        self.block = 0
        self.span = None            # (lo, hi) of the picture rows in the current sequence
        self.read, self.response = {}, {}          # block -> phrase -> [0-dim tensors]
        self.visible = {}           # block -> picture rows kept from the visible pass of this call
        self.calls = 0
        self._spans = {}

    def picture_span(self, args, src):
        seq = int(src.shape[0])
        if seq not in self._spans:
            self._spans[seq] = dit_hooks.row_span(
                dit_hooks.target_rows(args.get("mod_segments"), seq, src.device, "video"))
        return self._spans[seq]


def attention_override(state):
    """Hides the phrase under test (a masked pass) or reads attention mass (the visible pass)."""

    def override(func, q, k, v, heads, *args, mask=None, **kw):
        packed = (state.mode is not None and torch.is_tensor(k) and k.ndim == 4
                  and q.shape[2] == k.shape[2] and int(k.shape[2]) > state.cond_len)
        if not packed:
            return func(q, k, v, heads, *args, mask=mask, **kw)
        seq = int(k.shape[2])
        if state.mode >= 0:
            lo, hi = state.phrases[state.mode]
            bias = torch.zeros(1, 1, 1, seq, device=k.device, dtype=k.dtype)
            bias[..., lo:hi] = MASKED_BIAS
            return func(q, k, v, heads, *args, mask=bias if mask is None else mask + bias, **kw)
        try:
            if state.span is not None:
                lo, hi = state.span
                idx = torch.linspace(lo, hi - 1, steps=min(QUERY_SAMPLES, hi - lo), device=k.device).long()
                qs = q[:, :, idx].float()                              # [1,H,n,D]
                scale = float(q.shape[-1]) ** -0.5
                mass = None
                for h in range(int(q.shape[1])):                       # per head: bounded memory
                    scores = (qs[:, h] @ k[:, h].float().transpose(-1, -2)) * scale
                    if mask is not None:
                        scores = scores + (mask.float()[:, 0] if mask.ndim == 4 else mask.float())
                    probs = scores.softmax(dim=-1)
                    share = torch.stack([probs[..., a:b].sum(dim=-1).mean() for a, b in state.phrases])
                    mass = share if mass is None else mass + share
                mass = mass / float(q.shape[1])
                slot = state.read.setdefault(state.block, {})
                for i in range(len(state.phrases)):
                    slot.setdefault(i, []).append(mass[i].detach())
        except Exception:                                              # noqa: BLE001 -- a probe never costs the step
            pass
        return func(q, k, v, heads, *args, mask=mask, **kw)

    return override


def block_hook(state, block):
    """Runs the block, then keeps the picture rows (visible pass) or compares them (masked pass)."""

    def hook(args, extra):
        if state.mode is None:
            return extra["original_block"](args)
        state.block = block
        src = args.get("img")
        if torch.is_tensor(src) and src.dim() == 2:
            state.span = state.picture_span(args, src)
        out = extra["original_block"](args)
        try:
            img = out.get("img")
            if state.span is not None and torch.is_tensor(img):
                lo, hi = state.span
                rows = img[slice(lo, hi, max(1, (hi - lo) // ROW_SAMPLES))].detach()
                if state.mode < 0:
                    state.visible[block] = rows.to(torch.bfloat16)
                else:
                    ref = state.visible.get(block)
                    if ref is not None and ref.shape == rows.shape:
                        ref = ref.float()
                        rel = (rows.float() - ref).norm() / ref.norm().clamp(min=1e-8)
                        state.response.setdefault(block, {}).setdefault(state.mode, []).append(rel)
        except Exception:                                              # noqa: BLE001
            pass
        return out

    return hook


def result(state) -> dict:
    """Per-phrase maps as plain lists, means over steps. Anonymous: index + token count."""
    def mean(d, b, i):
        vals = d.get(b, {}).get(i)
        return float(torch.stack(vals).float().mean().item()) if vals else None

    blocks = sorted(set(state.read) | set(state.response))
    return {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "blocks": blocks, "model_calls": state.calls,
            "phrases": [{"phrase": i + 1, "tokens": hi - lo,
                         "read": [mean(state.read, b, i) for b in blocks],
                         "response": [mean(state.response, b, i) for b in blocks]}
                        for i, (lo, hi) in enumerate(state.phrases)]}


def save(res) -> None:
    LATEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = LATEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.replace(LATEST)


def latest():
    try:
        return json.loads(LATEST.read_text())
    except (OSError, ValueError):
        return None


def clear() -> None:
    try:
        LATEST.unlink()
    except OSError:
        pass


def peaks(res, top=5):
    """Per phrase: the blocks that read it most, and where its response grows most."""
    out = []
    for p in (res or {}).get("phrases", []):
        blocks = res.get("blocks", [])
        read = [(b, v) for b, v in zip(blocks, p["read"]) if v is not None]
        resp = [(b, v) for b, v in zip(blocks, p["response"]) if v is not None]
        inc = [(b1, v1 - v0) for (_b0, v0), (b1, v1) in zip(resp, resp[1:])]
        out.append({"phrase": p["phrase"], "tokens": p["tokens"],
                    "read_top": sorted(read, key=lambda t: -t[1])[:top],
                    "response_growth_top": sorted(inc, key=lambda t: -t[1])[:top],
                    "response_final": resp[-1][1] if resp else None})
    return out
