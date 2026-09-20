"""Phrase probe for MiniMax H3: where does the model READ a bracketed phrase, and where
does the phrase actually CHANGE the stream?

Two maps per bracketed phrase, one number per block:

* read[b]     - the share of attention the video tokens spend on the phrase's tokens at
                block b (mean over a strided subsample of video queries and over steps).
                Free: computed from q and k inside the attention call we already hook.
* response[b] - how far block b's video-row output moves when the phrase is MASKED from
                every video query, relative to the unmasked output. Cumulative by nature:
                block b's input already differs, so the number includes everything before
                it; the increment between consecutive blocks is where the reaction grows.
                Costs one extra full forward per phrase per model call - opt-in, for a
                look, not for every run.

Anonymous by construction: phrases are numbered in prompt order and only their token
COUNT is recorded - never the text (a shortcut's trigger or replacement is the user's
prompt, and the result file is the kind of thing that gets pasted around).

Measurement only. Nothing here changes what the sampler produces: the masked forwards are
discarded, the visible forward is returned untouched.
"""
from __future__ import annotations

import json
import os
import time

import torch

MASKED_BIAS = -30.0
QUERY_SAMPLES = 32      # video queries per block for the reading map
ROW_SAMPLES = 128       # video rows per block kept from the visible pass for the response map

_ENV_SWITCH = "FUNPACK_PHRASE_PROBE"


def _dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "refinements", "phrase_probe")


def collection_enabled():
    raw = os.environ.get(_ENV_SWITCH, "").strip().lower()
    if raw:
        return raw in ("1", "true", "yes", "on")
    try:
        with open(os.path.join(_dir(), "enabled"), "r", encoding="utf-8") as fh:
            return fh.read().strip() == "1"
    except (OSError, ValueError):
        return False


def set_collection_enabled(on):
    on = bool(on)
    os.environ[_ENV_SWITCH] = "1" if on else "0"
    try:
        os.makedirs(_dir(), exist_ok=True)
        with open(os.path.join(_dir(), "enabled"), "w", encoding="utf-8") as fh:
            fh.write("1" if on else "0")
    except OSError:
        pass
    return on


class ProbeState:
    """Shared between the model wrapper, the attention override and the block hooks."""

    def __init__(self, phrases, cond_len, n_video):
        # phrases: [(lo, hi)] absolute key positions in the packed sequence, prompt order
        self.phrases = [(int(lo), int(hi)) for lo, hi in phrases]
        self.cond_len = int(cond_len)
        self.n_video = int(n_video)
        self.mode = None            # None = not inside a probed call, -1 = visible, i = masked phrase i
        self.block = 0              # attention calls seen in the current forward
        self.read = {}              # block -> phrase -> [0-dim tensors]
        self.response = {}          # block -> phrase -> [0-dim tensors]
        self.visible_rows = {}      # block -> rows kept from the visible pass (this call)
        self.calls = 0

    def begin(self, mode):
        self.mode = mode
        self.block = 0

    def next_block(self):
        b = self.block
        self.block += 1
        return b


def make_attention_override(state, inner):
    """Outermost override: masks the phrase under test, and reads attention mass on the
    visible pass. Delegates everything else to `inner` (or the backend)."""

    def _run(func, q, k, v, heads, mask, **kw):
        if inner is not None:
            return inner(func, q, k, v, heads, mask=mask, **kw)
        return func(q, k, v, heads, mask=mask, **kw)

    def override(func, q, k, v, heads, mask=None, **kw):
        try:
            packed = (kw.get("skip_reshape") and torch.is_tensor(k) and k.ndim == 4
                      and q.shape[2] == k.shape[2] and int(k.shape[2]) > state.cond_len)
            if not packed or state.mode is None:
                return _run(func, q, k, v, heads, mask, **kw)
            block = state.next_block()
            seq_len = int(k.shape[2])
            if state.mode >= 0:
                lo, hi = state.phrases[state.mode]
                bias = torch.zeros(1, 1, 1, seq_len, device=k.device, dtype=k.dtype)
                bias[..., lo:hi] = MASKED_BIAS
                return _run(func, q, k, v, heads, bias if mask is None else mask + bias, **kw)
            # visible pass: reading map
            try:
                v0 = seq_len - state.n_video
                if v0 >= state.cond_len and state.n_video > 0:
                    idx = torch.linspace(v0, seq_len - 1, steps=min(QUERY_SAMPLES, state.n_video),
                                         device=k.device).long()
                    qs = q[:, :, idx].float()                       # [1,H,n,D]
                    scale = float(q.shape[-1]) ** -0.5
                    mass = None
                    for h in range(int(q.shape[1])):                # per head: bounded memory
                        scores = (qs[:, h] @ k[:, h].float().transpose(-1, -2)) * scale  # [1,n,S]
                        if mask is not None:
                            scores = scores + (mask.float()[:, 0] if mask.ndim == 4 else mask.float())
                        probs = scores.softmax(dim=-1)
                        share = torch.stack([probs[..., lo:hi].sum(dim=-1).mean()
                                             for lo, hi in state.phrases])
                        mass = share if mass is None else mass + share
                    mass = mass / float(q.shape[1])
                    slot = state.read.setdefault(block, {})
                    for i in range(len(state.phrases)):
                        slot.setdefault(i, []).append(mass[i].detach())
            except Exception:  # noqa: BLE001 - a probe never costs the step
                pass
            return _run(func, q, k, v, heads, mask, **kw)
        except Exception:  # noqa: BLE001
            return _run(func, q, k, v, heads, mask, **kw)

    return override


def make_block_hook(state, block, inner, video_mask_fn):
    """Records each block's video-row output on the visible pass and the relative change
    on every masked pass."""
    mask_cache = {}

    def hook(args, extra):
        out = extra["original_block"](args)["img"] if inner is None else inner(args, extra)["img"]
        try:
            if state.mode is None or not torch.is_tensor(out):
                return {"img": out}
            seq_len = int(out.shape[0])
            sel = mask_cache.get(seq_len, "MISS")
            if sel == "MISS":
                m = video_mask_fn(args.get("mod_segments"), seq_len, out.device)
                sel = None
                if m is not None:
                    idx = m.nonzero(as_tuple=True)[0]
                    if idx.numel():
                        sel = idx[::max(1, int(idx.numel()) // ROW_SAMPLES)]
                mask_cache[seq_len] = sel
            if sel is None:
                return {"img": out}
            rows = out[sel].detach()
            if state.mode < 0:
                state.visible_rows[block] = rows.to(torch.bfloat16)
            else:
                ref = state.visible_rows.get(block)
                if ref is not None and ref.shape == rows.shape:
                    ref = ref.float()
                    rel = (rows.float() - ref).norm() / ref.norm().clamp(min=1e-8)
                    state.response.setdefault(block, {}).setdefault(state.mode, []).append(rel)
        except Exception:  # noqa: BLE001
            pass
        return {"img": out}

    return hook


def make_model_wrapper(state, old_wrapper):
    """model_function_wrapper: one visible forward (returned), then one masked forward per
    phrase (discarded)."""

    def _call(apply_fn, a):
        if old_wrapper is not None:
            return old_wrapper(apply_fn, a)
        return apply_fn(a["input"], a["timestep"], **a.get("c", {}))

    def wrapper(apply_fn, args):
        state.begin(-1)
        try:
            out = _call(apply_fn, args)
        finally:
            state.mode = None
        state.calls += 1
        for i in range(len(state.phrases)):
            state.begin(i)
            try:
                _call(apply_fn, args)
            except Exception:  # noqa: BLE001 - the probe's own pass failing must not cost the step
                pass
            finally:
                state.mode = None
        state.visible_rows = {}
        return out

    return wrapper


def result(state, meta=None):
    """Per-phrase maps as plain lists, means over steps. Anonymous: index + token count."""
    def _mean(d, b, i):
        vals = d.get(b, {}).get(i)
        return float(torch.stack(vals).float().mean().item()) if vals else None

    blocks = sorted(set(state.read) | set(state.response))
    phrases = []
    for i, (lo, hi) in enumerate(state.phrases):
        phrases.append({
            "phrase": i + 1,
            "tokens": hi - lo,
            "read": [_mean(state.read, b, i) for b in blocks],
            "response": [_mean(state.response, b, i) for b in blocks],
        })
    return {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "blocks": blocks,
            "model_calls": state.calls, "phrases": phrases, **(meta or {})}


def save(res):
    try:
        os.makedirs(_dir(), exist_ok=True)
        path = os.path.join(_dir(), "latest.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(res, fh)
        os.replace(tmp, path)
        return path
    except OSError:
        return None


def load_latest():
    try:
        with open(os.path.join(_dir(), "latest.json"), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def peaks(res, top=5):
    """Compact readout: for each phrase, the blocks that read it most and the blocks where
    the response grows most (increment of the cumulative response)."""
    out = []
    for p in (res or {}).get("phrases", []):
        blocks = res.get("blocks", [])
        read = [(b, v) for b, v in zip(blocks, p["read"]) if v is not None]
        resp = [(b, v) for b, v in zip(blocks, p["response"]) if v is not None]
        inc = [(b1, v1 - v0) for (b0, v0), (b1, v1) in zip(resp, resp[1:])]
        out.append({
            "phrase": p["phrase"], "tokens": p["tokens"],
            "read_top": sorted(read, key=lambda t: -t[1])[:top],
            "response_growth_top": sorted(inc, key=lambda t: -t[1])[:top],
            "response_final": resp[-1][1] if resp else None,
        })
    return out
