"""Prompt markup: the text part. Pure -- strings in, spans out, tensors built.

Three markups, stripped in ONE left-to-right pass so every span indexes the
same clean text (stripping them one after another left the first list's
offsets pointing into a string that no longer existed):

* `(phrase:1.5)` / `[phrase:1.5]` -- weight. A logit bias of log(w) on that
  phrase's key positions: softmax(logits + log w) multiplies that key's share
  of the attention by w, which is what a weight is supposed to mean.
* `[phrase@2-3.5]` / `[phrase:1.5@2-3.5]` -- a timed phrase: only the video
  rows inside the window see it (weighted); outside, it is masked from video.
  Text and audio rows keep the untimed bias. Confirmed working on H3 in v4.
* `[phraseA|phraseB]` / `[A|B:0.8]` -- blend: (1-s)*A + s*B conditioning,
  row by row. Confirmed running end-to-end in v4; visual quality unjudged.

Weighting only exists because the text is IN the attention sequence (H3's
single packed stream). Scaling embeddings would not work: Qwen's rows are
contextual hidden states, with no per-word magnitude to turn up.
"""

import math
import re

_NUM = r"-?(?:\d+\.\d*|\.\d+|\d+)"
_WEIGHTED = re.compile(r"(?<!\\)\(([^():]*?):\s*(" + _NUM + r")\s*\)")
_TIMED = re.compile(
    r"(?<!\\)\[([^\[\]]*?)(?::\s*(" + _NUM + r"))?\s*@\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*\]")
_BRACKET_WEIGHTED = re.compile(r"(?<!\\)\[([^\[\]@]*?):\s*(" + _NUM + r")\s*\]")
_BLENDED = re.compile(r"(?<!\\)\[([^\[\]|]+)\|([^\[\]|]+?)(?::\s*(" + _NUM + r"))?\]")
_MARKUP = re.compile("|".join(p.pattern for p in (_TIMED, _BLENDED, _WEIGHTED, _BRACKET_WEIGHTED)))

BLEND_STRENGTH_DEFAULT = 0.5
# Below this a weight means "remove": log(0) is -inf, which is NaN the moment a
# query attends to nothing else.
MIN_WEIGHT = 1e-3
MASKED_BIAS = -30.0
# Past this a phrase wins every query regardless of content: the prompt reads as
# collapsing onto one word.
MAX_ABS_BIAS = 6.0
# Overlapping spans (a phrase and one of its own words) add, bounded at this
# multiple of the strongest single span -- a lone (word:2.0) still means 2.0.
OVERLAP_HEADROOM = 1.5


def parse(text):
    """-> {"clean", "weighted": [(c0, c1, w)], "timed": [(c0, c1, w, t0, t1)],
    "blended": [(c0, c1, alt, strength)]}, char spans into `clean`."""
    out = {"clean": text or "", "weighted": [], "timed": [], "blended": []}
    if not text or ("(" not in text and "[" not in text):
        return out
    pieces, pos = [], 0
    for m in _MARKUP.finditer(text):
        pieces.append(text[pos:m.start()])
        start = sum(len(p) for p in pieces)
        if m.group(1) is not None:
            phrase, t0, t1 = m.group(1), float(m.group(3)), float(m.group(4))
            if t1 > t0 and phrase.strip():
                out["timed"].append((start, start + len(phrase),
                                     float(m.group(2)) if m.group(2) else 1.0, t0, t1))
        elif m.group(5) is not None:
            phrase, alt = m.group(5), m.group(6)
            strength = float(m.group(7)) if m.group(7) is not None else BLEND_STRENGTH_DEFAULT
            if phrase.strip() and alt.strip():
                out["blended"].append((start, start + len(phrase), alt.strip(), strength))
        else:
            phrase = m.group(8) if m.group(8) is not None else m.group(10)
            weight = m.group(9) if m.group(9) is not None else m.group(11)
            out["weighted"].append((start, start + len(phrase), float(weight)))
        pieces.append(phrase)
        pos = m.end()
    pieces.append(text[pos:])
    out["clean"] = "".join(pieces)
    return out


def token_spans(offsets, char_spans):
    """Char spans -> (tok0, tok1, *rest) using a tokenizer's offset mapping."""
    out = []
    for c0, c1, *rest in char_spans:
        toks = [i for i, (a, b) in enumerate(offsets) if a < c1 and b > c0]
        if toks:
            out.append((toks[0], toks[-1] + 1, *rest))
    return out


def bias_value(weight):
    if weight <= MIN_WEIGHT:
        return MASKED_BIAS
    return max(-MAX_ABS_BIAS, min(MAX_ABS_BIAS, math.log(weight)))


def key_bias(spans, base, seq_len, device, dtype):
    """[1, 1, 1, seq_len] additive bias for (tok0, tok1, w) spans, prompt at `base`."""
    import torch
    bias, strongest = None, 0.0
    for t0, t1, weight in spans:
        b = bias_value(weight)
        if b == 0.0 or t1 <= t0:
            continue
        if bias is None:
            bias = torch.zeros(1, 1, 1, seq_len, device=device, dtype=dtype)
        bias[..., base + t0:base + t1] += b
        strongest = max(strongest, abs(b))
    if bias is not None:
        limit = min(MAX_ABS_BIAS, strongest * OVERLAP_HEADROOM)
        bias.clamp_(min=-limit, max=limit)
    return bias


def plan(weighted, timed, base, seq_len, video_start, device, dtype):
    """The bias to add, as one tensor, or [(q0, q1, bias)] query chunks when a
    phrase is timed. Softmax is per query row, so attending the video rows in
    chunks, each with its own key bias, is exact -- it only adds kernel launches.

    `timed` = [(tok0, tok1, w, r0, r1)] with r0/r1 absolute rows of the sequence.
    """
    import torch
    base_bias = key_bias(weighted, base, seq_len, device, dtype)
    if not timed:
        return base_bias
    edges = sorted({0, video_start, seq_len, *(r for *_x, r0, r1 in timed for r in (r0, r1))})
    chunks = []
    for a, b in zip(edges, edges[1:]):
        if b <= a:
            continue
        bias = (base_bias.clone() if base_bias is not None
                else torch.zeros(1, 1, 1, seq_len, device=device, dtype=dtype))
        if a >= video_start:
            for t0, t1, w, r0, r1 in timed:
                if r0 <= a and b <= r1:
                    bias[..., base + t0:base + t1] += bias_value(w)
                else:
                    bias[..., base + t0:base + t1] = MASKED_BIAS
        chunks.append((a, b, bias))
    return chunks
