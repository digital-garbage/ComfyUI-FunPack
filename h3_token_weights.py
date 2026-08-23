"""Real token weighting for MiniMax H3.

Qwen ignores `(word:1.2)`. ComfyUI's `MiniMaxH3Tokenizer.tokenize_with_weights` hardcodes
every weight to 1.0 and runs the raw HF tokenizer over the literal string, so the syntax
does not weight anything — it reaches the encoder as punctuation inside your sentence, and
`sd1_clip`'s weighting arm never runs because no weight is ever != 1.0.

Scaling the embeddings would not have worked anyway. Qwen's conditioning rows are
CONTEXTUAL hidden states — row i is "having read the prompt this far", not "the vector for
this word" — so there is no per-word magnitude to turn up.

What H3 does have is a single packed self-attention stream with the prompt's tokens IN it.
Every video token attends over those positions, so the weighting CLIP only ever
approximated can be done exactly: add `log(w)` to the attention logits at that phrase's key
positions. softmax(logits + log w) is softmax(logits) with that key's share of the
attention mass multiplied by w, which is what a weight is supposed to mean.

Costs one broadcast add per attention call. The catch is stated where it is installed: a
biased call cannot run on a block-sparse kernel, so SLA goes dense for the run.
"""
from __future__ import annotations

import math
import re
from typing import Optional

# (phrase:1.3) / (phrase:-0.4). Escaped \( is left alone so a literal bracket still works.
_WEIGHTED = re.compile(r"(?<!\\)\(([^():]*?):\s*(-?\d+(?:\.\d+)?)\s*\)")

# Below this a weight is treated as "remove entirely" rather than log(0) = -inf, which
# produces NaN the moment a query attends to nothing else.
MIN_WEIGHT = 1e-3
MASKED_BIAS = -30.0
# A guard, not a preference: bias is added to raw logits, and past this the phrase wins
# every query regardless of content, which reads as the prompt collapsing onto one word.
MAX_ABS_BIAS = 6.0


def parse(text: str):
    """Strip weight syntax. -> (clean_text, [(start_char, end_char, weight), ...]).

    Spans index into `clean_text`, so they survive the removal of the brackets. Returns the
    text unchanged and an empty list when there is nothing to parse, which is the common
    case and must cost nothing.
    """
    if not text or "(" not in text:
        return text, []
    out = []
    spans = []
    pos = 0
    for m in _WEIGHTED.finditer(text):
        out.append(text[pos:m.start()])
        phrase = m.group(1)
        start = sum(len(p) for p in out)
        out.append(phrase)
        spans.append((start, start + len(phrase), float(m.group(2))))
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), spans


def token_spans(tokenizer, clean_text: str, char_spans):
    """Map character spans to token index ranges. -> [(start_tok, end_tok, weight), ...].

    Uses the tokenizer's own offset mapping rather than re-tokenizing each phrase on its
    own: "cat" tokenized alone and "cat" inside a sentence are not always the same ids, and
    a span that is off by one token weights the wrong word.
    """
    if not char_spans:
        return []
    try:
        enc = tokenizer(clean_text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
    except Exception:  # noqa: BLE001 — a slow tokenizer has no offsets; weighting is off
        return []
    out = []
    for c_start, c_end, weight in char_spans:
        toks = [i for i, (a, b) in enumerate(offsets) if a < c_end and b > c_start]
        if toks:
            out.append((toks[0], toks[-1] + 1, weight))
    return out


def bias_value(weight: float) -> float:
    """The additive logit bias a multiplicative weight corresponds to."""
    if weight <= MIN_WEIGHT:
        return MASKED_BIAS
    return max(-MAX_ABS_BIAS, min(MAX_ABS_BIAS, math.log(weight)))


def build_bias(spans, prompt_tokens: int, cond_len: int, seq_len: int, device, dtype):
    """An additive [1, 1, 1, seq_len] attention bias, or None if nothing applies.

    The prompt is the LAST thing the H3 tokenizer appends — reference pictures, audio and
    video labels all go in front of it — so the prompt's tokens are the final
    `prompt_tokens` rows of the conditioning block, and the conditioning block leads the
    packed sequence. That places prompt token j at packed position
    `cond_len - prompt_tokens + j`, without needing to know what the references added.
    """
    import torch

    if not spans or prompt_tokens <= 0 or cond_len <= 0 or seq_len < cond_len:
        return None
    base = cond_len - prompt_tokens
    if base < 0:
        return None
    bias = None
    for start, end, weight in spans:
        b = bias_value(weight)
        if b == 0.0:
            continue
        lo, hi = base + max(0, start), base + min(prompt_tokens, end)
        if hi <= lo or hi > cond_len:
            continue
        if bias is None:
            bias = torch.zeros(1, 1, 1, seq_len, device=device, dtype=dtype)
        bias[..., lo:hi] += b
    return bias


def make_override(spans, prompt_tokens: int, cond_len: int, inner=None, on_apply=None):
    """An `optimized_attention_override` that adds the bias, then delegates.

    `inner` is the override this one displaces (SLA's, or the mask-safe one) so installing
    weighting never silently discards the backend the user chose. The bias is cached per
    (sequence length, device, dtype) — it is the same tensor on every call and every block.
    """
    cache: dict = {}

    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        def run(m):
            if inner is not None:
                return inner(func, q, k, v, heads, mask=m, attn_precision=attn_precision,
                             skip_reshape=skip_reshape,
                             skip_output_reshape=skip_output_reshape, **kwargs)
            return func(q, k, v, heads, mask=m, attn_precision=attn_precision,
                        skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape, **kwargs)

        try:
            # Only the packed self-attention: k and q are the same sequence there. The
            # 2-block token refiner and any cross-shaped call are left alone.
            if not skip_reshape or k.ndim != 4 or q.shape[2] != k.shape[2]:
                return run(mask)
            seq_len = int(k.shape[2])
            if seq_len < cond_len:
                return run(mask)
            key = (seq_len, k.device, k.dtype)
            if key not in cache:
                cache[key] = build_bias(spans, prompt_tokens, cond_len, seq_len,
                                        k.device, k.dtype)
                if on_apply is not None and cache[key] is not None:
                    on_apply(seq_len)
            bias = cache[key]
            if bias is None:
                return run(mask)
            return run(bias if mask is None else mask + bias)
        except Exception:  # noqa: BLE001 — weighting must never cost the step
            return run(mask)

    return override
