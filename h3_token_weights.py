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
    return token_spans_from_offsets(offsets, char_spans)


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


# --- rating-derived weights ------------------------------------------------
#
# Studio already knows which words to weight. `_v2_concept_units_for_run` decomposes every
# run into phrases, single words and n-grams; `_v2_update_phrase_memory` trains each one's
# category scores against the rating's missing / satisfied / regressed / wrong axes. What
# was missing was a CONSUMER: the only thing reading that signal was the attn2 K/V patch,
# which H3 skips outright (one packed self-attention stream, no cross-attention). So the
# learning ran every generation and landed nowhere.
#
# This turns an entry into a multiplicative weight, which `bias_value` turns into the logit
# bias. Nothing here decides what a phrase MEANS — that is already decided upstream.

# Boost only. A phrase carrying an axis the rating said was MISSING gets more attention;
# nothing is taken away unless damping is asked for explicitly. Damping can suppress a
# phrase the user typed, which is a knob going inert on its own — an opt-in, not a default.
MIN_LEARNED_WEIGHT = 0.25
MAX_LEARNED_WEIGHT = 3.0

_KIND_SCALE = {"prompt_phrase": 1.0, "phrase": 1.0, "auto_phrase": 0.72,
               "repair_candidate": 0.64, "ngram": 0.62, "token": 0.24}


def _kind_scale(kind):
    return _KIND_SCALE.get(str(kind or "phrase"), 0.5)


def weights_from_memory(phrases, phrase_memory=None, missing_axes=(), wrong_axes=(),
                        wrong_appearance=False, strength=0.5, damp=False,
                        kind_scale=None):
    """[(text, weight), ...] for the phrases the rating has something to say about.

    `phrases` are this run's classified units (each with `text`, `kind` and
    `effective_category_scores`); `phrase_memory` is the learned store, which overrides a
    unit's own scores when it has seen the phrase before. Phrases the rating is neutral
    about are omitted rather than returned at 1.0, so nothing is biased by accident.
    """
    scale = kind_scale or _kind_scale
    missing = tuple(missing_axes or ())
    wrong = tuple(wrong_axes or ())
    memory = phrase_memory if isinstance(phrase_memory, dict) else {}
    out = []
    for unit in phrases or []:
        if not isinstance(unit, dict):
            continue
        text = str(unit.get("text", "")).strip().lower()
        if not text:
            continue
        entry = memory.get(text) if isinstance(memory.get(text), dict) else {}
        scores = entry.get("effective_category_scores") or unit.get("effective_category_scores") \
            or unit.get("category_scores") or {}
        if not isinstance(scores, dict):
            continue
        try:
            up = sum(float(scores.get(a, 0.0)) for a in missing)
            down = sum(float(scores.get(a, 0.0)) for a in wrong)
            if wrong_appearance:
                down += float(scores.get("appearance", 0.0))
        except (TypeError, ValueError):
            continue
        delta = up - (down if damp else 0.0)
        if delta <= 0.0:
            continue
        weight = 1.0 + float(strength) * scale(entry.get("kind", unit.get("kind"))) * delta
        weight = max(MIN_LEARNED_WEIGHT, min(MAX_LEARNED_WEIGHT, weight))
        if abs(weight - 1.0) > 1e-3:
            out.append((text, weight))
    # Longest first: a phrase and one of its own words can both be weighted, and applying the
    # phrase first means the word's bias lands on top of it rather than instead of it.
    out.sort(key=lambda p: -len(p[0]))
    return out


def spans_from_ranges(ranges, weight):
    """`_v2_find_phrase_token_ranges` gives (start, end) pairs; pair them with a weight."""
    return [(int(s), int(e), float(weight)) for s, e in ranges or [] if int(e) > int(s)]


def locate(tokenizer, prompt_text, weighted_phrases):
    """Weighted phrases -> ([(start_tok, end_tok, weight)], prompt_token_count).

    Deliberately NOT `_v2_find_phrase_token_ranges`, which finds a phrase by cosine
    similarity against the encoded sequence and so re-encodes every phrase it is given. On
    H3 the text encoder is Qwen3-VL-32B, making that up to eight extra 32B forward passes
    per generation for something the tokenizer answers exactly and for free.

    Matching is case-insensitive because phrase memory stores everything lowercased while
    the prompt keeps the user's capitalisation. Every occurrence is weighted, not just the
    first: a word repeated for emphasis meant it both times.
    """
    text = str(prompt_text or "")
    if not text:
        return [], 0
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = list(enc["offset_mapping"])
    except Exception:  # noqa: BLE001 — no offsets means no weighting, not a broken run
        return [], 0
    total = len(offsets)
    if not weighted_phrases:
        return [], total
    haystack = text.lower()
    char_spans = []
    for phrase, weight in weighted_phrases:
        needle = str(phrase or "").strip().lower()
        if not needle:
            continue
        start = haystack.find(needle)
        while start != -1:
            char_spans.append((start, start + len(needle), float(weight)))
            start = haystack.find(needle, start + len(needle))
    return token_spans_from_offsets(offsets, char_spans), total


def token_spans_from_offsets(offsets, char_spans):
    """Character spans -> token index ranges, given a tokenizer's offset mapping."""
    out = []
    for c_start, c_end, weight in char_spans:
        toks = [i for i, (a, b) in enumerate(offsets) if a < c_end and b > c_start]
        if toks:
            out.append((toks[0], toks[-1] + 1, float(weight)))
    return out


def h3_tokenizer(clip):
    """The raw HF tokenizer behind an H3 CLIP, or None.

    `clip.tokenizer` is a MiniMaxH3Tokenizer, which holds a Qwen3VLSDTokenizer under the
    encoder's name, which holds the transformers tokenizer. Verified to return real offset
    mappings for this vocabulary; anything else here returns None and weighting stays off.
    """
    try:
        inner = getattr(getattr(clip, "tokenizer", None), "qwen3vl_32b", None)
        tok = getattr(inner, "tokenizer", None)
        if tok is None:
            return None
        tok("probe", add_special_tokens=False, return_offsets_mapping=True)
        return tok
    except Exception:  # noqa: BLE001
        return None
