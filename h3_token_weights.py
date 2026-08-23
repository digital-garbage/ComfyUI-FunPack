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
# How far the text run may exceed the measured prompt before placement is refused. A couple
# of tokens is a separator or a trimmed tag; more than that means a different string.
_BASE_SLACK = 4
# How far overlapping spans may reinforce each other, as a multiple of the STRONGEST single
# span at that position. A lone span is never touched — an explicit (word:2.0) means 2.0 —
# but three overlapping x1.5 spans stop at x1.75 instead of compounding to x3.4.
OVERLAP_HEADROOM = 1.5


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


def build_bias(spans, prompt_tokens: int, cond_len: int, seq_len: int, device, dtype,
               base: Optional[int] = None):
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
    if base is None:
        base = cond_len - prompt_tokens
    if base < 0 or base + prompt_tokens > cond_len:
        return None
    bias = None
    strongest = 0.0
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
        strongest = max(strongest, abs(b))
    if bias is not None and strongest > 0.0:
        # Spans OVERLAP — a phrase and one of its own words both match the same tokens, and
        # their biases ADD. That is intended, so the word reinforces the phrase. Unbounded it
        # compounds: three overlapping x1.5 spans is an effective x3.4 on one token, which is
        # far past anything weights_from_memory would ever return and reads as the prompt
        # collapsing onto a few words.
        #
        # Bounded against the strongest SINGLE span rather than a constant, so a lone span is
        # never touched: an explicit (word:2.0) still means 2.0.
        limit = min(MAX_ABS_BIAS, strongest * OVERLAP_HEADROOM)
        bias.clamp_(min=-limit, max=limit)
    return bias


def make_override(spans, prompt_tokens: int, cond_len: int, inner=None, on_apply=None,
                  base: Optional[int] = None):
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
                                        k.device, k.dtype, base=base)
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
        out.append((text, delta * scale(entry.get("kind", unit.get("kind")))))
    if not out:
        return []
    # RELATIVE, not absolute. Category scores are confidences in the 0..1 range and a phrase
    # rarely carries much of any one axis, so `1 + strength * score` put a whole prompt at
    # x1.03 — measurably applied and doing nothing. Normalising against the strongest
    # candidate makes `strength` mean "how much the best phrase gets", which is a number
    # worth setting, and keeps the ordering the memory learned.
    top = max(score for _, score in out)
    if top <= 0.0:
        return []
    scored = []
    for text, score in out:
        weight = 1.0 + float(strength) * (score / top)
        weight = max(MIN_LEARNED_WEIGHT, min(MAX_LEARNED_WEIGHT, weight))
        if abs(weight - 1.0) > 1e-3:
            scored.append((text, weight))
    # Strongest first, so a cap downstream keeps the phrases the rating cared MOST about —
    # taking the longest instead meant an arbitrary eight out of ninety-four.
    scored.sort(key=lambda p: -p[1])
    return scored


def order_for_application(weighted):
    """Longest first. A phrase and one of its own words can both be weighted, and applying
    the phrase first means the word's bias lands on top of it rather than instead of it.
    Separate from the ranking above, which decides WHICH phrases survive a cap."""
    return sorted(weighted or [], key=lambda p: -len(p[0]))


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


def prompt_base(token_tags, cond_len, prompt_tokens):
    """Index of the prompt's first token inside the conditioning block, or None.

    The fallback is arithmetic: the tokenizer appends the prompt LAST, so it is the final
    `prompt_tokens` rows. That holds only if the text tokenized here is exactly the text that
    was encoded, and a run whose conditioning is 367 rows against 368 modality tags shows how
    easily that bookkeeping drifts.

    `minimax_token_tags` removes the assumption. Tags are 1 for text and 0 for a vision
    block, and references are `<Picture N>: ` + vision BEFORE the prompt, so the prompt is
    the last unbroken run of 1s. Using it means a reference or a trimmed tag shifts the
    prompt and the spans follow.
    """
    if prompt_tokens <= 0 or cond_len <= 0:
        return None
    tags = _as_list(token_tags)
    if tags:
        end = min(len(tags), cond_len)
        run_end = end
        while run_end > 0 and int(tags[run_end - 1]) != 1:
            run_end -= 1                      # trailing non-text (should not happen; cheap)
        run_start = run_end
        while run_start > 0 and int(tags[run_start - 1]) == 1:
            run_start -= 1
        run = run_end - run_start
        # The prompt is the tokenizer's LAST segment, so the tail of the text run is the
        # prompt — but only if the text measured here is the text that was encoded. When a
        # reference node owns the conditioning it encoded the whole combined prompt while
        # this may be one scene's, and taking the tail then would bias a window of the wrong
        # words. A run that does not match the measurement is not something to place by.
        if run == prompt_tokens:
            return run_start
        if run > prompt_tokens and run - prompt_tokens <= _BASE_SLACK:
            return run_end - prompt_tokens
        return None
    base = cond_len - prompt_tokens
    return base if base >= 0 else None


def _as_list(tags):
    if tags is None:
        return []
    try:
        return tags.reshape(-1).tolist()
    except AttributeError:
        return list(tags) if isinstance(tags, (list, tuple)) else []


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
