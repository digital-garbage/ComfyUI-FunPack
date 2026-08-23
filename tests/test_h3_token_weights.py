"""Attention-logit token weighting for MiniMax H3.

Qwen never applies (word:1.2) — ComfyUI's H3 tokenizer hardcodes every weight to 1.0 — so
the syntax has to be stripped before encoding and re-applied where the model can act on it:
the prompt's key positions in the packed self-attention stream.
"""
import math
import sys

import pytest
import torch

sys.path.insert(0, ".")
import h3_token_weights as tw  # noqa: E402


# --- parsing ---------------------------------------------------------------

def test_plain_text_is_returned_untouched():
    assert tw.parse("a cat on a mat") == ("a cat on a mat", [])


def test_the_syntax_is_removed_from_the_text():
    """Left in, Qwen reads the brackets and digits as part of the sentence — worse than no
    weighting at all."""
    clean, spans = tw.parse("a (fluffy:1.4) cat")
    assert clean == "a fluffy cat"
    assert [(s, e) for s, e, _ in spans] == [(2, 8)]
    assert clean[2:8] == "fluffy"


def test_spans_index_the_cleaned_text_not_the_original():
    clean, spans = tw.parse("(red:1.2) car and a (blue:0.5) boat")
    assert clean == "red car and a blue boat"
    assert [clean[s:e] for s, e, _ in spans] == ["red", "blue"]
    assert [w for _, _, w in spans] == [1.2, 0.5]


def test_a_negative_weight_is_parsed():
    _, spans = tw.parse("(blurry:-0.4) shot")
    assert spans[0][2] == -0.4


def test_an_escaped_bracket_is_left_alone():
    clean, spans = tw.parse(r"a \(not a weight:1.2\) thing")
    assert spans == []
    assert clean == r"a \(not a weight:1.2\) thing"


# --- weight -> bias --------------------------------------------------------

def test_a_weight_of_one_is_no_bias_at_all():
    assert tw.bias_value(1.0) == 0.0


def test_the_bias_is_the_log_of_the_weight():
    """softmax(logits + log w) multiplies that key's attention share by w. That identity is
    the whole reason this is the right place to weight."""
    assert tw.bias_value(2.0) == pytest.approx(math.log(2.0))
    assert tw.bias_value(0.5) == pytest.approx(math.log(0.5))


def test_zero_masks_instead_of_producing_negative_infinity():
    """log(0) is -inf, and a query whose every key is -inf comes out NaN."""
    assert tw.bias_value(0.0) == tw.MASKED_BIAS
    assert math.isfinite(tw.bias_value(0.0))


def test_an_extreme_weight_is_clamped():
    assert tw.bias_value(10_000.0) == tw.MAX_ABS_BIAS


# --- placing the bias in the packed sequence -------------------------------

def test_the_bias_lands_on_the_prompt_tail_of_the_conditioning():
    """References are prepended by the tokenizer, so the prompt occupies the END of the
    conditioning block, and the conditioning block leads the packed sequence."""
    bias = tw.build_bias([(0, 2, 2.0)], prompt_tokens=5, cond_len=9, seq_len=100,
                         device="cpu", dtype=torch.float32)

    assert bias.shape == (1, 1, 1, 100)
    assert torch.allclose(bias[0, 0, 0, 4:6], torch.full((2,), math.log(2.0)))
    assert bias[0, 0, 0, :4].abs().sum() == 0      # the reference tokens are untouched
    assert bias[0, 0, 0, 6:].abs().sum() == 0      # so is audio and video


def test_no_spans_means_no_tensor():
    assert tw.build_bias([], 5, 9, 100, "cpu", torch.float32) is None


def test_a_span_reaching_past_the_prompt_is_dropped_not_wrapped():
    assert tw.build_bias([(4, 99, 2.0)], prompt_tokens=5, cond_len=9, seq_len=100,
                         device="cpu", dtype=torch.float32) is not None
    assert tw.build_bias([(50, 60, 2.0)], prompt_tokens=5, cond_len=9, seq_len=100,
                         device="cpu", dtype=torch.float32) is None


def test_a_sequence_shorter_than_the_conditioning_is_refused():
    assert tw.build_bias([(0, 2, 2.0)], 5, 9, 4, "cpu", torch.float32) is None


# --- the override ----------------------------------------------------------

def _q(seq_len, heads=2, dim=8):
    return torch.zeros(1, heads, seq_len, dim)


def test_the_override_biases_the_packed_self_attention():
    seen = {}

    def func(q, k, v, heads, mask=None, **kw):
        seen["mask"] = mask
        return q

    ov = tw.make_override([(0, 2, 2.0)], prompt_tokens=5, cond_len=9)
    ov(func, _q(100), _q(100), _q(100), 2, skip_reshape=True)

    assert seen["mask"] is not None
    assert seen["mask"][0, 0, 0, 4] == pytest.approx(math.log(2.0))


def test_an_existing_mask_is_added_to_not_replaced():
    """Stomping a real padding mask would corrupt attention rather than weight it."""
    seen = {}

    def func(q, k, v, heads, mask=None, **kw):
        seen["mask"] = mask
        return q

    existing = torch.full((1, 1, 1, 100), -1.0)
    ov = tw.make_override([(0, 2, 2.0)], prompt_tokens=5, cond_len=9)
    ov(func, _q(100), _q(100), _q(100), 2, mask=existing, skip_reshape=True)

    assert seen["mask"][0, 0, 0, 4] == pytest.approx(-1.0 + math.log(2.0))
    assert seen["mask"][0, 0, 0, 0] == pytest.approx(-1.0)


def test_the_token_refiner_and_cross_shaped_calls_are_left_alone():
    """q and k are the same sequence only in the packed self-attention."""
    seen = {}

    def func(q, k, v, heads, mask=None, **kw):
        seen["mask"] = mask
        return q

    ov = tw.make_override([(0, 2, 2.0)], prompt_tokens=5, cond_len=9)
    ov(func, _q(100), _q(40), _q(40), 2, skip_reshape=True)
    assert seen["mask"] is None

    ov(func, _q(100), _q(100), _q(100), 2, skip_reshape=False)
    assert seen["mask"] is None


def test_the_displaced_override_still_runs():
    """There is one override slot. Taking it without chaining would silently discard the
    attention backend the user selected."""
    calls = []

    def inner(func, q, k, v, heads, mask=None, **kw):
        calls.append(mask)
        return q

    ov = tw.make_override([(0, 2, 2.0)], prompt_tokens=5, cond_len=9, inner=inner)
    ov(lambda *a, **kw: None, _q(100), _q(100), _q(100), 2, skip_reshape=True)

    assert len(calls) == 1 and calls[0] is not None


def test_a_failure_inside_the_override_keeps_the_step():
    """Weighting is a refinement. It must never be the reason a render dies."""
    def func(q, k, v, heads, mask=None, **kw):
        return "ran"

    ov = tw.make_override([(0, 2, 2.0)], prompt_tokens=5, cond_len=9)
    broken = torch.zeros(1)          # no .shape[2]; build_bias will raise
    assert ov(func, broken, broken, broken, 2, skip_reshape=True) == "ran"


def test_the_bias_is_built_once_per_sequence_shape():
    built = []

    def func(q, k, v, heads, mask=None, **kw):
        return q

    ov = tw.make_override([(0, 2, 2.0)], prompt_tokens=5, cond_len=9,
                          on_apply=built.append)
    for _ in range(4):
        ov(func, _q(100), _q(100), _q(100), 2, skip_reshape=True)

    assert built == [100]


# --- token spans -----------------------------------------------------------

class _FakeTokenizer:
    """Character-per-token, which makes the offset arithmetic checkable by hand."""
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_char_spans_become_token_ranges():
    clean, spans = tw.parse("a (bc:1.5) d")
    assert clean == "a bc d"
    assert tw.token_spans(_FakeTokenizer(), clean, spans) == [(2, 4, 1.5)]


def test_a_tokenizer_without_offsets_turns_weighting_off_rather_than_guessing():
    class _NoOffsets:
        def __call__(self, *a, **kw):
            raise ValueError("slow tokenizer")

    assert tw.token_spans(_NoOffsets(), "a bc d", [(2, 4, 1.5)]) == []
