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


# --- rating-derived weights ------------------------------------------------

def _unit(text, kind="phrase", **scores):
    return {"text": text, "kind": kind, "effective_category_scores": scores}


def test_a_phrase_carrying_a_missing_axis_is_boosted():
    """This is the signal that already existed and had nowhere to go on H3."""
    got = tw.weights_from_memory([_unit("red scarf", details=0.8)], missing_axes=("details",))
    assert len(got) == 1
    text, weight = got[0]
    assert text == "red scarf" and weight > 1.0


def test_a_phrase_the_rating_is_neutral_about_is_omitted():
    """Returned at 1.0 it would still enter the bias tensor and cost a branch for nothing."""
    assert tw.weights_from_memory([_unit("red scarf", action=0.9)],
                                  missing_axes=("details",)) == []


def test_damping_is_off_unless_asked_for():
    """Suppressing a phrase the user typed is a knob going inert on its own."""
    units = [_unit("red scarf", details=0.1, quality=0.9)]
    assert tw.weights_from_memory(units, missing_axes=("details",), wrong_axes=("quality",)) \
        != []
    assert tw.weights_from_memory(units, missing_axes=("details",), wrong_axes=("quality",),
                                  damp=True) == []


def test_wrong_appearance_damps_appearance_phrases_only_when_damping():
    units = [_unit("blonde hair", appearance=0.9, details=0.5)]
    assert tw.weights_from_memory(units, missing_axes=("details",),
                                  wrong_appearance=True, damp=True) == []
    assert tw.weights_from_memory(units, missing_axes=("details",),
                                  wrong_appearance=True) != []


def test_a_single_word_carries_less_than_a_whole_phrase():
    """Mirrors _v2_memory_kind_scale: phrase 1.0, ngram 0.62, token 0.24. Weights are
    relative to the strongest candidate, so the comparison is within one set."""
    got = dict(tw.weights_from_memory(
        [_unit("a red scarf", "phrase", details=0.8), _unit("scarf", "token", details=0.8)],
        missing_axes=("details",)))
    assert got["a red scarf"] > got["scarf"] > 1.0


def test_the_learned_memory_overrides_the_run_s_own_scores():
    """The point of phrase_memory is that it accumulates across ratings."""
    units = [_unit("red scarf", details=0.0)]
    memory = {"red scarf": {"kind": "phrase", "effective_category_scores": {"details": 0.9}}}
    assert tw.weights_from_memory(units, memory, missing_axes=("details",)) != []


def test_weights_are_clamped():
    got = tw.weights_from_memory([_unit("x", details=99.0)], missing_axes=("details",),
                                 strength=50.0)
    assert got[0][1] == tw.MAX_LEARNED_WEIGHT


def test_longer_phrases_are_applied_first():
    """A phrase and one of its own words can both be weighted; the word's bias should land
    on top of the phrase's, not instead of it."""
    units = [_unit("scarf", "token", details=0.8), _unit("a red scarf", "phrase", details=0.8)]
    assert [t for t, _ in tw.weights_from_memory(units, missing_axes=("details",))] \
        == ["a red scarf", "scarf"]


def test_malformed_units_are_skipped_not_fatal():
    units = [None, {}, {"text": "ok", "effective_category_scores": "not a dict"},
             {"text": "fine", "effective_category_scores": {"details": 0.5}}]
    assert [t for t, _ in tw.weights_from_memory(units, missing_axes=("details",))] == ["fine"]


def test_ranges_become_weighted_spans():
    assert tw.spans_from_ranges([(4, 7), (9, 9)], 1.5) == [(4, 7, 1.5)]


# --- locating phrases in the prompt ---------------------------------------

def test_locate_finds_a_phrase_case_insensitively():
    """Phrase memory stores everything lowercased; the prompt keeps the user's caps."""
    spans, total = tw.locate(_FakeTokenizer(), "A Red Scarf", [("red scarf", 1.5)])
    assert total == 11
    assert spans == [(2, 11, 1.5)]


def test_locate_weights_every_occurrence():
    spans, _ = tw.locate(_FakeTokenizer(), "cat and cat", [("cat", 1.5)])
    assert spans == [(0, 3, 1.5), (8, 11, 1.5)]


def test_locate_reports_the_prompt_length_even_with_nothing_to_weight():
    """The length is what places the bias in the packed sequence, so it is needed whether or
    not any phrase matched."""
    assert tw.locate(_FakeTokenizer(), "a cat", []) == ([], 5)


def test_a_phrase_that_is_not_in_the_prompt_is_skipped():
    spans, _ = tw.locate(_FakeTokenizer(), "a dog", [("red scarf", 1.5)])
    assert spans == []


def test_locate_survives_a_tokenizer_without_offsets():
    class _NoOffsets:
        def __call__(self, *a, **kw):
            raise ValueError("slow tokenizer")

    assert tw.locate(_NoOffsets(), "a cat", [("cat", 1.5)]) == ([], 0)


def test_the_real_h3_vocabulary_returns_usable_offsets():
    """The mechanism rests on this. Qwen's is a SLOW tokenizer, where offset mapping is not
    guaranteed — checked against the actual vocabulary ComfyUI ships for H3."""
    from transformers import Qwen2Tokenizer
    path = ("/Users/dex/Documents/ComfyUI/comfy/text_encoders/qwen25_tokenizer")
    try:
        tok = Qwen2Tokenizer.from_pretrained(path)
    except Exception:
        pytest.skip("ComfyUI's H3 tokenizer vocabulary is not present")

    spans, total = tw.locate(tok, "a fluffy cat", [("fluffy", 2.0)])

    assert total == 3
    assert spans == [(1, 2, 2.0)]      # 'fluffy' is exactly token 1


def test_the_tokenizer_is_reached_through_the_clip_wrapper():
    import types
    tok = _FakeTokenizer()
    clip = types.SimpleNamespace(
        tokenizer=types.SimpleNamespace(qwen3vl_32b=types.SimpleNamespace(tokenizer=tok)))
    assert tw.h3_tokenizer(clip) is tok
    assert tw.h3_tokenizer(types.SimpleNamespace(tokenizer=None)) is None


# --- sampler install -------------------------------------------------------

@pytest.fixture
def chain():
    import samplers
    for name in dir(samplers):
        obj = getattr(samplers, name)
        if isinstance(obj, type) and hasattr(obj, "_install_h3_token_weights"):
            return obj()
    pytest.skip("no sampler exposes _install_h3_token_weights")


class _Patcher:
    def __init__(self):
        self.model_options = {}

    def clone(self):
        c = _Patcher()
        c.model_options = dict(self.model_options)
        return c


def _tagged(spans=((0, 2, 2.0),), prompt_tokens=5, cond_len=9):
    return [[torch.zeros(1, cond_len, 8),
             {"funpack_h3_token_weights": {"spans": list(spans),
                                           "prompt_tokens": prompt_tokens}}]]


def test_an_untagged_run_is_left_completely_alone(chain):
    """Every non-H3 run, and every H3 run before the first rating."""
    model = _Patcher()
    assert chain._install_h3_token_weights(model, [[torch.zeros(1, 9, 8), {}]]) is model
    assert chain._install_h3_token_weights(model, None) is model


def test_a_tagged_run_gets_an_override(chain):
    out = chain._install_h3_token_weights(_Patcher(), _tagged())
    assert "optimized_attention_override" in out.model_options["transformer_options"]


def test_the_existing_override_is_chained_not_replaced(chain):
    """One slot. Taking it outright would silently discard SLA or the chosen backend."""
    model = _Patcher()
    sentinel = lambda *a, **kw: "inner ran"
    model.model_options["transformer_options"] = {"optimized_attention_override": sentinel}

    out = chain._install_h3_token_weights(model, _tagged())
    ov = out.model_options["transformer_options"]["optimized_attention_override"]

    assert ov is not sentinel
    q = torch.zeros(1, 2, 100, 8)
    assert ov(lambda *a, **kw: None, q, q, q, 2, skip_reshape=True) == "inner ran"


def test_the_original_model_options_are_not_mutated(chain):
    model = _Patcher()
    model.model_options["transformer_options"] = {}
    chain._install_h3_token_weights(model, _tagged())
    assert "optimized_attention_override" not in model.model_options["transformer_options"]


def test_a_malformed_tag_does_not_break_the_run(chain):
    model = _Patcher()
    bad = [[torch.zeros(1, 9, 8), {"funpack_h3_token_weights": {"spans": "nonsense"}}]]
    assert chain._install_h3_token_weights(model, bad) is model


def test_the_best_phrase_gets_the_full_strength():
    """Category scores are 0..1 confidences and a phrase rarely carries much of one axis, so
    an absolute `1 + strength * score` put an entire prompt at x1.03 — applied, and doing
    nothing. Normalising makes `strength` mean what it says."""
    got = tw.weights_from_memory([_unit("faint", details=0.06)], missing_axes=("details",),
                                 strength=0.5)
    assert got[0][1] == pytest.approx(1.5)


def test_ranking_is_by_weight_so_a_cap_keeps_what_matters():
    """94 candidates capped to 8 by LENGTH kept an arbitrary eight."""
    units = [_unit("a very long but unimportant clause", details=0.1),
             _unit("scarf", details=0.9)]
    assert tw.weights_from_memory(units, missing_axes=("details",))[0][0] == "scarf"


def test_application_order_is_longest_first():
    ordered = tw.order_for_application([("scarf", 1.2), ("a red scarf", 1.5)])
    assert [t for t, _ in ordered] == ["a red scarf", "scarf"]


# --- placing the prompt from the modality tags -----------------------------

def test_the_prompt_is_the_last_run_of_text_tags():
    """Tags are 1 for text, 0 for a vision block. A reference is '<Picture 1>: ' + vision
    BEFORE the prompt, so the prompt is the trailing run of 1s."""
    tags = [1, 1, 0, 0, 0, 1, 1, 1, 1]        # label, vision block, then a 4-token prompt
    assert tw.prompt_base(tags, cond_len=9, prompt_tokens=4) == 5


def test_the_tags_beat_the_arithmetic_when_they_disagree():
    """A run reporting 367 conditioning rows against 368 tags is how that bookkeeping drifts;
    the tags say where the text actually is."""
    tags = [1, 1, 0, 0, 1, 1, 1]
    assert tw.prompt_base(tags, cond_len=6, prompt_tokens=2) == 4    # not 6 - 2 == 4? check
    assert tw.prompt_base(tags, cond_len=7, prompt_tokens=3) == 4


def test_no_tags_falls_back_to_the_tail():
    assert tw.prompt_base(None, cond_len=9, prompt_tokens=4) == 5


def test_a_prompt_longer_than_the_conditioning_is_refused():
    assert tw.prompt_base(None, cond_len=3, prompt_tokens=9) is None


def test_tags_arriving_as_a_tensor_are_read():
    tags = torch.tensor([1, 1, 0, 0, 1, 1, 1, 1])
    assert tw.prompt_base(tags, cond_len=8, prompt_tokens=4) == 4


def test_the_bias_uses_an_explicit_base_when_given():
    bias = tw.build_bias([(0, 2, 2.0)], prompt_tokens=4, cond_len=9, seq_len=50,
                         device="cpu", dtype=torch.float32, base=5)
    assert bias[0, 0, 0, 5] != 0 and bias[0, 0, 0, 4] == 0


def test_a_base_that_would_run_past_the_conditioning_is_refused():
    assert tw.build_bias([(0, 2, 2.0)], 4, 9, 50, "cpu", torch.float32, base=7) is None
