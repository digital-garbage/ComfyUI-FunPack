"""Attention-logit token weighting for MiniMax H3.

Qwen never applies (word:1.2) — ComfyUI's H3 tokenizer hardcodes every weight to 1.0 — so
the syntax has to be stripped before encoding and re-applied where the model can act on it:
the prompt's key positions in the packed self-attention stream.
"""
import math
import sys
import types

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


def test_the_ceiling_matches_studios_own_emphasis_constant():
    """_v2_build_attn2_patch has boosted an emphasised phrase by exactly 1.25 on LTX since
    long before this existed, and that path works. A different mechanism for the same intent
    has no business pushing harder."""
    assert tw.MAX_LEARNED_WEIGHT == pytest.approx(1.25)


def test_one_bad_rating_does_not_reach_the_ceiling():
    """The complaint that produced this: an Awful first generation is one data point."""
    one_bad = tw.strength_from_auto(0.030 * 1.45)      # _v2_auto_strength, bad_streak == 1
    assert 1.10 < 1 + one_bad < 1.15


def test_a_sustained_bad_streak_pushes_harder_but_still_short_of_the_ceiling():
    sustained = tw.strength_from_auto(0.030 * 2.20)    # bad_streak >= 3
    assert 1 + sustained > 1.18
    assert 1 + sustained < tw.MAX_LEARNED_WEIGHT


def test_a_good_streak_barely_emphasises_anything():
    assert 1 + tw.strength_from_auto(0.030 * 0.42) < 1.05


def test_an_unreadable_strength_lands_mid_scale_rather_than_at_the_top():
    assert tw.strength_from_auto(None) == pytest.approx(tw.EMPHASIS_CEILING * 0.5)
    assert tw.strength_from_auto(99.0) == pytest.approx(tw.EMPHASIS_CEILING)


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
                                 strength=0.2)
    assert got[0][1] == pytest.approx(1.2)


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


def test_a_text_run_that_does_not_match_the_measurement_refuses_to_place():
    """When a reference node owns the conditioning it encoded the WHOLE combined prompt,
    while the text measured here may be one scene's. Taking the tail then would bias a
    window of the wrong words — better to weight nothing than the wrong thing."""
    tags = [1] * 40                     # 40 text tokens encoded
    assert tw.prompt_base(tags, cond_len=40, prompt_tokens=6) is None


def test_an_exact_text_run_places_at_its_start():
    assert tw.prompt_base([1, 1, 0, 0, 1, 1, 1], cond_len=7, prompt_tokens=3) == 4


def test_a_small_overshoot_is_tolerated():
    """A separator token or a trimmed tag, not a different string."""
    assert tw.prompt_base([0, 0, 1, 1, 1, 1, 1], cond_len=7, prompt_tokens=4) == 3


def test_overlapping_spans_cannot_compound_without_limit():
    """A phrase and its own word both match the same tokens and their biases add. Intended —
    but three overlapping x1.5 spans is an effective x3.4 on one token, well past anything
    weights_from_memory would ever return."""
    spans = [(0, 4, 1.5), (0, 4, 1.5), (0, 4, 1.5)]
    bias = tw.build_bias(spans, prompt_tokens=4, cond_len=4, seq_len=20,
                         device="cpu", dtype=torch.float32)
    assert bias.max().item() == pytest.approx(math.log(1.5) * tw.OVERLAP_HEADROOM)
    assert math.exp(bias.max().item()) < 1.9


def test_a_single_span_is_never_clamped():
    """An explicit (word:2.0) has to still mean 2.0 — bounding the overlap must not quietly
    weaken a weight the user asked for."""
    bias = tw.build_bias([(0, 4, 2.0)], 4, 4, 20, "cpu", torch.float32)
    assert bias.max().item() == pytest.approx(math.log(2.0))


# --- the text that was actually encoded ------------------------------------

@pytest.fixture
def refiner():
    import conditioning
    return conditioning.FunPackVideoRefinerV2.__new__(conditioning.FunPackVideoRefinerV2)


class _H3Clip:
    def __init__(self):
        self.tokenizer = types.SimpleNamespace(
            qwen3vl_32b=types.SimpleNamespace(tokenizer=_FakeTokenizer()))

    def tokenize(self, text, **kwargs):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        # Deterministic stand-in for a real re-encode: every row of the "alternate" phrase's
        # conditioning is 5.0, so a test can predict the exact mean-vector shift by hand.
        return [(torch.full((1, len(tokens), 4), 5.0), {"pooled_output": None})]


def _apply(refiner, monkeypatch, meta, memory, variables=None, variability=0.0):
    import conditioning
    import minimax_h3
    monkeypatch.setattr(minimax_h3, "is_h3_clip", lambda c: True)
    monkeypatch.setattr(conditioning, "_log",
                        types.SimpleNamespace(failed=lambda *a, **k: None,
                                              note_on_change=lambda *a, **k: None, feature=lambda *a, **k: None),
                        raising=False)
    out = refiner._v2_apply_h3_token_weights(
        [[torch.zeros(1, 40, 8), dict(meta)]], _H3Clip(), phrase_memory=memory,
        axis_feedback={"missing_axes": ["details"]}, enabled=True,
        auto_strength=0.0435, variability=variability, variables=variables)
    return out[0][1].get("funpack_h3_token_weights")


def test_the_encoded_text_is_measured_not_the_raw_one(refiner, monkeypatch):
    """funpack_scene_text still holds `$style`; the conditioning was built from the resolved
    string. Measuring the raw one puts every span on the wrong words."""
    memory = {"neon rain": {"kind": "phrase",
                            "effective_category_scores": {"details": 0.9}}}
    tag = _apply(refiner, monkeypatch,
                 {"funpack_scene_text": "$style neon rain",
                  "funpack_encode_text": "cinematic neon rain"}, memory)

    assert tag is not None
    assert tag["prompt_tokens"] == len("cinematic neon rain")
    start, end, _w = tag["spans"][0]
    assert "cinematic neon rain"[start:end] == "neon rain"


def test_a_phrase_still_carrying_a_variable_is_resolved_before_matching(refiner, monkeypatch):
    """Phrase memory stores the RAW text, so `$style` survives in the phrase while the
    encoded prompt has it resolved — and then it matches nothing."""
    memory = {"$style rain": {"kind": "phrase",
                              "effective_category_scores": {"details": 0.9}}}
    tag = _apply(refiner, monkeypatch,
                 {"funpack_encode_text": "cinematic rain"}, memory,
                 variables=[{"name": "style", "value": "cinematic"}])

    assert tag is not None
    start, end, _w = tag["spans"][0]
    assert "cinematic rain"[start:end] == "cinematic rain"


def test_a_phrase_that_still_does_not_match_is_simply_skipped(refiner, monkeypatch):
    memory = {"nothing like it": {"kind": "phrase",
                                  "effective_category_scores": {"details": 0.9}}}
    assert _apply(refiner, monkeypatch, {"funpack_encode_text": "cinematic rain"},
                  memory) is None


def test_emphasis_is_skipped_when_a_wired_prompt_cannot_be_placed(refiner, monkeypatch):
    """Skipped only when no candidate can be the encoded prompt at all. A prompt LONGER than
    the tensor's text run cannot be the string that produced it, and placing it by arithmetic
    would bias a window of the wrong words silently."""
    memory = {"rain": {"kind": "phrase", "effective_category_scores": {"details": 0.9}}}
    tag = _apply(refiner, monkeypatch,
                 {"funpack_conditioning_owner": "wired",
                  "funpack_encode_text": "cinematic rain",   # 14 tokens
                  "minimax_token_tags": torch.ones(6),       # a 6-position text run
                  },
                 memory)
    assert tag is None


def test_a_wired_prompt_shorter_than_the_run_is_placed_at_the_end(refiner, monkeypatch):
    """Label tokens sit in front of the prompt; the prompt is still the tail."""
    memory = {"rain": {"kind": "phrase", "effective_category_scores": {"details": 0.9}}}
    tag = _apply(refiner, monkeypatch,
                 {"funpack_conditioning_owner": "wired",
                  "funpack_encode_text": "cinematic rain",   # 14 tokens
                  "minimax_token_tags": torch.ones(40),      # 26 label tokens, then the prompt
                  },
                 memory)
    assert tag is not None and tag["base"] == 26


def test_emphasis_still_applies_to_studios_own_encode_via_the_tail(refiner, monkeypatch):
    """Studio measured the text it encoded, so the arithmetic is sound there."""
    memory = {"rain": {"kind": "phrase", "effective_category_scores": {"details": 0.9}}}
    tag = _apply(refiner, monkeypatch, {"funpack_encode_text": "cinematic rain"}, memory)
    assert tag is not None


# --- finding the string the tensor was actually encoded from ---------------

def test_the_candidate_matching_the_text_run_is_chosen():
    """A wired conditioning was encoded by another node from the editor's expansion, not from
    Studio's copy. The tag map says how many text positions the tensor holds, so the right
    candidate can be CHECKED rather than guessed."""
    tags = [0] * 5 + [1] * 6                       # a vision block, then 6 text positions
    text, n, base = tw.choose_encoded_text(
        _FakeTokenizer(), ["a much longer prompt", "abcdef"], tags, cond_len=11)

    assert text == "abcdef"
    assert n == 6
    assert base == 5


def test_the_closest_fit_wins_when_several_are_close():
    tags = [0, 0] + [1] * 8
    text, _n, _b = tw.choose_encoded_text(
        _FakeTokenizer(), ["abcdef", "abcdefgh"], tags, cond_len=10)
    assert text == "abcdefgh"


def test_a_candidate_longer_than_the_run_is_never_chosen():
    """It cannot be what was encoded — the tensor has nowhere to put the extra tokens."""
    tags = [0, 0] + [1] * 4
    assert tw.choose_encoded_text(_FakeTokenizer(), ["abcdefgh"], tags, cond_len=6) \
        == (None, 0, None)


def test_no_tags_means_no_verified_choice():
    assert tw.choose_encoded_text(_FakeTokenizer(), ["abc"], None, cond_len=6) \
        == (None, 0, None)


def test_the_editor_link_texts_are_offered_as_candidates():
    """server._expanded_link_texts is what a node outside Studio actually received."""
    import conditioning
    src = inspect_source(conditioning.FunPackVideoRefinerV2._v2_apply_h3_token_weights)
    assert 'link_texts.get(k) for k in ("full_prompt", "prompt")' in src
    assert "choose_encoded_text" in src


def inspect_source(fn):
    import inspect
    return inspect.getsource(fn)


# --- the reference / prompt boundary ---------------------------------------

def test_the_prompt_is_the_trailing_run_of_text_tags():
    """An r2v conditioning is laid out reference-first: label, vision block, prompt."""
    assert tw.prompt_region([1, 0, 0, 0, 1, 1, 1], cond_len=7) == (4, 7)


def test_a_text_only_conditioning_is_all_prompt():
    assert tw.prompt_region([1, 1, 1], cond_len=3) == (0, 3)


def test_no_tags_means_no_boundary():
    assert tw.prompt_region(None, cond_len=7) is None
    assert tw.prompt_region([], cond_len=7) is None


def test_a_conditioning_shorter_than_its_tags_is_measured_by_the_tensor():
    assert tw.prompt_region([1, 0, 0, 1, 1, 1, 1], cond_len=5) == (3, 5)


def test_an_all_image_conditioning_has_no_prompt():
    assert tw.prompt_region([0, 0, 0], cond_len=3) is None


def test_an_audio_label_in_front_of_the_prompt_does_not_shift_it():
    """An AUDIO reference contributes "<Audio n>: " and no vision block, so it lands inside
    the same run of text tags as the prompt. Measuring the prompt from the END of the run
    places it correctly however many label tokens precede it; requiring the run to nearly
    equal the prompt refused these runs outright."""
    # [<Picture 1>: , vision x3, <Audio 1>:  (7 chars), prompt (6 chars)]
    tags = [1] * 2 + [0] * 3 + [1] * 13
    text, n, base = tw.choose_encoded_text(_FakeTokenizer(), ["abcdef"], tags, cond_len=18)

    assert text == "abcdef" and n == 6
    assert base == 12                       # 18 - 6, not the run start at 5


def test_the_region_start_is_still_where_the_last_vision_block_ends():
    """Protection uses the region; placement uses the verified prompt. They differ exactly by
    the label tokens between them, which belong to the reference, not the prompt."""
    tags = [1] * 2 + [0] * 3 + [1] * 13
    assert tw.prompt_region(tags, cond_len=18) == (5, 18)


def test_every_learned_phrase_in_the_prompt_is_weighted(refiner, monkeypatch):
    """No cap. Locating a phrase is a substring find plus an offset scan — the 8 this
    inherited belongs to _v2_find_phrase_token_ranges, which re-encodes each phrase through
    a 32B text encoder."""
    words = [f"w{i:02d}" for i in range(20)]      # fixed width: no phrase contains another
    memory = {w: {"kind": "phrase", "effective_category_scores": {"details": 0.9}}
              for w in words}
    tag = _apply(refiner, monkeypatch, {"funpack_encode_text": " ".join(words)}, memory)

    assert tag is not None
    assert len(tag["spans"]) == 20


def test_a_phrase_no_longer_in_the_prompt_is_not_weighted(refiner, monkeypatch):
    """Phrase memory keeps everything the session rated, including phrases from prompts
    since edited away. They matched nothing and were still named as weighted."""
    memory = {"gone": {"kind": "phrase", "effective_category_scores": {"details": 0.9}},
              "here": {"kind": "phrase", "effective_category_scores": {"details": 0.9}}}
    tag = _apply(refiner, monkeypatch, {"funpack_encode_text": "here"}, memory)

    assert tag is not None
    assert len(tag["spans"]) == 1


# --- variability dial -------------------------------------------------------

def test_variability_zero_matches_the_old_no_dial_behaviour(refiner, monkeypatch):
    """0 is the historical default — the dial must be able to fully disappear."""
    memory = {"neon rain": {"kind": "phrase",
                            "effective_category_scores": {"details": 0.9}}}
    meta = {"funpack_encode_text": "cinematic neon rain"}
    with_dial = _apply(refiner, monkeypatch, meta, memory, variability=0.0)
    no_dial = _apply(refiner, monkeypatch, meta, memory)
    assert with_dial["spans"][0][2] == pytest.approx(no_dial["spans"][0][2])


def test_variability_one_removes_the_bias_entirely(refiner, monkeypatch):
    """1 = the emphasis this rating earned is switched off — spans still get located (so a
    weaker signal is visible if it ever comes back on) but the learned weight is 1.0, meaning
    no bias reaches the attention override at all."""
    memory = {"neon rain": {"kind": "phrase",
                            "effective_category_scores": {"details": 0.9}}}
    tag = _apply(refiner, monkeypatch,
                 {"funpack_encode_text": "cinematic neon rain"}, memory, variability=1.0)
    assert tag is None  # weight collapses to 1.0 -> weights_from_memory drops it -> nothing to locate


def test_variability_partial_shrinks_the_weight_without_removing_it(refiner, monkeypatch):
    memory = {"neon rain": {"kind": "phrase",
                            "effective_category_scores": {"details": 0.9}}}
    meta = {"funpack_encode_text": "cinematic neon rain"}
    full = _apply(refiner, monkeypatch, meta, memory, variability=0.0)
    half = _apply(refiner, monkeypatch, meta, memory, variability=0.5)
    full_w = full["spans"][0][2]
    half_w = half["spans"][0][2]
    assert 1.0 < half_w < full_w


def test_variability_out_of_range_is_clamped_not_inverted(refiner, monkeypatch):
    """A value above 1 (or a stray negative) must not push the bias past full strength or
    flip its sign — it is a dial with a floor at zero, not open-ended."""
    memory = {"neon rain": {"kind": "phrase",
                            "effective_category_scores": {"details": 0.9}}}
    meta = {"funpack_encode_text": "cinematic neon rain"}
    full = _apply(refiner, monkeypatch, meta, memory, variability=0.0)
    over = _apply(refiner, monkeypatch, meta, memory, variability=2.0)
    assert over is None  # 1.0 - 2.0 clamped to 0.0 by max(0.0, ...) -> same as variability=1.0
    assert full["spans"][0][2] > 1.0


# --- timed phrases -----------------------------------------------------------------

def test_timed_syntax_is_stripped_and_windows_kept():
    clean, spans = tw.parse_timed("a man [walks left@1.0-2.5] then [waves:1.5 @ 3-4] (x:1.2)")
    assert clean == "a man walks left then waves x"
    assert spans == [(6, 16, 1.0, 1.0, 2.5), (22, 27, 1.5, 3.0, 4.0)]


def test_both_markups_are_parsed_in_one_pass_on_one_clean_text():
    clean, weighted, timed, blended = tw.parse_markup("a (cat:1.5) [runs@1-2] (dog:0.5)")
    assert clean == "a cat runs dog"
    assert weighted == [(2, 5, 1.5), (11, 14, 0.5)]
    assert timed == [(6, 10, 1.0, 1.0, 2.0)]
    assert blended == []


def test_a_blended_phrase_is_parsed_and_phrase_a_stays_in_the_clean_text():
    clean, weighted, timed, blended = tw.parse_markup("a cat [sits|stands] on a mat")
    assert clean == "a cat sits on a mat"
    assert weighted == []
    assert timed == []
    assert blended == [(6, 10, "stands")]


def test_a_blend_alongside_a_weight_and_a_window_in_one_pass():
    clean, weighted, timed, blended = tw.parse_markup(
        "a cat [sits|stands] then [runs@1-2] (fast:1.5)")
    assert clean == "a cat sits then runs fast"
    assert blended == [(6, 10, "stands")]
    assert timed == [(16, 20, 1.0, 1.0, 2.0)]
    assert weighted == [(21, 25, 1.5)]


def test_an_empty_alternate_phrase_is_not_a_blend():
    assert tw.parse_markup("a cat [sits|  ]")[3] == []


def test_an_empty_window_is_dropped():
    assert tw.parse_timed("[walks@2-2]")[1] == []


def test_a_window_maps_to_whole_latent_frames():
    # latent_t=27 <-> 90 pixel frames: frame k spans FRAME_PER_TOKEN[k % 5] pixels.
    # 1.0s = pixel 24 sits in latent frame 7 (pixels 22-25); 2.5s = pixel 60 ends frame 17.
    assert tw.video_row_window(27, 4, 1.0, 2.5) == (7 * 4, 18 * 4)
    assert tw.video_row_window(27, 4, 0.0, 99.0) == (0, 27 * 4)
    assert tw.video_row_window(27, 4, 10.0, 11.0) is None


def _softmax_attention(q, k, v, heads, mask=None, **kw):
    s = q @ k.transpose(-1, -2)
    if mask is not None:
        s = s + mask
    return s.softmax(-1) @ v


def test_the_chunked_override_equals_a_full_per_query_bias():
    """Splitting queries at window edges must be EXACT, and only video rows see windows."""
    torch.manual_seed(0)
    cond_len, prompt_tokens, latent_t, frame_rows = 9, 5, 4, 3
    audio = 6
    seq = cond_len + audio + latent_t * frame_rows
    q, k, v = (torch.randn(1, 2, seq, 8) for _ in range(3))
    # prompt tokens 0-2 = "walks", allowed only in latent frame 1 (pixels 1-4 -> 0.05-0.2s)
    ov = tw.make_override([], prompt_tokens, cond_len, timed=[(0, 2, 1.0, 0.05, 0.2)],
                          latent_t=latent_t, frame_rows=frame_rows)
    out = ov(_softmax_attention, q, k, v, 2, skip_reshape=True, skip_output_reshape=True)

    full = torch.zeros(1, 1, seq, seq)
    video_start = cond_len + audio
    lo, hi = cond_len - prompt_tokens, cond_len - prompt_tokens + 2
    for r in range(video_start, seq):
        frame = (r - video_start) // frame_rows
        if frame != 1:
            full[0, 0, r, lo:hi] = tw.MASKED_BIAS
    want = _softmax_attention(q, k, v, 2, mask=full)
    assert torch.allclose(out, want, atol=1e-6)


def test_timed_and_untimed_spans_stack_and_a_window_weight_is_a_boost():
    seen = []

    def func(q, k, v, heads, mask=None, **kw):
        seen.append((q.shape[2], mask.clone()))
        return q

    cond_len, prompt_tokens, latent_t, frame_rows = 9, 5, 2, 2
    seq = cond_len + latent_t * frame_rows
    ov = tw.make_override([(3, 5, 2.0)], prompt_tokens, cond_len,
                          timed=[(0, 2, 1.5, 0.0, 0.03)], latent_t=latent_t, frame_rows=frame_rows)
    ov(func, _q(seq), _q(seq), _q(seq), 2, skip_reshape=True)
    # prefix chunk, frame 0 (inside), frame 1 (outside)
    assert [n for n, _ in seen] == [cond_len, frame_rows, frame_rows]
    prefix, inside, outside = (m for _, m in seen)
    # prompt starts at row 4: untimed (3,5) -> rows 7-8, timed (0,2) -> rows 4-5
    assert prefix[0, 0, 0, 7] == inside[0, 0, 0, 7] == pytest.approx(math.log(2.0))
    assert inside[0, 0, 0, 4] == pytest.approx(math.log(1.5)) and prefix[0, 0, 0, 4] == 0
    assert outside[0, 0, 0, 4] == tw.MASKED_BIAS


def test_a_phrase_may_contain_parentheses_colons_and_at_signs():
    clean, spans = tw.parse_timed("[camera: (slow) push-in, mail@x.com @1-2] [fast:0.5@2-3]")
    assert clean == "camera: (slow) push-in, mail@x.com fast"
    assert spans == [(0, 34, 1.0, 1.0, 2.0), (35, 39, 0.5, 2.0, 3.0)]


# --- timed phrases reach the sampler on both encode paths ----------------------

def _quiet(monkeypatch):
    import conditioning
    import minimax_h3
    monkeypatch.setattr(minimax_h3, "is_h3_clip", lambda c: True)
    monkeypatch.setattr(conditioning, "_log",
                        types.SimpleNamespace(failed=lambda *a, **k: None,
                                              note_on_change=lambda *a, **k: None,
                                              feature=lambda *a, **k: None), raising=False)


def test_studio_encoded_timed_phrases_become_token_windows(refiner, monkeypatch):
    _quiet(monkeypatch)
    clean, timed = tw.parse_timed("cat [runs@1-2]")
    meta = {"funpack_h3_timed": timed, "funpack_h3_timed_text": clean,
            "minimax_token_tags": [1] * 8}
    out = refiner._v2_apply_h3_timed_phrases([[torch.zeros(1, 8, 4), meta]], _H3Clip())
    tag = out[0][1]["funpack_h3_token_weights"]
    assert tag["prompt_tokens"] == 8 and tag["base"] == 0
    assert tag["timed"] == [(4, 8, 1.0, 1.0, 2.0)]


def test_wired_conditioning_takes_its_windows_from_the_editor(refiner, monkeypatch):
    """The editor stripped the syntax before the encoding node saw it and hands the windows
    over as char spans on the text it gave that node."""
    _quiet(monkeypatch)
    link = {"prompt": "cat runs", "full_prompt": "cat runs", "timed": [[4, 8, 1.0, 1.0, 2.0]]}
    # reference block (tag 0) in front, then the 8-token prompt
    meta = {"funpack_conditioning_owner": "wired", "minimax_token_tags": [0, 0, 0] + [1] * 8}
    out = refiner._v2_apply_h3_timed_phrases([[torch.zeros(1, 11, 4), meta]], _H3Clip(),
                                             link_texts=link)
    tag = out[0][1]["funpack_h3_token_weights"]
    assert tag["base"] == 3 and tag["timed"] == [(4, 8, 1.0, 1.0, 2.0)]


def test_a_wired_conditioning_without_editor_windows_is_left_alone(refiner, monkeypatch):
    _quiet(monkeypatch)
    meta = {"funpack_conditioning_owner": "wired", "minimax_token_tags": [1] * 8}
    out = refiner._v2_apply_h3_timed_phrases([[torch.zeros(1, 8, 4), meta]], _H3Clip(),
                                             link_texts={"prompt": "cat runs"})
    assert "funpack_h3_token_weights" not in out[0][1]


def test_typed_weights_reach_the_sampler_next_to_the_windows(refiner, monkeypatch):
    _quiet(monkeypatch)
    clean, weighted, timed, _blended = tw.parse_markup("(cat:2) [runs@1-2]")
    meta = {"funpack_h3_timed": timed, "funpack_h3_weighted": weighted,
            "funpack_h3_timed_text": clean, "minimax_token_tags": [1] * 8}
    out = refiner._v2_apply_h3_timed_phrases([[torch.zeros(1, 8, 4), meta]], _H3Clip())
    tag = out[0][1]["funpack_h3_token_weights"]
    assert tag["spans"] == [(0, 3, 2.0)] and tag["timed"] == [(4, 8, 1.0, 1.0, 2.0)]


def test_a_typed_weight_alone_is_enough(refiner, monkeypatch):
    _quiet(monkeypatch)
    meta = {"funpack_h3_weighted": [(0, 3, 1.5)], "funpack_h3_timed_text": "cat runs",
            "minimax_token_tags": [1] * 8}
    out = refiner._v2_apply_h3_timed_phrases([[torch.zeros(1, 8, 4), meta]], _H3Clip())
    assert out[0][1]["funpack_h3_token_weights"]["spans"] == [(0, 3, 1.5)]


def test_a_bracketed_weight_without_a_window_is_a_plain_weight():
    clean, weighted, timed, _blended = tw.parse_markup("a [cat (sitting):1.5] [runs@1-2]")
    assert clean == "a cat (sitting) runs"
    assert weighted == [(2, 15, 1.5)] and timed == [(16, 20, 1.0, 1.0, 2.0)]


# --- phrase blend: mean-vector shift, not a per-token blend --------------------------

def test_the_blended_span_is_shifted_by_half_the_mean_difference(refiner, monkeypatch):
    """cond's span is all 1.0; the fake clip's re-encode of the alternate phrase is all 5.0.
    Default strength is 0.5, so the span should land at 1.0 + 0.5*(5.0-1.0) = 3.0 — and
    nowhere else in the tensor should move."""
    _quiet(monkeypatch)
    text = "a cat sits on a mat"           # "sits" is chars [6:10]
    cond = torch.zeros(1, len(text), 4)
    cond[:, 6:10, :] = 1.0
    meta = {"funpack_h3_blended": [(6, 10, "stands")], "funpack_h3_timed_text": text,
            "minimax_token_tags": [1] * len(text)}
    out = refiner._v2_apply_h3_phrase_blend([[cond, meta]], _H3Clip())
    result = out[0][0]
    assert torch.allclose(result[:, 6:10, :], torch.full((1, 4, 4), 3.0))
    assert torch.allclose(result[:, :6, :], torch.zeros(1, 6, 4))
    assert torch.allclose(result[:, 10:, :], torch.zeros(1, len(text) - 10, 4))


def test_an_untagged_entry_is_left_completely_alone(refiner, monkeypatch):
    _quiet(monkeypatch)
    cond = torch.zeros(1, 8, 4)
    meta = {"minimax_token_tags": [1] * 8}
    out = refiner._v2_apply_h3_phrase_blend([[cond, meta]], _H3Clip())
    assert out[0][0] is cond


def test_wired_conditioning_blends_from_the_editors_link_texts(refiner, monkeypatch):
    """The wire owns the prompt: Studio never encoded it, so there is no
    `funpack_h3_timed_text` to read. The editor hands the alternate phrase over as
    `link_texts['blended']`, char spans on `link_texts['prompt']`, same mechanism as timed
    phrases. `choose_encoded_text` verifies "cat runs" against the tensor's own tag run
    (3 reference tokens + 8 text tokens) before trusting it."""
    _quiet(monkeypatch)
    link = {"prompt": "cat runs", "full_prompt": "cat runs", "blended": [[4, 8, "stands"]]}
    cond = torch.zeros(1, 11, 4)
    cond[:, 7:11, :] = 1.0                      # "runs" sits after the 3-token ref block
    meta = {"funpack_conditioning_owner": "wired", "minimax_token_tags": [0, 0, 0] + [1] * 8}
    out = refiner._v2_apply_h3_phrase_blend([[cond, meta]], _H3Clip(), link_texts=link)
    result = out[0][0]
    assert torch.allclose(result[:, 7:11, :], torch.full((1, 4, 4), 3.0))
    assert torch.allclose(result[:, :7, :], torch.zeros(1, 7, 4))


def test_wired_conditioning_is_skipped_not_silently_dropped_when_nothing_matches(refiner, monkeypatch):
    """Same wired setup, but neither editor text tokenizes to the tensor's prompt run — the
    honest result is no blend, logged, not a guess."""
    _quiet(monkeypatch)
    logged = []
    import conditioning
    monkeypatch.setattr(conditioning, "_log",
                        types.SimpleNamespace(failed=lambda *a, **k: None,
                                              note_on_change=lambda *a, **k: None,
                                              feature=lambda *a, **k: logged.append(a)),
                        raising=False)
    link = {"prompt": "totally different text", "full_prompt": "totally different text",
            "blended": [[4, 8, "stands"]]}
    cond = torch.zeros(1, 11, 4)
    cond[:, 7:11, :] = 1.0
    meta = {"funpack_conditioning_owner": "wired", "minimax_token_tags": [0, 0, 0] + [1] * 8}
    out = refiner._v2_apply_h3_phrase_blend([[cond, meta]], _H3Clip(), link_texts=link)
    assert torch.equal(out[0][0], cond)          # untouched
    assert logged and logged[0][1] == "H3 phrase blend" and logged[0][2] is False


def test_a_wired_conditioning_without_editor_blend_spans_is_left_alone(refiner, monkeypatch):
    _quiet(monkeypatch)
    cond = torch.zeros(1, 11, 4)
    meta = {"funpack_conditioning_owner": "wired", "minimax_token_tags": [0, 0, 0] + [1] * 8}
    out = refiner._v2_apply_h3_phrase_blend([[cond, meta]], _H3Clip())
    assert out[0][0] is cond


def test_a_non_h3_clip_is_left_alone(refiner, monkeypatch):
    import minimax_h3
    monkeypatch.setattr(minimax_h3, "is_h3_clip", lambda c: False)
    cond = torch.zeros(1, 8, 4)
    meta = {"funpack_h3_blended": [(0, 3, "x")], "funpack_h3_timed_text": "cat runs"}
    out = refiner._v2_apply_h3_phrase_blend([[cond, meta]], _H3Clip())
    assert out[0][0] is cond


# --- adversarial-review fixes: a colon in the alt phrase, double-apply, partial failure ----

def test_a_colon_in_the_alt_phrase_does_not_become_a_blend():
    """[cat|dog:1.5] used to parse as a blend whose alt phrase was the literal string
    "dog:1.5" -- colon and digit intact -- which then reached the real text encoder
    unstripped on re-encode (there is no bracket left around it once substituted into the
    sentence for a second parse_markup pass to catch). Excluding ':' and '@' from the alt
    phrase's character class means this now falls through to the plain bracket-weight
    syntax instead of silently leaking markup into the model."""
    clean, weighted, timed, blended = tw.parse_markup("a [cat|dog:1.5] man")
    assert blended == []
    assert weighted == [(2, 9, 1.5)]
    assert clean == "a cat|dog man"


def test_applying_the_blend_twice_does_not_compound(refiner, monkeypatch):
    """Every other H3 conditioning stage here only writes metadata, so re-running it is
    harmless. This one mutates the tensor, so the tag must not survive being consumed --
    otherwise a second pass would measure the ALREADY-shifted span as `mean_a` and shift it
    again past the intended one-time strength."""
    _quiet(monkeypatch)
    text = "a cat sits on a mat"
    cond = torch.zeros(1, len(text), 4)
    cond[:, 6:10, :] = 1.0
    meta = {"funpack_h3_blended": [(6, 10, "stands")], "funpack_h3_timed_text": text,
            "minimax_token_tags": [1] * len(text)}
    once = refiner._v2_apply_h3_phrase_blend([[cond, meta]], _H3Clip())
    twice = refiner._v2_apply_h3_phrase_blend(once, _H3Clip())
    assert torch.equal(once[0][0], twice[0][0])
    assert "funpack_h3_blended" not in once[0][1]


def test_applying_a_wired_blend_twice_with_the_same_link_texts_does_not_compound(refiner, monkeypatch):
    """The Studio-tagged case above is protected by the pop alone, because nothing re-adds
    `funpack_h3_blended` afterward. A WIRED entry is different: the top-of-function block
    re-injects that tag on every call whenever it is absent -- which is exactly the state
    the pop leaves an already-blended entry in -- so a second call with the SAME link_texts
    would re-tag and re-blend it without a separate, never-popped 'already handled'
    marker."""
    _quiet(monkeypatch)
    link = {"prompt": "cat runs", "full_prompt": "cat runs", "blended": [[4, 8, "stands"]]}
    cond = torch.zeros(1, 11, 4)
    cond[:, 7:11, :] = 1.0
    meta = {"funpack_conditioning_owner": "wired", "minimax_token_tags": [0, 0, 0] + [1] * 8}
    once = refiner._v2_apply_h3_phrase_blend([[cond, meta]], _H3Clip(), link_texts=link)
    twice = refiner._v2_apply_h3_phrase_blend(once, _H3Clip(), link_texts=link)
    assert torch.equal(once[0][0], twice[0][0])
    assert torch.allclose(once[0][0][:, 7:11, :], torch.full((1, 4, 4), 3.0))


def test_one_failed_span_is_reported_not_swallowed(refiner, monkeypatch, capsys):
    """Two blend spans in one prompt, one of which cannot be located (its char span points
    past the end of the tokenized text) -- the surviving span still applies, and the
    console line says something did NOT land rather than only counting what did."""
    _quiet(monkeypatch)
    text = "a cat sits on a mat"
    cond = torch.zeros(1, len(text), 4)
    cond[:, 6:10, :] = 1.0
    meta = {"funpack_h3_blended": [(6, 10, "stands"), (100, 104, "nowhere")],
            "funpack_h3_timed_text": text, "minimax_token_tags": [1] * len(text)}
    out = refiner._v2_apply_h3_phrase_blend([[cond, meta]], _H3Clip())
    assert torch.allclose(out[0][0][:, 6:10, :], torch.full((1, 4, 4), 3.0))
    printed = capsys.readouterr().out
    assert "1 phrase(s)" in printed and "1 could not be placed" in printed
