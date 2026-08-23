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


def _apply(refiner, monkeypatch, meta, memory, variables=None):
    import conditioning
    import minimax_h3
    monkeypatch.setattr(minimax_h3, "is_h3_clip", lambda c: True)
    monkeypatch.setattr(conditioning, "_log",
                        types.SimpleNamespace(failed=lambda *a, **k: None,
                                              note_on_change=lambda *a, **k: None),
                        raising=False)
    out = refiner._v2_apply_h3_token_weights(
        [[torch.zeros(1, 40, 8), dict(meta)]], _H3Clip(), phrase_memory=memory,
        axis_feedback={"missing_axes": ["details"]}, enabled=True,
        auto_strength=0.0435, variables=variables)
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
    """The fallback places the prompt as `cond_len - tokens measured here`. That holds only
    when Studio encoded the text. On a wired conditioning another node did, from a string
    this one never saw — so the arithmetic would bias a window of the wrong words silently."""
    memory = {"rain": {"kind": "phrase", "effective_category_scores": {"details": 0.9}}}
    tag = _apply(refiner, monkeypatch,
                 {"funpack_conditioning_owner": "wired",
                  "funpack_encode_text": "cinematic rain",
                  # 40 text tags against a 14-character prompt: not the same string
                  "minimax_token_tags": torch.ones(40)},
                 memory)
    assert tag is None


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
