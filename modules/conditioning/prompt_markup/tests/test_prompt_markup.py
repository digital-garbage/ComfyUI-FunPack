"""Prompt markup, with H3's REAL tokenizer (offline) and the tiny real H3 model.

The encoder is a stand-in -- one fixed vector per token id -- because Qwen3-VL-32B
is not a unit-test dependency; placement depends on the tokenizer and the tags,
which are real here.
"""

import math

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


class FakeH3Clip:
    def __init__(self):
        from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
        self.tokenizer = MiniMaxH3Tokenizer(embedding_directory=None, tokenizer_data={})

    def tokenize(self, text):
        return self.tokenizer.tokenize_with_weights(text)

    def encode_from_tokens_scheduled(self, tokens):
        ids = [t[0] for t in tokens["qwen3vl_32b"][0]]
        rows = torch.stack([torch.randn(48, generator=torch.Generator().manual_seed(int(i)))
                            for i in ids]).unsqueeze(0)
        return [[rows, {"minimax_token_tags": torch.ones(len(ids), dtype=torch.long)}]]


def _encode(clip, text):
    return clip.encode_from_tokens_scheduled(clip.tokenize(text))


def _apply(tiny, raw):
    from modules.conditioning.prompt_markup.nodes import (FunPackApplyPromptMarkup,
                                                          FunPackPromptMarkup)
    clean, markup, _ = FunPackPromptMarkup.execute(raw).result
    clip = FakeH3Clip()
    positive = _encode(clip, clean)
    tiny.context = positive[0][0]
    model, pos, status = FunPackApplyPromptMarkup.execute(
        tiny.patcher, clip, positive, {"samples": tiny.video}, markup).result
    return model, pos, status, positive


def _spy_masks(tiny, model):
    """Every (q rows, mask) the backend receives, via an override UNDER ours."""
    from core import dit_hooks
    seen = []

    def spy(func, q, k, v, *a, mask=None, **kw):
        seen.append((q.shape[2], mask))
        return func(q, k, v, *a, mask=mask, **kw)

    under = tiny.patcher.clone()
    dit_hooks.add_attention_override(under, "spy", spy)
    ours = model.model_options["transformer_options"]["optimized_attention_override"]
    # Put ours on top of the spy.
    under.model_options["transformer_options"]["optimized_attention_override"] = (
        lambda func, *a, **kw: ours(lambda *x, **y: under_spy(func, *x, **y), *a, **kw))
    under_spy = lambda func, *a, **kw: spy(func, *a, **kw)      # noqa: E731
    tiny.run(under)
    return seen


def test_the_node_strips_markup_and_reports_it():
    from modules.conditioning.prompt_markup.nodes import FunPackPromptMarkup
    clean, markup, status = FunPackPromptMarkup.execute("a (red:1.5) car [sits|runs]").result
    assert clean == "a red car sits"
    assert status == "1 weighted, 1 blended"


def test_a_weight_biases_exactly_that_words_keys(tiny_h3):
    model, _p, status, _pos = _apply(tiny_h3, "a (red:1.5) car drives")
    assert status == "1 weight(s)"
    masks = [m for n, m in _spy_masks(tiny_h3, model) if m is not None]
    assert masks, "the bias never reached attention"
    row = masks[0].reshape(-1)
    # tokens: "a" " red" " car" " drives" -> "red" is token 1, prompt at row 0
    assert row[1].item() == pytest.approx(math.log(1.5), rel=1e-4)
    assert torch.count_nonzero(row).item() == 1


def test_it_changes_the_output_and_weight_one_does_not(tiny_h3):
    base_model, _p, _s, pos = _apply(tiny_h3, "a red car drives")
    base = tiny_h3.run(base_model)
    weighted, *_ = _apply(tiny_h3, "a (red:3) car drives")
    assert not torch.allclose(tiny_h3.run(weighted)[0], base[0])


def test_a_timed_phrase_is_masked_from_video_outside_its_window(tiny_h3):
    # Latent frame 0 = pixel frame 0, frame 1 = pixel frames 1-4 at 24 fps;
    # 0.05-0.2 s touches only latent frame 1 = video rows 4..8 of 8.
    model, _p, status, _pos = _apply(tiny_h3, "a red [car@0.05-0.2] drives")
    assert status == "1 timed phrase(s)"
    calls = _spy_masks(tiny_h3, model)
    # The text-only token refiner runs first on exactly the prompt: untouched.
    n, m = calls[0]
    assert n == int(tiny_h3.context.shape[1]) and m is None, "the refiner saw the window"
    per_block = [(n, m) for n, m in calls if m is not None][:3]   # block 0's chunks
    assert [n for n, _m in per_block] == [10, 4, 4]               # text+audio | video out | in
    assert sum(n for n, _m in per_block) == tiny_h3_seq(tiny_h3)
    key = [m.reshape(-1)[2].item() for _n, m in per_block]        # "car" is token 2
    assert key[0] == 0.0, "text/audio rows lost the phrase"
    assert key[1] == pytest.approx(-30.0), "video outside the window still sees it"
    assert key[2] == 0.0, "video inside the window lost it"


def tiny_h3_seq(tiny):
    video_rows = 2 * 2 * 2
    return int(tiny.context.shape[1]) + video_rows + 2 * 3


def test_a_blend_moves_only_the_prompt_rows_and_zero_strength_moves_nothing(tiny_h3):
    _m, pos, status, original = _apply(tiny_h3, "a red car [drives|flies]")
    assert status == "1 blend(s)"
    assert not torch.allclose(pos[0][0], original[0][0])
    _m, pos0, _s, original0 = _apply(tiny_h3, "a red car [drives|flies:0]")
    assert torch.allclose(pos0[0][0], original0[0][0])


def test_an_encoder_no_model_module_knows_is_refused_out_loud(tiny_h3):
    from modules.conditioning.prompt_markup.nodes import (FunPackApplyPromptMarkup,
                                                          FunPackPromptMarkup)
    _clean, markup, _ = FunPackPromptMarkup.execute("a (red:2) car").result

    class OtherClip:
        tokenizer = object()

    positive = [[torch.zeros(1, 4, 48), {}]]
    model, pos, status = FunPackApplyPromptMarkup.execute(
        tiny_h3.patcher, OtherClip(), positive, {"samples": tiny_h3.video}, markup).result
    assert status.startswith("not applied") and pos is positive


def test_the_phrases_it_placed_are_published_for_anything_that_wants_to_look_at_them(tiny_h3):
    from modules.conditioning.prompt_markup.nodes import PHRASES
    model, _p, _s, pos = _apply(tiny_h3, "a (red:1.5) car [drives@0.05-0.2]")
    published = model.model_options["transformer_options"][PHRASES]
    assert published["cond_len"] == int(pos[0][0].shape[1])
    assert [b - a for a, b in published["spans"]] == [1, 1]          # "red" and "drives": one token each
    plain, *_ = _apply(tiny_h3, "a red car drives")
    assert PHRASES not in plain.model_options.get("transformer_options", {})
