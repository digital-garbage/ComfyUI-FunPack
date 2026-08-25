"""Classifying N uncertain phrases costs one encode, not N.

A phrase's vector used to come from encoding that phrase alone — a full pass of the text
encoder each, and on H3 that encoder is Qwen3-VL-32B. The phrases are encoded together now
and each one's vector is the mean of the rows its own tokens occupy, which the tokenizer's
offset mapping identifies exactly.
"""
import sys
import types

import pytest
import torch

sys.path.insert(0, ".")


class _Tok:
    """Offset mapping over whitespace/comma-free spans, like a real subword tokenizer."""
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        offsets, i = [], 0
        for chunk in text.split(" "):
            if chunk:
                offsets.append((i, i + len(chunk)))
            i += len(chunk) + 1
        return {"offset_mapping": offsets}


@pytest.fixture
def refiner(monkeypatch):
    import conditioning as C
    r = C.FunPackVideoRefinerV2()
    r._v2_reset_encode_tally()
    return r


@pytest.fixture
def wired(monkeypatch, refiner):
    """One encode returns a conditioning with one row per token of the joined text."""
    import conditioning as C
    import h3_token_weights as tw
    calls = {"n": 0}

    def fake_encode(self, clip, text, encode_cache=None, purpose="prompt", **kw):
        calls["n"] += 1
        n = len(_Tok()(text)["offset_mapping"])
        rows = torch.arange(n, dtype=torch.float32).unsqueeze(-1).repeat(1, 4)
        return rows.unsqueeze(0), {"minimax_token_tags": [1] * n}, ""

    monkeypatch.setattr(C.FunPackVideoRefinerV2, "_v2_encode_prompt", fake_encode)
    monkeypatch.setattr(tw, "h3_tokenizer", lambda clip: _Tok())
    return refiner, calls


def test_many_phrases_cost_one_encode(wired):
    refiner, calls = wired
    out = refiner._v2_phrase_vectors_in_one_pass(object(), ["alpha", "beta", "gamma"])
    assert out is not None and set(out) == {"alpha", "beta", "gamma"}
    assert calls["n"] == 1


def test_each_phrase_gets_its_own_rows(wired):
    """The rows are numbered, so the pooled value says which span was read."""
    refiner, _ = wired
    out = refiner._v2_phrase_vectors_in_one_pass(object(), ["alpha", "beta"])
    # joined = "alpha, beta" -> tokens ["alpha,", "beta"] at rows 0 and 1
    assert float(out["alpha"][0]) == 0.0
    assert float(out["beta"][0]) == 1.0


def test_a_phrase_inside_another_is_not_given_its_rows(wired):
    """'dress' is a substring of 'red dress'. Searching the text would hand both the same
    rows; the positions come from the join instead."""
    refiner, _ = wired
    out = refiner._v2_phrase_vectors_in_one_pass(object(), ["red dress", "dress"])
    assert not torch.equal(out["red dress"], out["dress"])


def test_one_phrase_is_not_worth_batching(wired):
    refiner, calls = wired
    assert refiner._v2_phrase_vectors_in_one_pass(object(), ["alpha"]) is None
    assert calls["n"] == 0


def test_no_offsets_means_fall_back(monkeypatch, refiner):
    import h3_token_weights as tw
    monkeypatch.setattr(tw, "h3_tokenizer", lambda clip: None)
    assert refiner._v2_phrase_vectors_in_one_pass(object(), ["a", "b"]) is None


def test_both_paths_score_a_vector_the_same_way(refiner):
    """The cosine step is shared, so a phrase scores the same however its vector was got."""
    cats = {"action": torch.tensor([1.0, 0.0]), "style": torch.tensor([0.0, 1.0])}
    scores = refiner._v2_scores_against_categories(torch.tensor([1.0, 0.0]), cats)
    assert scores["action"] > scores["style"]
    assert refiner._v2_scores_against_categories(None, cats) == {}
