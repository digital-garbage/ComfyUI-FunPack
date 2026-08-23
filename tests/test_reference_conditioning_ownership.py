"""A wired positive CONDITIONING owns the prompt; Studio does not re-encode it.

CLIP used to win whenever both were connected, so Studio replaced the wired conditioning
with its own text-only encode. That is not an approximation of the same tensor. The node
that built the conditioning may have shown Qwen a reference image — and Qwen3-VL is causal,
so the vision block precedes the prompt and every prompt row is a hidden state that has
already read the picture. Studio cannot reproduce that: nothing in the editor pipeline wires
`h3_references` or `source_image`, so its encode never sees an image at all.
"""
import sys
import types

import pytest
import torch

sys.path.insert(0, ".")
sys.modules.setdefault("folder_paths", types.SimpleNamespace(models_dir=""))


@pytest.fixture
def studio(monkeypatch):
    import conditioning
    node = conditioning.FunPackVideoRefinerV2.__new__(conditioning.FunPackVideoRefinerV2)
    monkeypatch.setattr(node, "_v2_encode_prompt",
                        lambda *a, **k: (torch.ones(1, 4, 8), {}, "encoded from text"),
                        raising=False)
    monkeypatch.setattr(node, "_v2_extract_conditioning",
                        lambda c: (c[0][0], dict(c[0][1])), raising=False)
    monkeypatch.setattr(node, "_v2_text_tokenizer_status", lambda: "tokenizer ok",
                        raising=False)
    return node


def _wired(seq=6, dim=8, **meta):
    return [[torch.zeros(1, seq, dim), meta]]


def test_a_reference_carrying_conditioning_survives(studio):
    """The case that motivated the change: an r2v node's conditioning reaching the sampler
    intact instead of being replaced by a text-only re-encode."""
    wired = _wired(minimax_token_tags=torch.tensor([1, 1, 0, 0, 1, 1]))
    cond, _meta, _note, owner = studio._v2_conditioning_source(object(), "a cat", wired)
    assert owner == "CONDITIONING-owned"
    assert torch.equal(cond, wired[0][0])


def test_a_text_only_conditioning_survives_too(studio):
    """Not gated on detecting a vision block. An i2v checkpoint pins its anchor as a keyframe
    rather than tokenizing it, so its conditioning carries no vision block and a detector
    would have missed it — while the node still prepared something Studio cannot rebuild."""
    wired = _wired(minimax_token_tags=torch.tensor([1, 1, 1, 1, 1, 1]))
    cond, _meta, _note, owner = studio._v2_conditioning_source(object(), "a cat", wired)
    assert owner == "CONDITIONING-owned"
    assert torch.equal(cond, wired[0][0])


def test_clip_keeps_every_other_job(studio):
    """Both stay connected: CLIP is still what encodes the negative and the references."""
    _cond, _meta, note, _owner = studio._v2_conditioning_source(object(), "a cat", _wired())
    assert "CLIP still encodes the negative and references" in note


def test_no_setting_selects_between_them(studio):
    """The wire IS the instruction. Wanting CLIP to own the prompt means not wiring a
    conditioning, so there is nothing here to configure."""
    import inspect
    params = inspect.signature(studio._v2_conditioning_source).parameters
    assert "clip_owns_prompt" not in params
    assert "prefer_wired" not in params


def test_no_conditioning_wired_still_uses_clip(studio):
    cond, _meta, _note, owner = studio._v2_conditioning_source(object(), "a cat", None)
    assert owner == "CLIP-owned"
    assert torch.equal(cond, torch.ones(1, 4, 8))


def test_an_invalid_wired_conditioning_falls_back_to_clip(studio, monkeypatch, capsys):
    """Falling through in silence would look like the wire did nothing."""
    monkeypatch.setattr(studio, "_v2_extract_conditioning",
                        lambda c: ("not a tensor", {}), raising=False)
    cond, _meta, _note, owner = studio._v2_conditioning_source(object(), "a cat", _wired())
    assert owner == "CLIP-owned"
    assert torch.equal(cond, torch.ones(1, 4, 8))
    assert "invalid" in capsys.readouterr().out
