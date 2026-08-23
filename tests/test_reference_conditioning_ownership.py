"""A conditioning that saw a reference image must not be re-encoded from text.

Qwen3-VL is a causal VLM and the H3 tokenizer puts a reference's vision block BEFORE the
prompt, so every prompt row is a hidden state that has already read the picture. Studio's
own encode receives no image in the editor pipeline — nothing wires h3_references or
source_image — so replacing the wired conditioning does not approximate it, it discards the
reference and hands an i2v/r2v sampler a text-only conditioning.
"""
import sys
import types

import pytest
import torch

sys.path.insert(0, ".")


@pytest.fixture
def studio():
    import conditioning
    obj = conditioning.FunPackVideoRefinerV2.__new__(conditioning.FunPackVideoRefinerV2)
    return obj


def _cond(tags=None, seq=6, dim=8):
    meta = {}
    if tags is not None:
        meta["minimax_token_tags"] = tags
    return [[torch.zeros(1, seq, dim), meta]]


def test_a_vision_block_is_detected(studio):
    """1 is text, 0 is a whole vision block."""
    assert studio._wired_conditioning_saw_a_reference(_cond(torch.tensor([1, 1, 0, 0, 1, 1])))


def test_text_only_tags_are_not_a_reference(studio):
    assert not studio._wired_conditioning_saw_a_reference(_cond(torch.tensor([1, 1, 1])))


def test_tags_as_a_plain_list_are_read(studio):
    assert studio._wired_conditioning_saw_a_reference(_cond([1, 0, 1]))


def test_no_tags_means_no_claim(studio):
    """A text-only or non-H3 graph must behave exactly as before."""
    assert not studio._wired_conditioning_saw_a_reference(_cond(None))
    assert not studio._wired_conditioning_saw_a_reference(None)
    assert not studio._wired_conditioning_saw_a_reference([])


def test_a_malformed_conditioning_is_not_a_reference(studio):
    assert not studio._wired_conditioning_saw_a_reference(["nonsense"])
    assert not studio._wired_conditioning_saw_a_reference([[torch.zeros(1, 3, 4)]])


def test_the_reference_conditioning_wins_over_clip(studio, monkeypatch):
    """The whole point: CLIP used to win here and the reference was silently discarded."""
    monkeypatch.setattr(studio, "_v2_encode_prompt",
                        lambda *a, **kw: (torch.ones(1, 3, 8), {}, "encoded from text"))
    wired = _cond(torch.tensor([1, 0, 0, 1, 1, 1]))

    cond, _meta, note, owner = studio._v2_conditioning_source(
        clip=object(), prompt_text="a cat", positive_conditioning=wired)

    assert owner == "CONDITIONING-owned"
    assert torch.equal(cond, wired[0][0])
    assert "reference image" in note


def test_a_text_only_wired_conditioning_still_loses_to_clip(studio, monkeypatch):
    """Unchanged behaviour where there is no reference to protect."""
    encoded = torch.ones(1, 3, 8)
    monkeypatch.setattr(studio, "_v2_encode_prompt",
                        lambda *a, **kw: (encoded, {}, "encoded from text"))

    cond, _meta, _note, owner = studio._v2_conditioning_source(
        clip=object(), prompt_text="a cat",
        positive_conditioning=_cond(torch.tensor([1, 1, 1])))

    assert owner == "CLIP-owned"
    assert torch.equal(cond, encoded)


def test_no_clip_at_all_still_takes_the_wired_one(studio):
    wired = _cond(torch.tensor([1, 0, 1]))
    cond, _meta, _note, owner = studio._v2_conditioning_source(
        clip=None, prompt_text="a cat", positive_conditioning=wired)
    assert owner == "CONDITIONING-owned"
    assert torch.equal(cond, wired[0][0])
