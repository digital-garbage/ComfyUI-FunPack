"""A learned conditioning survives a prompt whose length has changed.

`_v2_shape_compatible` checks the channel width and the batch, deliberately not the sequence
length. On LTX that was fine — the encoder pads to a fixed length, so a stored conditioning
always matched. Qwen does not pad (`pad_to_max_length=False`), so on H3 the length changes
with every prompt edit: a direction learned at 499 positions met a conditioning of 492 and
the blend threw, taking the whole learned direction down with it.

`_resize_conditioning_sequence_like` was already in the file for exactly this, with no
callers.
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
    monkeypatch.setattr(conditioning, "_log",
                        types.SimpleNamespace(failed=lambda *a, **k: None,
                                              note_on_change=lambda *a, **k: None, feature=lambda *a, **k: None),
                        raising=False)
    return conditioning.FunPackVideoRefinerV2.__new__(conditioning.FunPackVideoRefinerV2)


def _payload(seq, dim=8, value=1.0):
    import conditioning
    return conditioning.tensor_to_serializable(torch.full((1, seq, dim), value))


def test_a_payload_from_a_longer_prompt_still_blends(studio):
    """499 -> 492 is one prompt edit. It used to throw and lose the direction entirely."""
    mixed = torch.zeros(1, 492, 8)
    out = studio._v2_apply_conditioning_payload(mixed, _payload(499), 0.5)
    assert out.shape == mixed.shape
    assert torch.allclose(out, torch.full_like(mixed, 0.5))


def test_a_payload_from_a_shorter_prompt_still_blends(studio):
    mixed = torch.zeros(1, 528, 8)
    out = studio._v2_apply_conditioning_payload(mixed, _payload(492), 0.5)
    assert out.shape == mixed.shape
    assert torch.allclose(out, torch.full_like(mixed, 0.5))


def test_repel_survives_the_same_mismatch(studio):
    mixed = torch.zeros(1, 492, 8)
    out = studio._v2_repel_conditioning_payload(mixed, _payload(528, value=1.0), 0.5)
    assert out.shape == mixed.shape
    assert torch.allclose(out, torch.full_like(mixed, -0.5))


def test_an_exact_match_is_not_resampled(studio):
    """The common case — same prompt, re-rolled — must stay bit-exact."""
    mixed = torch.zeros(1, 64, 8)
    out = studio._v2_apply_conditioning_payload(mixed, _payload(64, value=2.0), 1.0)
    assert torch.equal(out, torch.full_like(mixed, 2.0))


def test_a_different_channel_width_is_still_refused(studio):
    """Resampling is along the SEQUENCE. A different model's conditioning is not blendable."""
    mixed = torch.zeros(1, 64, 8)
    out = studio._v2_apply_conditioning_payload(mixed, _payload(64, dim=16), 0.5)
    assert torch.equal(out, mixed)


def test_a_broken_payload_leaves_the_conditioning_alone(studio):
    mixed = torch.zeros(1, 64, 8)
    assert torch.equal(studio._v2_apply_conditioning_payload(mixed, {"shape": [1, 64, 8]}, 0.5),
                       mixed)


def test_the_liked_blend_survives_a_prompt_whose_length_changed(monkeypatch):
    """The Absolute/liked pull was the one payload path still doing a raw lerp. On H3 the
    encoder does not pad, so a direction learned at 586 positions met a 494-position prompt
    and the blend threw — silently dropping the pull toward what the user rated well."""
    import torch
    from conditioning import FunPackVideoRefinerV2, tensor_to_serializable

    refiner = FunPackVideoRefinerV2()
    mixed = torch.zeros(1, 494, 16)
    liked = torch.ones(1, 586, 16)
    out = refiner._v2_payload_like(tensor_to_serializable(liked), mixed)

    assert out is not None
    assert list(out.shape) == [1, 494, 16]
    assert float(mixed.lerp(out, 0.5).mean()) > 0.0
