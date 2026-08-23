"""A reference image's encoded rows are not steerable.

MiniMax H3 tokenizes a reference INTO the conditioning: comfy's
`token_tags_from_embeds_info` marks the whole vision block — plus its flanking
<|vision_start|>/<|vision_end|> — as 0 and leaves text at 1. Those rows are Qwen's encoding
of the picture.

Every steering path in Studio (learned directions, taste pull, concept deltas, the absolute
store, negative_erase) is a lerp or an add over the WHOLE tensor. Applied to the vision rows
it moves the encoded picture toward a direction learned from TEXT, and the character comes
out looking like someone else. Nothing read the tag map before this.
"""
import sys
import types

import pytest
import torch

sys.path.insert(0, ".")
sys.modules.setdefault("folder_paths", types.SimpleNamespace(models_dir=""))


@pytest.fixture
def studio():
    import conditioning
    return conditioning.FunPackVideoRefinerV2.__new__(conditioning.FunPackVideoRefinerV2)


# The real r2v layout: "<Picture 1>: " label (text), the vision block, then the prompt.
TAGS = torch.tensor([1, 0, 0, 0, 1, 1])


def _entry(studio, value=1.0):
    cond = torch.full((1, 6, 4), value)
    cond[:, :4] = 7.0                             # label + vision block = the reference
    meta = studio._v2_stash_reference_rows(cond, {"minimax_token_tags": TAGS})
    return cond, meta


def test_everything_before_the_prompt_is_put_back_after_steering(studio):
    cond, meta = _entry(studio)
    steered = torch.zeros_like(cond)              # as if every row had been lerped away

    out = studio._v2_restore_reference_rows([[steered, meta]])

    assert torch.equal(out[0][0][:, :4], torch.full((1, 4, 4), 7.0))


def test_the_prompt_rows_keep_whatever_steering_did(studio):
    """Protection is for the reference, not a veto on refinement. The prompt is still
    Studio's to steer — that is the whole job."""
    cond, meta = _entry(studio)
    steered = torch.zeros_like(cond)

    out = studio._v2_restore_reference_rows([[steered, meta]])

    assert torch.equal(out[0][0][:, 4:], torch.zeros(1, 2, 4))


def test_the_picture_s_label_is_protected_too(studio):
    """The `<Picture 1>: ` label sits between the two and is not part of anything Studio
    wrote. Protecting only the image-tagged rows would leave it steerable."""
    cond, meta = _entry(studio)
    out = studio._v2_restore_reference_rows([[torch.zeros_like(cond), meta]])
    assert torch.equal(out[0][0][:, 0], torch.full((1, 4), 7.0))


def test_the_stash_never_travels_on_to_the_sampler(studio):
    cond, meta = _entry(studio)
    out = studio._v2_restore_reference_rows([[cond, meta]])
    assert studio.REFERENCE_ROWS_KEY not in out[0][1]


def test_a_text_only_conditioning_is_not_stashed_at_all(studio):
    """The prompt starts at row 0 and there is nothing in front of it."""
    cond = torch.ones(1, 6, 4)
    meta = studio._v2_stash_reference_rows(cond, {"minimax_token_tags": torch.ones(6)})
    assert studio.REFERENCE_ROWS_KEY not in meta


def test_a_non_h3_conditioning_is_untouched(studio):
    meta = studio._v2_stash_reference_rows(torch.ones(1, 6, 4), {"pooled_output": None})
    assert studio.REFERENCE_ROWS_KEY not in meta


def test_an_entry_without_a_stash_passes_straight_through(studio):
    entry = [torch.ones(1, 6, 4), {"pooled_output": None}]
    assert studio._v2_restore_reference_rows([entry])[0] is entry


def test_a_conditioning_resized_after_the_stash_does_not_crash(studio):
    """The scene split and the tag trimmer both change length under this."""
    cond, meta = _entry(studio)
    out = studio._v2_restore_reference_rows([[torch.zeros(1, 2, 4), meta]])
    assert out[0][0].shape == (1, 2, 4)
    assert studio.REFERENCE_ROWS_KEY not in out[0][1]


def test_the_stash_is_a_copy_not_a_view(studio):
    """Steering happens in place in places; a view would record the damage it is meant to
    undo."""
    cond, meta = _entry(studio)
    cond.zero_()
    out = studio._v2_restore_reference_rows([[cond, meta]])
    assert torch.equal(out[0][0][:, :4], torch.full((1, 4, 4), 7.0))
