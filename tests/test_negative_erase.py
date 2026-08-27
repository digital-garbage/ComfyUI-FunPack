"""Using the negative prompt at CFG 1 by erasing its direction from the positive.

MiniMax H3 never evaluates the negative branch, so the negative prompt is dead weight. These
tests pin what the replacement actually does — and, more importantly, what it refuses to do:
touch the vision span, change a token that has nothing to do with the negative, or fail
silently when there is nothing to work with.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import negative_erase as ne  # noqa: E402


def _cond(t, meta=None):
    return [(t, dict(meta or {}))]


# --- the direction ---------------------------------------------------------------------


def test_direction_is_the_pooled_unit_vector():
    t = torch.tensor([[[2.0, 0.0], [4.0, 0.0]]])
    d = ne.direction(t)
    assert torch.allclose(d, torch.tensor([1.0, 0.0]))


def test_direction_pools_only_the_negatives_text_positions():
    """H3 packs [text | vision | audio] into one sequence. Pooling the vision span into the
    'negative prompt direction' would make an image the model was shown part of what gets
    removed from the prompt."""
    t = torch.tensor([[[1.0, 0.0], [0.0, 9.0]]])
    meta = {"minimax_token_tags": torch.tensor([1, 0])}
    assert torch.allclose(ne.direction(t, meta), torch.tensor([1.0, 0.0]))


def test_a_zero_negative_has_no_direction():
    assert ne.direction(torch.zeros(1, 3, 4)) is None


def test_mismatched_tags_are_ignored_rather_than_trusted():
    """Tags shorter than the conditioning mean the two disagree; selecting on them would pick
    the wrong rows. Fall back to pooling everything."""
    t = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    d = ne.direction(t, {"minimax_token_tags": torch.tensor([1])})
    assert d is not None and float(d[1]) > 0


# --- projection ------------------------------------------------------------------------


def test_project_removes_the_component_and_leaves_the_rest():
    """The whole reason projection is the default: a token orthogonal to the negative is not
    moved at all, while a token aligned with it loses that alignment."""
    u = torch.tensor([1.0, 0.0])
    pos = torch.tensor([[[3.0, 4.0], [0.0, 5.0]]])
    out = ne.erase(pos, u, 1.0, renorm=False)
    assert float(out[0, 0, 0]) == pytest.approx(0.0)   # the aligned part is gone
    assert float(out[0, 0, 1]) == pytest.approx(4.0)   # the rest is untouched
    assert torch.allclose(out[0, 1], pos[0, 1])        # orthogonal token: unmoved


def test_strength_scales_the_removal():
    u = torch.tensor([1.0, 0.0])
    out = ne.erase(torch.tensor([[[2.0, 1.0]]]), u, 0.5, renorm=False)
    assert float(out[0, 0, 0]) == pytest.approx(1.0)


def test_above_one_pushes_past_orthogonal_into_the_opposite():
    u = torch.tensor([1.0, 0.0])
    out = ne.erase(torch.tensor([[[2.0, 1.0]]]), u, 2.0, renorm=False)
    assert float(out[0, 0, 0]) == pytest.approx(-2.0)


def test_renorm_keeps_each_tokens_strength():
    """Projection always shrinks. Without this, a big strength reads partly as a quieter
    prompt rather than purely as less of the unwanted thing."""
    u = torch.tensor([1.0, 0.0])
    pos = torch.tensor([[[3.0, 4.0]]])
    out = ne.erase(pos, u, 1.0, renorm=True)
    assert float(out.norm()) == pytest.approx(5.0, abs=1e-4)
    assert float(out[0, 0, 0]) == pytest.approx(0.0, abs=1e-4)


def test_subtract_moves_every_token_by_the_same_relative_amount():
    """The blunt mode, kept for comparison: it moves a token that had nothing to do with the
    negative just as much as one that did."""
    u = torch.tensor([1.0, 0.0])
    out = ne.erase(torch.tensor([[[0.0, 1.0]]]), u, 0.5, mode="subtract", renorm=False)
    assert float(out[0, 0, 0]) == pytest.approx(-0.5)   # an orthogonal token IS moved


def test_the_vision_span_of_the_positive_is_never_touched():
    """Erasing a text direction out of the reference image's conditioning is not what a
    negative PROMPT was asked to do."""
    u = torch.tensor([1.0, 0.0])
    pos = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    meta = {"minimax_token_tags": torch.tensor([1, 0])}
    out = ne.erase(pos, u, 1.0, meta=meta, renorm=False)
    assert float(out[0, 0, 0]) == pytest.approx(0.0)    # text position: erased
    assert float(out[0, 1, 0]) == pytest.approx(1.0)    # vision position: untouched


def test_dtype_and_device_survive():
    u = torch.tensor([1.0, 0.0])
    pos = torch.tensor([[[3.0, 4.0]]], dtype=torch.float16)
    assert ne.erase(pos, u, 1.0).dtype == torch.float16


# --- the list-level entry point --------------------------------------------------------


def test_apply_returns_a_note_saying_what_it_did():
    pos = _cond(torch.tensor([[[3.0, 4.0]]]))
    neg = _cond(torch.tensor([[[1.0, 0.0]]]))
    out, note = ne.apply(pos, neg, 1.0)
    assert "negative_erase" in note and "project" in note
    assert float(out[0][0][0, 0, 0]) == pytest.approx(0.0, abs=1e-4)


def test_strength_zero_is_a_true_no_op():
    pos = _cond(torch.tensor([[[3.0, 4.0]]]))
    out, note = ne.apply(pos, _cond(torch.tensor([[[1.0, 0.0]]])), 0.0)
    assert out is pos and note == ""


def test_an_empty_negative_says_so_instead_of_going_quiet():
    """A switch that is on and doing nothing has to say which. Silence reads as 'it worked'."""
    pos = _cond(torch.tensor([[[3.0, 4.0]]]))
    out, note = ne.apply(pos, [], 1.0)
    assert out is pos and "no negative conditioning" in note


def test_a_useless_negative_says_so_too():
    pos = _cond(torch.tensor([[[3.0, 4.0]]]))
    out, note = ne.apply(pos, _cond(torch.zeros(1, 2, 2)), 1.0)
    assert out is pos and "no usable direction" in note


def test_every_scene_entry_is_treated():
    """A multi-scene prompt is a list of conditionings; erasing from only the first would
    make the negative apply to scene 1 alone."""
    pos = [(torch.tensor([[[3.0, 4.0]]]), {}), (torch.tensor([[[5.0, 12.0]]]), {})]
    out, note = ne.apply(pos, _cond(torch.tensor([[[1.0, 0.0]]])), 1.0)
    assert "2 conditioning entries" in note
    assert all(float(e[0][0, 0, 0]) == pytest.approx(0.0, abs=1e-3) for e in out)


def test_a_dimension_mismatch_leaves_the_conditioning_alone():
    pos = _cond(torch.tensor([[[3.0, 4.0, 5.0]]]))
    out, note = ne.apply(pos, _cond(torch.tensor([[[1.0, 0.0]]])), 1.0)
    assert "no conditioning entry could be modified" in note
    assert out is pos


# --- the renorm gain cap ---------------------------------------------------------------
# Restoring a token's norm after projection is fine until the projection removed almost all
# of it. Then what is left is rounding error, and "restore the original norm" means putting
# amplified noise into the conditioning at full prompt strength — which the sampler will
# happily bank into the refinement key and keep steering toward afterwards.


def test_a_token_that_WAS_the_negative_goes_quiet_instead_of_exploding():
    u = torch.tensor([1.0, 0.0])
    pos = torch.tensor([[[5.0, 0.001]]])          # almost exactly the negative's direction
    out = ne.erase(pos, u, 1.0, renorm=True)
    assert float(out.norm()) < 0.01               # not 5.0 of amplified residue
    assert float(out[0, 0, 1]) == pytest.approx(0.002, abs=1e-4)   # capped at 2x, no more


def test_an_ordinary_token_still_gets_its_norm_back():
    """The cap must not break the normal case it exists to serve."""
    u = torch.tensor([1.0, 0.0])
    out = ne.erase(torch.tensor([[[3.0, 4.0]]]), u, 1.0, renorm=True)
    assert float(out.norm()) == pytest.approx(5.0, abs=1e-4)


def test_the_gain_never_exceeds_the_cap():
    u = torch.tensor([1.0, 0.0])
    for orth in (0.001, 0.01, 0.1, 1.0, 4.0):
        pos = torch.tensor([[[5.0, orth]]])
        before = float(pos.norm())
        after = float(ne.erase(pos, u, 1.0, renorm=True).norm())
        residue = float(torch.tensor([0.0, orth]).norm())
        assert after <= residue * ne.RENORM_MAX_GAIN + 1e-4
        assert after <= before + 1e-4


def test_a_non_finite_result_is_refused_rather_than_passed_on():
    """It would not fail here — it fails deep in the model, and a run that captures
    conditioning for the refinement key would bank the bad vector first."""
    u = torch.tensor([float("nan"), 0.0])
    pos = torch.tensor([[[3.0, 4.0]]])
    assert ne.erase(pos, u, 1.0) is pos
