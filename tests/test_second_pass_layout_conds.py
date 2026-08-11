"""Guide keyframes must not survive a resolution-changing second_pass_op.

Guide keyframes are recorded as TOKEN indices into pass 1's latent grid. `upscale_2x`
doubles each spatial dim, so a latent frame is suddenly 4x as many tokens and those indices
address the wrong ones. Since the LTX-2.5 commit core raises on it outright ("keyframe_idxs
holds N tokens, which is not a whole number of M-token latent frames"); before that it
silently mis-placed the guides, which is worse. Pass 2 therefore drops them — and only
pass 2, because pass 1 ran at the grid they were recorded against.
"""
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import detailing  # noqa: E402
import samplers  # noqa: E402


def _cond(**meta):
    return [[torch.zeros(1, 4, 128), dict(meta)]]


def _latent(h, w, frames=3):
    return {"samples": torch.zeros(1, 128, frames, h, w)}


def _chain():
    return samplers.FunPackLTXAVSceneChainSampler()


# --- the shared predicate ------------------------------------------------------------------

def test_spatial_change_is_detected():
    chain = _chain()
    assert chain._latent_spatial_changed(_latent(32, 48), _latent(16, 24)) is True


def test_same_size_is_not_a_change():
    chain = _chain()
    assert chain._latent_spatial_changed(_latent(16, 24), _latent(16, 24)) is False


def test_more_frames_alone_is_not_a_spatial_change():
    """Only H/W matter: token-per-frame count is spatial. A differing frame count must not
    be read as a resize, or 'sharpen' runs would drop guides for no reason."""
    chain = _chain()
    assert chain._latent_spatial_changed(_latent(16, 24, frames=5), _latent(16, 24, frames=3)) is False


def test_unreadable_shapes_report_no_change():
    """A probe failure must not cost the user their guides."""
    chain = _chain()
    assert chain._latent_spatial_changed({"samples": "not a tensor"}, _latent(16, 24)) is False


# --- the stripper -------------------------------------------------------------------------

def test_layout_keys_are_detected_and_stripped():
    conds = _cond(keyframe_idxs=torch.zeros(1, 3, 512), pooled_output=None)
    assert detailing.has_layout_conds(conds) is True
    stripped = detailing.strip_layout_conds(conds)
    assert "keyframe_idxs" not in stripped[0][1]
    assert "pooled_output" in stripped[0][1], "only layout keys go; text conditioning stays"


def test_every_layout_key_is_covered():
    conds = _cond(**{k: 1 for k in detailing._LAYOUT_COND_KEYS})
    assert detailing.strip_layout_conds(conds)[0][1] == {}


def test_plain_text_conditioning_has_no_layout_keys():
    """The common case: no guides, so pass 2 must be told nothing was dropped."""
    assert detailing.has_layout_conds(_cond(pooled_output=None)) is False


def test_stripping_does_not_mutate_the_input():
    """Pass 1's conditioning is the same object; mutating it would retroactively strip the
    pass that legitimately used the guides."""
    conds = _cond(keyframe_idxs=torch.zeros(1, 3, 512))
    detailing.strip_layout_conds(conds)
    assert "keyframe_idxs" in conds[0][1]


def test_tensor_identity_is_preserved():
    """Only the meta dict is rebuilt — the conditioning tensor must not be copied."""
    conds = _cond(keyframe_idxs=1)
    assert detailing.strip_layout_conds(conds)[0][0] is conds[0][0]


def test_non_entry_items_pass_through_untouched():
    assert detailing.strip_layout_conds([None, "junk"]) == [None, "junk"]
    assert detailing.has_layout_conds([None, "junk"]) is False
