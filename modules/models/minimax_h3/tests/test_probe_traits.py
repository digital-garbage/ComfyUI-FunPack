"""probe_traits(): what traits() would answer for a loaded H3 model, read
from a checkpoint's tensor names alone.

Cross-checked against the real comfy.latent_formats.MiniMaxH3AV class rather
than hand-copied numbers, so this stays correct if upstream ever changes
those constants without anyone remembering to update a second copy here.
"""

import pytest


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """probe_traits() imports comfy.latent_formats; detect() does not."""


def _module():
    from modules.models import minimax_h3
    return minimax_h3


def _h3_keys():
    return {"video_patch_proj.weight", "audio_patch_proj.weight"}


def test_an_undetected_file_gets_no_traits():
    m = _module()
    assert m.probe_traits({"unrelated.weight"}) == []


def test_an_empty_key_set_gets_no_traits():
    m = _module()
    assert m.probe_traits(set()) == []


def test_a_detected_file_always_gets_audio_stream():
    m = _module()
    assert "audio_stream" in m.probe_traits(_h3_keys())


def test_traits_match_the_real_latent_format_class():
    """Pinned against comfy's own class, not a hardcoded string -- if
    upstream ever made H3's latent 2-D or dropped temporal compression, this
    test (not just probe_traits) would need to change too."""
    from comfy.latent_formats import MiniMaxH3AV
    from core import traits as core_traits

    m = _module()
    found = m.probe_traits(_h3_keys())

    expected_rank = core_traits.LATENT_RANK.get(MiniMaxH3AV.latent_dimensions)
    assert expected_rank in found

    if getattr(MiniMaxH3AV, "temporal_downscale_ratio", 1) not in (None, 1):
        assert "temporal_compression" in found


def test_does_not_claim_adaln_modalities_without_a_real_load():
    """The one trait probe_traits deliberately does not answer -- full-form
    vs. curve-form checkpoints only differ in which blocks are actually
    present, which the header's key NAMES alone (as read here) do not
    distinguish reliably enough to assert either way."""
    m = _module()
    assert "adaln_modalities" not in m.probe_traits(_h3_keys())
