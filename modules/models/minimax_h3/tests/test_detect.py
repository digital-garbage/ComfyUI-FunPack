"""detect(): whether a set of tensor names is MiniMax H3's own.

Pinned against ComfyUI's real prefix-resolution algorithm
(comfy/model_detection.py's unet_prefix_from_state_dict, and what
comfy/sd.py's load_diffusion_model_state_dict does with its answer) rather
than only against this module's own idea of what a "reasonable" prefix
looks like -- a probe that says yes to a file ComfyUI's real loader would
refuse is worse than one that says "not detected", which is exactly the
failure a prior review round found and this file exists to pin shut.
"""

import pytest


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Only needed to import the module under test, which imports comfy.* at
    module level -- detect() itself touches no comfy API."""


def _module():
    from modules.models import minimax_h3
    return minimax_h3


def _keys(prefix, *, both=True):
    names = ["video_patch_proj.weight", "audio_patch_proj.weight"]
    if not both:
        names = names[:1]
    return {f"{prefix}{n}" for n in names}


def test_bare_keys_are_recognised():
    """A standalone unet-only file: ComfyUI's own prefix guess strips
    nothing for this layout (no key starts with "model."), so the ORIGINAL
    unprefixed keys are what its real loader ends up reading."""
    m = _module()
    assert m.detect(_keys(""))


def test_model_diffusion_model_prefix_is_recognised():
    """The first, and only H3-relevant, candidate
    unet_prefix_from_state_dict actually tries for a full-checkpoint
    layout."""
    m = _module()
    assert m.detect(_keys("model.diffusion_model."))


def test_bare_diffusion_model_prefix_is_not_recognised():
    """Found by adversarial review: this prefix is not one of the three real
    candidates ComfyUI's own unet_prefix_from_state_dict tries, and tracing
    what its fallback guess ("model.") does with a state dict keyed this way
    shows ComfyUI's real loader would never find these tensors either -- so
    this probe must not claim it can."""
    m = _module()
    assert not m.detect(_keys("diffusion_model."))


def test_only_one_of_the_two_signature_keys_is_not_enough():
    m = _module()
    assert not m.detect(_keys("", both=False))
    assert not m.detect(_keys("model.diffusion_model.", both=False))


def test_the_two_signature_keys_split_across_different_prefixes_do_not_match():
    """One tensor under one prefix, the other under a different one -- not a
    real file's shape, and must not be read as one by accident."""
    m = _module()
    mixed = {"video_patch_proj.weight",
             "model.diffusion_model.audio_patch_proj.weight"}
    assert not m.detect(mixed)


def test_unrelated_keys_are_not_recognised():
    m = _module()
    assert not m.detect({"model.diffusion_model.unrelated.weight", "vae.decoder.weight"})


def test_an_empty_key_set_is_not_recognised():
    m = _module()
    assert not m.detect(set())
