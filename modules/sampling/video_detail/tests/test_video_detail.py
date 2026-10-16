"""Video detail on a real H3 forward: picture changes, sound is bit-identical."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def _load(tiny, patcher=None, **values):
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    return FunPackLoadModifiers.execute(
        patcher or tiny.patcher, {"video_detail": {"enabled": True, **values}}).result


def _run_patched(tiny, patched, **run):
    """Object patches apply at load; do what ComfyUI's load does."""
    patched.patch_model()
    try:
        return tiny.run(patched, **run)
    finally:
        patched.unpatch_model()


def test_picture_moves_and_sound_does_not(tiny_h3):
    base = tiny_h3.run()
    patched, status = _load(tiny_h3, amount=1.5)
    assert "video_detail: 1.5x on the picture" in status
    out = _run_patched(tiny_h3, patched)
    assert not torch.allclose(out[0], base[0])
    assert torch.equal(out[1], base[1]), "the sound changed"


def test_both_directions_move_the_picture(tiny_h3):
    base = tiny_h3.run()[0]
    mags = []
    for amount in (0.5, 1.5):
        patched, _ = _load(tiny_h3, amount=amount)
        mags.append(float((_run_patched(tiny_h3, patched)[0] - base).abs().mean()))
    assert all(m > 0 for m in mags)


def test_switched_off_on_a_second_pass_the_patch_is_gone(tiny_h3):
    once, _ = _load(tiny_h3, amount=1.5)
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    off, _ = FunPackLoadModifiers.execute(once, {"video_detail": {"enabled": False}}).result
    assert not off.object_patches, "a feature switched off kept its patch"
    assert torch.allclose(_run_patched(tiny_h3, off)[0], tiny_h3.run()[0])


def test_the_norm_is_left_as_it_was(tiny_h3):
    patched, _ = _load(tiny_h3, amount=1.5)
    _run_patched(tiny_h3, patched)
    assert "forward" not in tiny_h3.patcher.model.diffusion_model.final_layer.norm.__dict__
