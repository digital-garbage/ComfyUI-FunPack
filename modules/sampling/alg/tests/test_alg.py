"""ALG, as far as a machine with no GPU can take it.

What IS checked here: the blur does what a low-pass does, the per-step choice
follows sigma, the inert cases are inert, and the whole thing runs through
ComfyUI's real KSAMPLER with a stand-in sampler function -- so the anchor a
denoiser sees really does change between steps.

What is NOT checked here, and cannot be: whether the result looks better. That
needs a GPU and real weights.
"""

import pytest


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy and torch."""


def _latent(frames=4, size=16, seed=0):
    import torch
    torch.manual_seed(seed)
    return torch.randn(1, 3, frames, size, size)


# --- the filter ------------------------------------------------------------

def test_blurring_removes_high_frequency_detail():
    """A low-pass, stated as something measurable: neighbouring pixels of the
    blurred frame differ less than they did."""
    import torch
    from modules.sampling.alg.blur import blur_frames

    sharp = _latent()
    out = blur_frames(sharp, kappa=4.0, frame_indices=(0,))

    def roughness(frame):
        return (frame[..., 1:] - frame[..., :-1]).abs().mean()

    assert roughness(out[:, :, 0]) < roughness(sharp[:, :, 0])


def test_only_the_named_frames_change():
    """The anchor is frame 0. Blurring the whole clip would remove the detail
    every other frame is supposed to resolve."""
    import torch
    from modules.sampling.alg.blur import blur_frames

    sharp = _latent(frames=4)
    out = blur_frames(sharp, kappa=4.0, frame_indices=(0,))

    assert not torch.equal(out[:, :, 0], sharp[:, :, 0])
    for index in (1, 2, 3):
        assert torch.equal(out[:, :, index], sharp[:, :, index]), f"frame {index} changed"


def test_the_original_is_not_modified():
    """The sharp anchor is swapped back in later, so it has to survive."""
    import torch
    from modules.sampling.alg.blur import blur_frames

    sharp = _latent()
    keep = sharp.clone()
    blur_frames(sharp, kappa=4.0)
    assert torch.equal(sharp, keep)


def test_shape_and_dtype_are_preserved():
    import torch
    from modules.sampling.alg.blur import blur_frames
    sharp = _latent().to(torch.float16)
    out = blur_frames(sharp, kappa=4.0)
    assert out.shape == sharp.shape and out.dtype == sharp.dtype


@pytest.mark.parametrize("bad", ["not a tensor", None])
def test_a_latent_it_does_not_understand_yields_none(bad):
    from modules.sampling.alg.blur import blur_frames
    assert blur_frames(bad, kappa=4.0) is None


def test_a_latent_of_the_wrong_rank_yields_none():
    """An image latent has no anchor frame to loosen. None means "not
    applicable", and the caller leaves sampling alone."""
    import torch
    from modules.sampling.alg.blur import blur_frames
    assert blur_frames(torch.randn(1, 4, 64, 64), kappa=4.0) is None


def test_a_frame_index_past_the_end_is_skipped_not_guessed():
    from modules.sampling.alg.blur import blur_frames
    assert blur_frames(_latent(frames=2), kappa=4.0, frame_indices=(9,)) is None


def test_a_strength_that_would_change_nothing_yields_none():
    """kappa 1.0 downsamples to the same size, so the "blurred" copy would be
    the sharp one and every step would swap between identical tensors."""
    from modules.sampling.alg.blur import blur_frames
    assert blur_frames(_latent(), kappa=1.0) is None


# --- the per-step choice ---------------------------------------------------

@pytest.mark.parametrize("sigma,threshold,expected", [
    (0.9, 0.6, True),      # early, high noise: loosen it
    (0.6, 0.6, False),     # exactly at the threshold: sharp
    (0.1, 0.6, False),     # late: sharp, so detail can resolve
])
def test_the_choice_follows_sigma(sigma, threshold, expected):
    from modules.sampling.alg.blur import use_blurred
    assert use_blurred(sigma, threshold) is expected


def test_a_sigma_it_cannot_read_falls_back_to_sharp():
    """Never cost a step over the guidance: unreadable means untouched."""
    from modules.sampling.alg.blur import use_blurred
    assert use_blurred("nonsense", 0.6) is False


def test_a_tensor_sigma_is_read_like_the_sampler_passes_it():
    import torch
    from modules.sampling.alg.blur import use_blurred
    assert use_blurred(torch.tensor([0.9]), 0.6) is True


# --- the proxy -------------------------------------------------------------

def test_the_proxy_repins_the_anchor_from_each_calls_own_sigma():
    """Multi-evaluation samplers call the denoiser more than once per step at
    different sigmas, and each call must get the anchor its own sigma calls for."""
    from modules.sampling.alg.blur import AnchorSwap

    class Denoiser:
        latent_image = None

        def __call__(self, x, sigma, **kwargs):
            return self.latent_image

    inner = Denoiser()
    proxy = AnchorSwap(inner, sharp="SHARP", blurred="BLURRED", threshold=0.6)

    assert proxy(None, 0.9) == "BLURRED"
    assert proxy(None, 0.2) == "SHARP"
    assert proxy(None, 0.7) == "BLURRED"


def test_the_proxy_is_transparent_for_everything_else():
    from modules.sampling.alg.blur import AnchorSwap

    class Denoiser:
        inner_model = "the real model"

        def __call__(self, x, sigma, **kwargs):
            return None

    inner = Denoiser()
    proxy = AnchorSwap(inner, "S", "B", 0.6)

    assert proxy.inner_model == "the real model"
    proxy.something = 7
    assert inner.something == 7, "an assignment did not reach the real denoiser"


def test_a_threshold_that_is_not_a_number_would_disable_it_everywhere():
    """Why the check at the load node matters, in ALG's own terms. NaN compares
    False against every sigma, so the anchor is never loosened at any step --
    and nothing about the run looks wrong."""
    from modules.sampling.alg.blur import use_blurred
    assert all(use_blurred(sigma, float("nan")) is False
               for sigma in (0.99, 0.5, 0.01))
