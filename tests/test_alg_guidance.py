"""Unit tests for the experimental ALG (Adaptive Low-Pass Guidance, arXiv:2506.08456)
i2v anchor de-staticking helper used by sample_funpack_distilled_flow.

Covers: frame-0-only blurring of the packed video stream, shape/identity preservation
of every other region (later video frames + audio), and the no-op fallbacks when there's
no usable latent_image or the packed layout can't be read.
"""
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Minimal comfy stubs so `import samplers` works without a full ComfyUI env.
for _name in (
    "comfy", "comfy.k_diffusion", "comfy.k_diffusion.sampling",
    "comfy.model_sampling", "comfy.nested_tensor", "comfy.sample",
    "comfy.samplers", "comfy.utils",
):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["comfy.nested_tensor"].NestedTensor = object

import samplers  # noqa: E402

B, C, T, H, W = 1, 2, 2, 4, 4
VIDEO_SHAPE = (B, C, T, H, W)
AUDIO_SHAPE = (B, 6)
VIDEO_SIZE = C * T * H * W  # 64
AUDIO_SIZE = 6
N = VIDEO_SIZE + AUDIO_SIZE


def _fake_model(shapes=(VIDEO_SHAPE, AUDIO_SHAPE)):
    return types.SimpleNamespace(inner_model=types.SimpleNamespace(conds={
        "positive": [{"model_conds": {"latent_shapes": types.SimpleNamespace(cond=list(shapes))}}],
    }))


def _packed_latent_image():
    video = torch.arange(VIDEO_SIZE, dtype=torch.float32).reshape(1, 1, VIDEO_SIZE)
    # Make frame 0 a sharp checkerboard so blurring visibly changes it.
    checker = torch.tensor([[(i + j) % 2 for j in range(W)] for i in range(H)], dtype=torch.float32)
    video = video.reshape(B, C, T, H, W)
    video[:, :, 0] = checker
    audio = torch.full((B, 1, AUDIO_SIZE), 5.0)
    return torch.cat([video.reshape(B, 1, VIDEO_SIZE), audio], dim=-1)


def test_alg_blur_returns_none_without_latent_image():
    model = _fake_model()
    assert samplers._alg_blur_video_frame0(model, None, 2.5) is None


def test_alg_blur_returns_none_without_latent_shapes():
    model = types.SimpleNamespace(inner_model=types.SimpleNamespace(conds={}))
    latent_image = _packed_latent_image()
    assert samplers._alg_blur_video_frame0(model, latent_image, 2.5) is None


def test_alg_blur_returns_none_on_layout_mismatch():
    model = _fake_model()
    wrong_size_latent = torch.zeros(1, 1, N + 1)
    assert samplers._alg_blur_video_frame0(model, wrong_size_latent, 2.5) is None


def test_alg_blur_only_touches_frame_zero():
    model = _fake_model()
    latent_image = _packed_latent_image()
    out = samplers._alg_blur_video_frame0(model, latent_image, 2.5)
    assert out is not None
    assert out.shape == latent_image.shape

    video_out = out[..., :VIDEO_SIZE].reshape(B, C, T, H, W)
    video_in = latent_image[..., :VIDEO_SIZE].reshape(B, C, T, H, W)

    # Frame 0 changed (blurred away from the sharp checkerboard).
    assert not torch.allclose(video_out[:, :, 0], video_in[:, :, 0])
    # Every later video frame is untouched.
    assert torch.equal(video_out[:, :, 1:], video_in[:, :, 1:])
    # Audio region is byte-identical — ALG never touches audio.
    assert torch.equal(out[..., VIDEO_SIZE:], latent_image[..., VIDEO_SIZE:])


def test_alg_blur_reduces_high_frequency_variance():
    model = _fake_model()
    latent_image = _packed_latent_image()
    out = samplers._alg_blur_video_frame0(model, latent_image, 2.5)
    frame0_in = latent_image[..., :VIDEO_SIZE].reshape(B, C, T, H, W)[:, :, 0]
    frame0_out = out[..., :VIDEO_SIZE].reshape(B, C, T, H, W)[:, :, 0]
    # A checkerboard has maximal local variance; the down+up-sampled copy should not.
    assert float(frame0_out.var()) < float(frame0_in.var())


def test_alg_blur_is_deterministic():
    model = _fake_model()
    latent_image = _packed_latent_image()
    out1 = samplers._alg_blur_video_frame0(model, latent_image, 2.5)
    out2 = samplers._alg_blur_video_frame0(model, latent_image, 2.5)
    assert torch.equal(out1, out2)
