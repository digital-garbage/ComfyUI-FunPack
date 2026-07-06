"""Unit tests for the experimental ALG (Adaptive Low-Pass Guidance, arXiv:2506.08456)
de-staticking helper used by sample_funpack_distilled_flow.

Covers: blurring specific frame indices and/or a trailing tail of the packed video stream,
shape/identity preservation of every other region (untouched video frames + audio), and the
no-op fallbacks when there's no usable latent_image or the packed layout can't be read.
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

B, C, T, H, W = 1, 2, 4, 4, 4
VIDEO_SHAPE = (B, C, T, H, W)
AUDIO_SHAPE = (B, 6)
VIDEO_SIZE = C * T * H * W  # 128
AUDIO_SIZE = 6
N = VIDEO_SIZE + AUDIO_SIZE


def _fake_model(shapes=(VIDEO_SHAPE, AUDIO_SHAPE)):
    return types.SimpleNamespace(inner_model=types.SimpleNamespace(conds={
        "positive": [{"model_conds": {"latent_shapes": types.SimpleNamespace(cond=list(shapes))}}],
    }))


def _checker():
    return torch.tensor([[(i + j) % 2 for j in range(W)] for i in range(H)], dtype=torch.float32)


def _packed_latent_image():
    # Every frame is the same sharp checkerboard, so any blurred frame is easy to spot.
    video = _checker().expand(B, C, T, H, W).clone()
    audio = torch.full((B, 1, AUDIO_SIZE), 5.0)
    return torch.cat([video.reshape(B, 1, VIDEO_SIZE), audio], dim=-1)


def _video_frames(packed):
    return packed[..., :VIDEO_SIZE].reshape(B, C, T, H, W)


def test_alg_blur_returns_none_without_latent_image():
    model = _fake_model()
    assert samplers._alg_blur_frames(model, None, 2.5, frame_indices=(0,)) is None


def test_alg_blur_returns_none_without_latent_shapes():
    model = types.SimpleNamespace(inner_model=types.SimpleNamespace(conds={}))
    latent_image = _packed_latent_image()
    assert samplers._alg_blur_frames(model, latent_image, 2.5, frame_indices=(0,)) is None


def test_alg_blur_returns_none_on_layout_mismatch():
    model = _fake_model()
    wrong_size_latent = torch.zeros(1, 1, N + 1)
    assert samplers._alg_blur_frames(model, wrong_size_latent, 2.5, frame_indices=(0,)) is None


def test_alg_blur_returns_none_when_no_indices_selected():
    model = _fake_model()
    latent_image = _packed_latent_image()
    assert samplers._alg_blur_frames(model, latent_image, 2.5, frame_indices=(), tail_count=0) is None


def test_alg_blur_only_touches_frame_zero():
    model = _fake_model()
    latent_image = _packed_latent_image()
    out = samplers._alg_blur_frames(model, latent_image, 2.5, frame_indices=(0,))
    assert out is not None
    assert out.shape == latent_image.shape

    video_out = _video_frames(out)
    video_in = _video_frames(latent_image)

    assert not torch.allclose(video_out[:, :, 0], video_in[:, :, 0])
    assert torch.equal(video_out[:, :, 1:], video_in[:, :, 1:])
    assert torch.equal(out[..., VIDEO_SIZE:], latent_image[..., VIDEO_SIZE:])  # audio untouched


def test_alg_blur_tail_count_touches_only_trailing_frames():
    model = _fake_model()
    latent_image = _packed_latent_image()
    # tail_count=2 on a 4-frame video -> blur frames {2, 3} only.
    out = samplers._alg_blur_frames(model, latent_image, 2.5, tail_count=2)
    video_out = _video_frames(out)
    video_in = _video_frames(latent_image)

    assert torch.equal(video_out[:, :, :2], video_in[:, :, :2])  # head untouched
    assert not torch.allclose(video_out[:, :, 2], video_in[:, :, 2])
    assert not torch.allclose(video_out[:, :, 3], video_in[:, :, 3])
    assert torch.equal(out[..., VIDEO_SIZE:], latent_image[..., VIDEO_SIZE:])  # audio untouched


def test_alg_blur_combines_frame_indices_and_tail():
    model = _fake_model()
    latent_image = _packed_latent_image()
    # i2v anchor (frame 0) + a 1-frame guide tail (frame 3) on a 4-frame video; frames 1, 2 stay sharp.
    out = samplers._alg_blur_frames(model, latent_image, 2.5, frame_indices=(0,), tail_count=1)
    video_out = _video_frames(out)
    video_in = _video_frames(latent_image)

    assert not torch.allclose(video_out[:, :, 0], video_in[:, :, 0])
    assert torch.equal(video_out[:, :, 1:3], video_in[:, :, 1:3])
    assert not torch.allclose(video_out[:, :, 3], video_in[:, :, 3])


def test_alg_blur_tail_count_zero_is_just_frame_indices():
    model = _fake_model()
    latent_image = _packed_latent_image()
    out = samplers._alg_blur_frames(model, latent_image, 2.5, frame_indices=(0,), tail_count=0)
    video_out = _video_frames(out)
    video_in = _video_frames(latent_image)
    assert torch.equal(video_out[:, :, 1:], video_in[:, :, 1:])


def test_alg_blur_reduces_high_frequency_variance():
    model = _fake_model()
    latent_image = _packed_latent_image()
    out = samplers._alg_blur_frames(model, latent_image, 2.5, frame_indices=(0,))
    frame0_in = _video_frames(latent_image)[:, :, 0]
    frame0_out = _video_frames(out)[:, :, 0]
    # A checkerboard has maximal local variance; the down+up-sampled copy should not.
    assert float(frame0_out.var()) < float(frame0_in.var())


def test_alg_blur_composes_with_independent_strengths():
    """Anchor and guide-tail blur are separate passes with their own kappa: composing the
    tail pass onto the anchor-blurred latent must leave the anchor frame untouched, and a
    stronger tail kappa must blur the tail harder than the anchor kappa would."""
    model = _fake_model()
    latent_image = _packed_latent_image()
    anchor_only = samplers._alg_blur_frames(model, latent_image, 2.0, frame_indices=(0,))
    both = samplers._alg_blur_frames(model, anchor_only, 4.0, tail_count=1)
    v_anchor_only = _video_frames(anchor_only)
    v_both = _video_frames(both)

    assert torch.equal(v_both[:, :, 0], v_anchor_only[:, :, 0])   # anchor pass preserved
    assert torch.equal(v_both[:, :, 1:3], v_anchor_only[:, :, 1:3])  # middle untouched

    # Strength independence needs content that survives downsampling distinctly (the 4x4
    # checkerboard averages to uniform 0.5 at ANY factor) — use a seeded random 8x8 frame.
    # Different kappa must yield a different blurred tail: proof the per-region strength
    # actually reaches the filter (variance ordering is NOT guaranteed without antialias).
    h8 = 8
    model8 = _fake_model(shapes=((B, C, T, h8, h8), AUDIO_SHAPE))
    video8 = torch.rand(B, C, T, h8, h8, generator=torch.Generator().manual_seed(7))
    packed8 = torch.cat([video8.reshape(B, 1, C * T * h8 * h8),
                         torch.full((B, 1, AUDIO_SIZE), 5.0)], dim=-1)
    weak = samplers._alg_blur_frames(model8, packed8, 2.0, tail_count=1)
    strong = samplers._alg_blur_frames(model8, packed8, 4.0, tail_count=1)
    v_sz = C * T * h8 * h8
    tail_weak = weak[..., :v_sz].reshape(B, C, T, h8, h8)[:, :, 3]
    tail_strong = strong[..., :v_sz].reshape(B, C, T, h8, h8)[:, :, 3]
    tail_orig = video8[:, :, 3]
    assert not torch.allclose(tail_weak, tail_strong)  # kappa reaches the filter
    assert not torch.allclose(tail_weak, tail_orig) and not torch.allclose(tail_strong, tail_orig)


def test_alg_blur_is_deterministic():
    model = _fake_model()
    latent_image = _packed_latent_image()
    out1 = samplers._alg_blur_frames(model, latent_image, 2.5, frame_indices=(0,), tail_count=1)
    out2 = samplers._alg_blur_frames(model, latent_image, 2.5, frame_indices=(0,), tail_count=1)
    assert torch.equal(out1, out2)
