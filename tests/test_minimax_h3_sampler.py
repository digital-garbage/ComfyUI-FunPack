"""Chain Sampler on MiniMax H3 — the stream-axis and keyframe routing that used to be LTX-only.

These exercise the sampler's own helpers directly rather than a full sample() run, because the
thing under test is geometry: which axis of which stream a slice lands on, and where a guide
goes when the model has no guide-attention API. Both are silent failures — the LTXAV code path
returns a tensor of a plausible shape either way.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeNestedTensor:
    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors

    @property
    def shape(self):
        return self.tensors[0].shape


for _name, _attrs in (
    ("comfy", {}),
    ("comfy.k_diffusion", {}),
    ("comfy.k_diffusion.sampling", {}),
    ("comfy.model_sampling", {}),
    ("comfy.nested_tensor", {"NestedTensor": FakeNestedTensor}),
    ("comfy.sample", {"prepare_noise": lambda *a, **k: None, "sample_custom": lambda *a, **k: None}),
    ("comfy.samplers", {}),
    ("comfy.utils", {"ProgressBar": lambda total: types.SimpleNamespace(
        update_absolute=lambda *a, **k: None)}),
):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        for _k, _v in _attrs.items():
            setattr(_mod, _k, _v)
        sys.modules[_name] = _mod
_comfy = sys.modules["comfy"]
for _sub in ("k_diffusion", "model_sampling", "nested_tensor", "sample", "samplers", "utils"):
    setattr(_comfy, _sub, sys.modules["comfy." + _sub])
setattr(sys.modules["comfy.k_diffusion"], "sampling", sys.modules["comfy.k_diffusion.sampling"])

from samplers import FunPackLTXAVSceneChainSampler  # noqa: E402

NT = FakeNestedTensor


@pytest.fixture(autouse=True)
def _own_nested_tensor():
    """Force the sampler to rebuild nested latents with THIS file's stub.

    Several modules in this suite install their own `comfy.nested_tensor` stub into
    sys.modules, and whichever imports first wins for the whole run — so a test that
    passes alone can fail in a full run purely on import order. Pinning it per test
    keeps these assertions about the sampler rather than about collection order.
    """
    import comfy.nested_tensor as mod
    previous = mod.NestedTensor
    mod.NestedTensor = FakeNestedTensor
    try:
        yield
    finally:
        mod.NestedTensor = previous


class H3VAE:
    """What comfy/sd.py builds for the H3 video VAE."""
    downscale_ratio = (lambda a: max(1, (a - 5) // 17 * 5 + 2) if a > 1 else 1, 16, 16)
    downscale_index_formula = (4, 16, 16)


class LTXVAE:
    downscale_ratio = (lambda a: max(0, (a + 7) // 8), 32, 32)
    downscale_index_formula = (8, 32, 32)


def h3_node(frame_count=124):
    """A Chain Sampler already told it is running on H3, as sample() would tell it."""
    node = FunPackLTXAVSceneChainSampler()
    node._is_h3 = True
    node._time_dims = (2, -1, -1, -1)
    node._h3_frame_count = frame_count
    return node


def av_latent(video_t=37, audio_t=207):
    video = torch.zeros(1, 24, video_t, 48, 84)
    audio = torch.zeros(1, 32, 2, audio_t)
    return {"samples": NT([video, audio])}


# ── stream axes ──────────────────────────────────────────────────────────────

def test_frame_counts_read_the_right_axis_per_stream():
    node = h3_node()
    video, audio = av_latent()["samples"].unbind()
    assert node._tensor_frames(video, stream=0) == 37
    assert node._tensor_frames(audio, stream=1) == 207     # not 2, the stereo axis
    # an LTXAV run keeps both streams on dim 2, exactly as before
    ltx = FunPackLTXAVSceneChainSampler()
    assert ltx._tensor_frames(torch.zeros(1, 8, 40, 4), stream=1) == 40


def test_continuation_chunk_pins_the_audio_tail_not_a_speaker():
    """The overlap prefix must land on time; on the stereo axis it would blank a channel."""
    node = h3_node()
    template = av_latent()
    previous = av_latent()
    for t in previous["samples"].unbind():
        t.fill_(1.0)

    chunk = node._build_continuation_chunk(template, previous, video_overlap=2)
    video, audio = chunk["samples"].unbind()
    vmask, amask = chunk["noise_mask"].unbind()

    assert video.shape == (1, 24, 37, 48, 84)
    assert audio.shape == (1, 32, 2, 207)                  # shape preserved, not reshaped
    # video: first 2 latent frames carried and protected
    assert video[:, :, :2].eq(1.0).all() and video[:, :, 2:].eq(0.0).all()
    assert vmask[:, :, :2].eq(0).all() and vmask[:, :, 2:].eq(1).all()
    # audio: the proportional overlap lands on the LAST axis, both speakers intact
    aov = int(round(2 * 207 / 37))
    assert audio[..., :aov].eq(1.0).all() and audio[..., aov:].eq(0.0).all()
    assert amask[..., :aov].eq(0).all() and amask[..., aov:].eq(1).all()
    assert amask.shape[2] == 2                              # stereo axis untouched


def test_blending_two_scenes_keeps_stereo_and_extends_time():
    node = h3_node()
    previous, current = av_latent(video_t=10, audio_t=56), av_latent(video_t=10, audio_t=56)
    for t in current["samples"].unbind():
        t.fill_(1.0)

    blended = node._blend_latents(previous, current, video_overlap=3)
    video, audio = blended["samples"].unbind()
    assert video.shape[2] == 10 + 10 - 3
    assert audio.shape[2] == 2                              # stereo, not consumed as time
    aov = int(round(3 * 56 / 10))
    assert audio.shape[-1] == 56 + 56 - aov


def test_cropping_the_joyai_audio_tail_trims_time():
    node = h3_node()
    latent = av_latent(video_t=10, audio_t=56)
    out = node._crop_audio_tail(latent, 6)
    video, audio = out["samples"].unbind()
    assert audio.shape == (1, 32, 2, 50)
    assert video.shape[2] == 10                             # video untouched


def test_joyai_audio_memory_appends_on_the_time_axis():
    node = h3_node()
    chunk = av_latent(video_t=10, audio_t=56)
    frame = torch.ones(1, 32, 2, 1)
    out, appended = node._append_joyai_audio_memory(chunk, [frame, frame])
    assert appended == 2
    _video, audio = out["samples"].unbind()
    assert audio.shape == (1, 32, 2, 58)
    assert audio[..., -2:].eq(1.0).all()
    _vmask, amask = out["noise_mask"].unbind()
    assert amask[..., -2:].eq(0).all()                      # pinned, never denoised


def test_joyai_audio_memory_rejects_a_frame_of_the_wrong_stereo_width():
    node = h3_node()
    chunk = av_latent(video_t=10, audio_t=56)
    mono = torch.ones(1, 32, 1, 1)
    out, appended = node._append_joyai_audio_memory(chunk, [mono])
    assert appended == 0
    assert out is chunk


# ── frame grid ───────────────────────────────────────────────────────────────

def test_template_validation_uses_the_vaes_own_count_map():
    node = h3_node()
    template = av_latent(video_t=37)
    # 124 pixel frames -> 37 latent frames on H3's 17k+5 grid
    assert node._validate_template_length(template, 124, time_scale=4, vae=H3VAE()) == 37
    with pytest.raises(ValueError):
        node._validate_template_length(template, 141, time_scale=4, vae=H3VAE())


def test_ltx_template_validation_is_unchanged():
    node = FunPackLTXAVSceneChainSampler()
    template = {"samples": torch.zeros(1, 128, 16, 4, 4)}
    assert node._validate_template_length(template, 121, time_scale=8, vae=LTXVAE()) == 16
    # and identical with no VAE passed at all (the old uniform formula)
    assert node._validate_template_length(template, 121, time_scale=8) == 16


def test_decode_tile_size_is_divided_by_the_vaes_spatial_ratio():
    node = FunPackLTXAVSceneChainSampler()
    assert node._decode_tile_latent(H3VAE(), 512) == 32     # 16x downscale
    assert node._decode_tile_latent(LTXVAE(), 512) == 16    # 32x downscale
    assert node._decode_tile_latent(object(), 512) == 64    # unknown VAE -> old behaviour


# ── guides become keyframe pins ──────────────────────────────────────────────

def test_a_first_frame_guide_becomes_a_conditioning_keyframe_with_no_latent_tail():
    node = h3_node(frame_count=124)
    chunk = av_latent()
    guide = torch.ones(1, 24, 1, 48, 84)
    positive = [[torch.zeros(1, 12, 5120), {"funpack_scene_text": "shot 2"}]]

    out_chunk, pos, neg, tail = node._append_guide_latent(
        chunk, guide, apply_at=0, strength=1.0, positive=positive, negative=[], vae=H3VAE())

    assert tail == 0                                    # nothing appended -> nothing to crop
    assert out_chunk is chunk                           # the latent is not touched at all
    meta = pos[0][1]
    assert meta["minimax_frame_count"] == 124
    assert [kf["resolved_frame_index"] for kf in meta["minimax_keyframes"]] == [0]
    assert meta["minimax_keyframes"][0]["latent"] is guide
    assert meta["funpack_scene_text"] == "shot 2"       # existing metadata preserved
    assert "minimax_visual_cond_noise_aug" not in meta  # strength 1.0 = clean pin


def test_a_last_frame_guide_is_accepted():
    node = h3_node(frame_count=124)
    pos, _neg, tail = node._append_h3_keyframe(
        torch.ones(1, 24, 1, 48, 84), apply_at=-1, strength=0.8,
        positive=[[torch.zeros(1, 12, 5120), {}]], negative=[])
    assert tail == 0
    assert pos[0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 123
    assert pos[0][1]["minimax_visual_cond_noise_aug"] == pytest.approx(0.8)


def test_a_mid_clip_guide_is_refused_rather_than_crashing_the_sample(capsys):
    """PackedLayout raises for anything but first/last — better to say so up front."""
    node = h3_node(frame_count=124)
    positive = [[torch.zeros(1, 12, 5120), {}]]
    pos, _neg, tail = node._append_h3_keyframe(
        torch.ones(1, 24, 1, 48, 84), apply_at=60, strength=1.0,
        positive=positive, negative=[])
    assert tail == 0
    assert pos is positive                              # conditioning untouched
    assert "minimax_keyframes" not in positive[0][1]
    out = capsys.readouterr().out
    assert "first (0) or last (123)" in out


def test_ltx_guides_still_take_the_ltx_path():
    """The H3 branch must not fire for an LTXAV run; without comfy_extras it no-ops as before."""
    node = FunPackLTXAVSceneChainSampler()
    assert node._is_h3 is False
    chunk = {"samples": torch.zeros(1, 128, 8, 4, 4)}
    out_chunk, pos, neg, tail = node._append_guide_latent(
        chunk, torch.ones(1, 128, 1, 4, 4), apply_at=0, strength=1.0,
        positive=[[torch.zeros(1, 12, 4096), {}]], negative=[], vae=LTXVAE())
    assert tail == 0
    assert "minimax_keyframes" not in pos[0][1]
