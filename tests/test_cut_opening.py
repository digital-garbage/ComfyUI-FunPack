"""cut_opening_frames — "fake t2v": generate with the real untouched i2v anchor, then cut it.

The point of the feature is that the anchor is NOT weakened on the way in (ALG blurs it and
loses character detail; Best-FaceID overlap tokens approximate it and lose some too) — it is
pinned at full strength for the whole schedule and the opening is then cut off the FINISHED
clip. Nothing is regrown to replace it (regrowing was tried and the invented ending came out
with worse or missing movement), so the invariants worth testing are:

1. The clip really does get SHORTER — no tail is invented, no joint is blended, and every
   surviving frame is the original untouched one.
2. Video and audio still describe the same duration. Audio timing comes from the audio
   stream's own index, so cropping the two by different amounts desyncs the clip silently.
3. The pin goes with the frame it belonged to.
4. The cut can never consume the whole clip, and never mutates its input.
5. The t2v guard: no pinned anchor means there is nothing to cut out.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _name in (
    "comfy", "comfy.k_diffusion", "comfy.k_diffusion.sampling",
    "comfy.model_sampling", "comfy.nested_tensor", "comfy.sample",
    "comfy.samplers", "comfy.utils",
):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["comfy.nested_tensor"].NestedTensor = object

import samplers  # noqa: E402


class _FakeNested(list):
    """Minimal stand-in for comfy's NestedTensor: enough for _is_nested + unbind."""
    is_nested = True

    def unbind(self):
        return list(self)


@pytest.fixture
def nested_stub():
    """Give the module a usable NestedTensor for the packed AV latents.

    Snapshot/restore: these comfy stubs are SHARED module objects across every test module,
    so leaving mutations behind changes other files' results depending on collection order.
    """
    comfy_mod = sys.modules["comfy"]
    nested_mod = sys.modules["comfy.nested_tensor"]
    saved = (nested_mod.NestedTensor, getattr(comfy_mod, "nested_tensor", None))
    nested_mod.NestedTensor = _FakeNested
    comfy_mod.nested_tensor = nested_mod  # attribute access, not just sys.modules
    yield
    nested_mod.NestedTensor = saved[0]
    if saved[1] is None:
        delattr(comfy_mod, "nested_tensor")
    else:
        comfy_mod.nested_tensor = saved[1]


def _node():
    return samplers.FunPackLTXAVSceneChainSampler()


def _latent(frames=10):
    video = torch.arange(frames, dtype=torch.float32).view(1, 1, frames, 1, 1).repeat(1, 4, 1, 2, 2)
    return {"samples": video, "noise_mask": torch.zeros_like(video)}


def _av_latent(video_frames=10, audio_frames=20):
    """A packed-style AV latent: video [B,C,F,H,W] + audio [B,C,T] at its own rate."""
    video = torch.arange(video_frames, dtype=torch.float32).view(
        1, 1, video_frames, 1, 1).repeat(1, 4, 1, 2, 2)
    audio = torch.arange(audio_frames, dtype=torch.float32).view(1, 1, audio_frames).repeat(1, 4, 1)
    return {"samples": _FakeNested([video, audio]), "noise_mask": torch.zeros_like(video)}


# ── the cut ─────────────────────────────────────────────────────────────────

def test_cut_shortens_the_clip_instead_of_regrowing_a_tail():
    cut, dropped = _node()._cut_opening_latent(_latent(frames=10), 3)
    assert dropped == 3
    # 10 - 3, not 10: nothing is invented at the end, which is the whole point.
    assert cut["samples"].shape[2] == 7
    # And what survives is the ORIGINAL frames 3..9, untouched — no blend, no joint.
    assert [int(v) for v in cut["samples"][0, 0, :, 0, 0]] == [3, 4, 5, 6, 7, 8, 9]


def test_cut_drops_the_pin_with_the_frame_it_belonged_to():
    latent = _latent(frames=10)
    assert "noise_mask" in latent
    cut, _ = _node()._cut_opening_latent(latent, 1)
    assert "noise_mask" not in cut


def test_cut_never_consumes_the_whole_clip():
    cut, dropped = _node()._cut_opening_latent(_latent(frames=4), 99)
    assert dropped == 3  # clamped to frames - 1
    assert cut["samples"].shape[2] == 1


def test_zero_drop_is_a_no_op():
    cut, dropped = _node()._cut_opening_latent(_latent(frames=6), 0)
    assert dropped == 0
    assert torch.equal(cut["samples"], _latent(frames=6)["samples"])


def test_source_latent_is_never_mutated():
    latent = _latent(frames=8)
    before = latent["samples"].clone()
    _node()._cut_opening_latent(latent, 3)
    assert torch.equal(latent["samples"], before)


# ── audio stays in sync ─────────────────────────────────────────────────────

def test_cut_keeps_audio_and_video_describing_the_same_duration(nested_stub):
    """Audio timing comes from the audio stream's own index and video's from its own RoPE,
    so cropping them by different amounts of TIME desyncs the clip silently — no error,
    just drifting sound. The kept lengths must stay in proportion."""
    cut, _ = _node()._cut_opening_latent(_av_latent(10, 20), 3)
    video, audio = cut["samples"].unbind()
    assert video.shape[2] == 7 and audio.shape[2] == 14      # 7/10 == 14/20
    # Cropped from the HEAD, matching the video — the tail is what survives.
    assert [int(v) for v in audio[0, 0]] == list(range(6, 20))


def test_cut_keeps_proportion_when_the_rate_does_not_divide_evenly(nested_stub):
    """The rounding trap: deriving audio's KEPT length from the new video length keeps the
    two in proportion; subtracting a separately-rounded drop can leave them a frame apart."""
    for v_frames, a_frames, drop in ((10, 15, 3), (9, 14, 4), (13, 7, 5), (10, 21, 7)):
        cut, _ = _node()._cut_opening_latent(_av_latent(v_frames, a_frames), drop)
        video, audio = cut["samples"].unbind()
        assert audio.shape[2] == max(1, round(a_frames * video.shape[2] / v_frames))
        assert audio.shape[2] >= 1 and video.shape[2] >= 1


def test_cut_uses_each_streams_own_time_axis(nested_stub):
    """MiniMax H3 puts the STEREO CHANNEL on dim 2 and time last ([B, 32, 2, T]); LTXAV puts
    time on dim 2 for both streams. Slicing an H3 audio stream on dim 2 would crop the stereo
    pair down to mono and leave the duration untouched — silent, and wrong both ways."""
    n = _node()
    video = torch.zeros(1, 4, 10, 2, 2)
    audio = torch.arange(20, dtype=torch.float32).view(1, 1, 1, 20).repeat(1, 32, 2, 1)
    n._time_dims = (2, 3, 3, 3)  # what _set_stream_axes records on H3
    cut, _ = n._cut_opening_latent(
        {"samples": _FakeNested([video, audio])}, 3)
    _v, a = cut["samples"].unbind()
    assert a.shape[2] == 2       # both stereo channels survive
    assert a.shape[3] == 14      # 7/10 of the duration, cropped from the head
    assert [int(v) for v in a[0, 0, 0]] == list(range(6, 20))


# ── H3: the cut lands on the decoded frames, not the latent ─────────────────

def test_h3_cut_drops_the_opening_off_the_image_batch(nested_stub):
    """H3's anchor is a keyframe condition row, not a pinned latent frame, and its video
    latent sits on a 5k+2 grid an arbitrary cut cannot land on. So the frames come off the
    DECODED batch — exact to the frame, no time-scale rounding."""
    n = _node()
    n._time_dims = (2, 3, 3, 3)
    images = torch.arange(48, dtype=torch.float32).view(48, 1, 1, 1)
    latent = {"samples": _FakeNested([torch.zeros(1, 4, 12, 2, 2),
                                      torch.arange(80, dtype=torch.float32).view(1, 1, 1, 80).repeat(1, 32, 2, 1)])}
    out_images, out_latent, dropped = n._cut_opening_pixels(images, latent, 17)
    assert dropped == 17
    assert out_images.shape[0] == 31
    assert [int(v) for v in out_images[:, 0, 0, 0]] == list(range(17, 48))
    video, audio = out_latent["samples"].unbind()
    # Audio moves with the picture: 31/48 of its own duration, cropped from the head.
    assert audio.shape[3] == round(80 * 31 / 48)
    # The video latent is deliberately LEFT ALONE — cutting it would put it off-grid.
    assert video.shape[2] == 12


def test_h3_cut_never_consumes_the_whole_batch_and_never_mutates_its_input(nested_stub):
    n = _node()
    n._time_dims = (2, 3, 3, 3)
    images = torch.arange(9, dtype=torch.float32).view(9, 1, 1, 1)
    before = images.clone()
    latent = {"samples": _FakeNested([torch.zeros(1, 4, 4, 2, 2), torch.zeros(1, 32, 2, 16)])}
    out_images, _out_latent, dropped = n._cut_opening_pixels(images, latent, 999)
    assert dropped == 8 and out_images.shape[0] == 1
    assert torch.equal(images, before)


def test_h3_zero_cut_is_a_no_op(nested_stub):
    n = _node()
    images = torch.zeros(6, 1, 1, 1)
    latent = {"samples": _FakeNested([torch.zeros(1, 4, 4, 2, 2), torch.zeros(1, 32, 2, 16)])}
    out_images, out_latent, dropped = n._cut_opening_pixels(images, latent, 0)
    assert dropped == 0 and out_images is images and out_latent is latent


# ── the t2v guard ───────────────────────────────────────────────────────────

def test_pinned_frames_counts_the_anchor_prefix():
    n = _node()
    frames = 6
    video = torch.zeros(1, 4, frames, 2, 2)
    mask = torch.ones(1, 4, frames, 2, 2)
    mask[:, :, :2] = 0.0  # two pinned anchor frames
    assert n._anchor_pinned_frames({"samples": video, "noise_mask": mask}) == 2


def test_pinned_frames_is_zero_for_a_real_t2v_scene():
    """No anchor image attached — Easy Gen's default. Cutting would throw away real content,
    so the caller must skip: this is the 'fake t2v' trick and it needs an anchor to fake with."""
    n = _node()
    video = torch.zeros(1, 4, 6, 2, 2)
    assert n._anchor_pinned_frames({"samples": video, "noise_mask": torch.ones(1, 4, 6, 2, 2)}) == 0
    assert n._anchor_pinned_frames({"samples": video}) == 0  # no mask at all
    assert n._anchor_pinned_frames({"samples": None}) == 0   # unreadable -> refuse, don't raise
