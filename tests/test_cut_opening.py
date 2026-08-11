"""cut_opening_frames — "fake t2v": generate with the real untouched i2v anchor, then cut it.

The point of the feature is that the anchor is NOT weakened on the way in (ALG blurs it and
loses character detail; Best-FaceID overlap tokens approximate it and lose some too) — it is
pinned at full strength for the whole schedule and the opening is then cut off the FINISHED
clip. Nothing is regrown to replace it (regrowing was tried and the invented ending came out
with worse or missing movement).

The cut happens on DECODED PIXELS, never on the latent — on both LTX and H3 now. Cutting the
latent promoted a continuation frame to position 0, and the causal VAE decoded it with the
temporal-origin handling it was never generated for, which showed up as a noisy first frame.
It also could not be exact: a latent cut removed whole latent frames, and since the promoted
frame went from covering 1 pixel to covering `scale`, the clip shortened by a different
amount than the span that was removed.

Invariants worth testing:

1. The clip really does get SHORTER — no tail is invented, no joint is blended, and every
   surviving frame is the original untouched one.
2. Exactly N frames go, from the right place, including for a mid-chain scene opening, and
   multiple spans do not shift each other's indices.
3. Video and audio still describe the same duration. Audio timing comes from the audio
   stream's own index, so cropping the two by different amounts desyncs the clip silently.
4. The latent's VIDEO stream is left whole — that is what keeps the origin intact.
5. The cut can never consume the whole clip, and never mutates its input.
6. The t2v guard: no pinned anchor means there is nothing to cut out.
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


# ── the cut: pixels, never latents ──────────────────────────────────────────

def test_scene_pixel_start_accounts_for_the_causal_origin():
    """The VAE decodes f latent frames to (f-1)*scale+1 pixels: latent frame 0 is the
    temporal origin and covers ONE pixel, every later frame covers `scale`. A scene starting
    at latent frame i therefore starts at pixel (i-1)*scale+1, not i*scale."""
    n = _node()
    assert n._scene_pixel_start(0, 8) == 0
    assert n._scene_pixel_start(1, 8) == 1
    assert n._scene_pixel_start(2, 8) == 9
    assert n._scene_pixel_start(5, 8) == 33


def test_cut_removes_exactly_the_requested_pixel_frames():
    """N means N. The old latent cut could only remove whole latent frames, and because the
    promoted frame changed from covering 1 pixel to covering `scale`, the clip actually
    shortened by a different amount than the span it removed."""
    images = torch.arange(20, dtype=torch.float32).view(20, 1, 1, 1)
    cut, dropped = _node()._cut_opening_pixel_spans(images, [(0, 6)])
    assert dropped == 6
    assert cut.shape[0] == 14
    # The survivors are the ORIGINAL frames 6..19, untouched — no blend, no joint.
    assert [int(v) for v in cut[:, 0, 0, 0]] == list(range(6, 20))


def test_cut_handles_a_mid_chain_scene_opening():
    """A hard-cut scene later in the chain has its own anchor to remove, so the cut is a
    span, not just a head crop."""
    images = torch.arange(20, dtype=torch.float32).view(20, 1, 1, 1)
    cut, dropped = _node()._cut_opening_pixel_spans(images, [(10, 4)])
    assert dropped == 4
    assert [int(v) for v in cut[:, 0, 0, 0]] == list(range(10)) + list(range(14, 20))


def test_multiple_spans_do_not_shift_each_other():
    """Removing an earlier span first would move every later index — the classic
    delete-while-iterating bug. Later spans must be removed first."""
    images = torch.arange(20, dtype=torch.float32).view(20, 1, 1, 1)
    cut, dropped = _node()._cut_opening_pixel_spans(images, [(0, 3), (10, 2)])
    assert dropped == 5
    assert [int(v) for v in cut[:, 0, 0, 0]] == list(range(3, 10)) + list(range(12, 20))


def test_cut_never_consumes_the_whole_clip():
    """A cut longer than the clip is a misconfiguration; an empty batch turns it into an
    obscure downstream crash instead of a visibly-too-short video."""
    images = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1)
    cut, dropped = _node()._cut_opening_pixel_spans(images, [(0, 99)])
    assert cut.shape[0] == 1
    assert dropped == 4


def test_no_spans_is_a_no_op():
    images = torch.arange(6, dtype=torch.float32).view(6, 1, 1, 1)
    cut, dropped = _node()._cut_opening_pixel_spans(images, [])
    assert dropped == 0
    assert cut is images


def test_source_images_are_never_mutated():
    images = torch.arange(8, dtype=torch.float32).view(8, 1, 1, 1)
    before = images.clone()
    _node()._cut_opening_pixel_spans(images, [(0, 3)])
    assert torch.equal(images, before)


# ── audio stays in sync ─────────────────────────────────────────────────────

def test_audio_loses_the_same_proportion_of_time(nested_stub):
    """Audio timing comes from the audio stream's own index and video's from its own RoPE,
    so cropping them by different amounts of TIME desyncs the clip silently — no error, just
    drifting sound. Spans are mapped by proportion, not by frame count."""
    latent = _av_latent(10, 20)
    out = _node()._remove_latent_time_spans(latent, [(0, 40)], 100)
    _v, audio = out["samples"].unbind()
    assert audio.shape[2] == 12          # 40% of 20 removed from the head
    assert [int(v) for v in audio[0, 0]] == list(range(8, 20))


def test_video_stream_is_left_whole(nested_stub):
    """The video latent must keep its temporal origin — that is the entire point of moving
    the cut to pixels. Video comes from the decoded IMAGES instead."""
    latent = _av_latent(10, 20)
    out = _node()._remove_latent_time_spans(latent, [(0, 40)], 100)
    video, _a = out["samples"].unbind()
    assert video.shape[2] == 10


def test_audio_span_removal_never_empties_the_stream(nested_stub):
    out = _node()._remove_latent_time_spans(_av_latent(10, 20), [(0, 100)], 100)
    _v, audio = out["samples"].unbind()
    assert audio.shape[2] >= 1


def test_source_latent_is_never_mutated(nested_stub):
    latent = _av_latent(10, 20)
    before = latent["samples"].unbind()[1].clone()
    _node()._remove_latent_time_spans(latent, [(0, 40)], 100)
    assert torch.equal(latent["samples"].unbind()[1], before)


def test_audio_uses_its_own_time_axis(nested_stub):
    """MiniMax H3 puts the STEREO CHANNEL on dim 2 and time last ([B, 32, 2, T]); LTXAV puts
    time on dim 2. Slicing an H3 audio stream on dim 2 would crop the stereo pair down to
    mono and leave the duration untouched — silent, and wrong both ways."""
    n = _node()
    video = torch.zeros(1, 4, 10, 2, 2)
    audio = torch.arange(20, dtype=torch.float32).view(1, 1, 1, 20).repeat(1, 32, 2, 1)
    n._time_dims = (2, 3, 3, 3)  # what _set_stream_axes records on H3
    out = n._remove_latent_time_spans({"samples": _FakeNested([video, audio])}, [(0, 40)], 100)
    _v, a = out["samples"].unbind()
    assert a.shape[2] == 2       # both stereo channels survive
    assert a.shape[3] == 12      # 40% of the duration, removed from the head
    assert [int(v) for v in a[0, 0, 0]] == list(range(8, 20))


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
