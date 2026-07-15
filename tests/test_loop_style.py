"""Unit tests for the loop temporal style (Mobius latent roll, arXiv:2502.20307).

Covers: per-stream temporal roll of the packed [B,1,N] latent (video and audio each by
their own frame count), exact unroll round-trips, denoise-mask rolls staying in step,
the eligibility gates (near-noise plateau, guide-pinned calls, tiny clips, unreadable
packed layouts), van der Corput shift spreading with per-scene reset, and wrapper
chaining semantics (inner wrappers see rolled args, callers always get canonical
predictions back).
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ltx_enhancements import (  # noqa: E402
    _loop_roll_mask,
    _loop_roll_packed,
    _loop_stream_shapes,
    _van_der_corput,
    make_loop_temporal_wrapper,
)

VSHAPE, ASHAPE = (1, 128, 16, 4, 6), (1, 8, 32, 16)


def _packed():
    video = torch.arange(math.prod(VSHAPE), dtype=torch.float32).reshape(VSHAPE)
    audio = torch.arange(math.prod(ASHAPE), dtype=torch.float32).reshape(ASHAPE) + 1e6
    packed = torch.cat([video.reshape(1, 1, -1), audio.reshape(1, 1, -1)], dim=-1)
    return video, audio, packed


def _args(sigma, packed, mask=None, extra_c=None):
    c = {"latent_shapes": [torch.Size(VSHAPE), torch.Size(ASHAPE)]}
    if mask is not None:
        c["denoise_mask"] = mask
    if extra_c:
        c.update(extra_c)
    return {"input": packed, "timestep": torch.tensor([sigma]), "c": c}


def test_van_der_corput_prefix():
    assert [_van_der_corput(n) for n in range(1, 6)] == [0.5, 0.25, 0.75, 0.125, 0.625]


def test_roll_per_stream_and_round_trip():
    video, audio, packed = _packed()
    rolled = _loop_roll_packed(packed, [VSHAPE, ASHAPE], 0.5, 1)
    assert torch.equal(rolled[..., : video.numel()].reshape(VSHAPE), torch.roll(video, 8, dims=2))
    assert torch.equal(rolled[..., video.numel():].reshape(ASHAPE), torch.roll(audio, 16, dims=2))
    assert torch.equal(_loop_roll_packed(rolled, [VSHAPE, ASHAPE], 0.5, -1), packed)


def test_roll_fraction_keeps_streams_time_aligned():
    video, audio, packed = _packed()
    rolled = _loop_roll_packed(packed, [VSHAPE, ASHAPE], 0.25, 1)
    assert torch.equal(rolled[..., : video.numel()].reshape(VSHAPE), torch.roll(video, 4, dims=2))
    assert torch.equal(rolled[..., video.numel():].reshape(ASHAPE), torch.roll(audio, 8, dims=2))


def test_mask_roll_round_trip():
    mask = torch.rand(1, 1, 16, 4, 6)
    rolled = _loop_roll_mask(mask, 0.5, 1)
    assert torch.equal(rolled, torch.roll(mask, 8, dims=2))
    assert torch.equal(_loop_roll_mask(rolled, 0.5, -1), mask)


def test_stream_shapes_refuses_mismatched_packing():
    _, _, packed = _packed()
    ok = _loop_stream_shapes(_args(0.7, packed))
    assert ok == [VSHAPE, ASHAPE]
    assert _loop_stream_shapes(_args(0.7, packed[..., :-5])) is None


def test_wrapper_rolls_in_unrolls_out():
    video, _, packed = _packed()
    mask = torch.rand(1, 1, 16, 4, 6)
    seen = {}

    def fake_apply(x, ts, **c):
        seen["input"], seen["mask"] = x, c.get("denoise_mask")
        return x * 1.0

    w = make_loop_temporal_wrapper(None)
    out = w(fake_apply, _args(0.909, packed, mask))
    assert torch.equal(seen["input"][..., : video.numel()].reshape(VSHAPE), torch.roll(video, 8, dims=2))
    assert torch.equal(seen["mask"], torch.roll(mask, 8, dims=2))
    assert torch.equal(out, packed)  # canonical on the way out
    # second eligible step advances the shift sequence (0.25 -> 4 frames)
    w(fake_apply, _args(0.725, packed, mask))
    assert torch.equal(seen["input"][..., : video.numel()].reshape(VSHAPE), torch.roll(video, 4, dims=2))


def test_wrapper_gates_plateau_unparseable_guides_and_tiny_clips():
    video, _, packed = _packed()
    seen = {}

    def fake_apply(x, ts, **c):
        seen["input"] = x
        return x * 1.0

    w = make_loop_temporal_wrapper(None)
    w(fake_apply, _args(0.99, packed))  # near-noise plateau
    assert torch.equal(seen["input"], packed)
    # malformed keyframe_idxs (too few dims to count tokens): canonical, never a blind roll
    w(fake_apply, _args(0.725, packed, extra_c={"keyframe_idxs": torch.zeros(1, 1)}))
    assert torch.equal(seen["input"], packed)
    # attention entries without keyframe_idxs: guide context we can't size -> canonical
    w(fake_apply, _args(0.725, packed, extra_c={"guide_attention_entries": [object()]}))
    assert torch.equal(seen["input"], packed)

    tiny_shape = (1, 128, 2, 4, 6)
    tiny = torch.rand(1, 1, math.prod(tiny_shape[1:]))
    w2 = make_loop_temporal_wrapper(None)
    w2(fake_apply, {"input": tiny, "timestep": torch.tensor([0.7]),
                    "c": {"latent_shapes": [torch.Size(tiny_shape)]}})
    assert torch.equal(seen["input"], tiny)


def _keyframe_idxs_for(n_frames, h=4, w=6):
    # per-token layout: [B, coords, n_tokens, start/end] — the wrapper only reads shape[2]
    return torch.zeros(1, 1, n_frames * h * w, 2)


def test_wrapper_content_only_roll_with_guide_tail():
    """Guide frames appended at the video tail stay pinned; only content rolls."""
    v_tail = 2
    video, audio, packed = _packed()  # video T=16 -> content 14; audio T=32, no tail
    mask = torch.rand(1, 1, 16, 4, 6)
    seen = {}

    def fake_apply(x, ts, **c):
        seen["input"], seen["mask"] = x, c.get("denoise_mask")
        return x * 1.0

    w = make_loop_temporal_wrapper(None)
    out = w(fake_apply, _args(0.909, packed, mask,
                              extra_c={"keyframe_idxs": _keyframe_idxs_for(v_tail)}))
    v_in = seen["input"][..., : video.numel()].reshape(VSHAPE)
    # first eligible step: frac 0.5 of the 14 content frames -> shift 7
    assert torch.equal(v_in[:, :, :14], torch.roll(video[:, :, :14], 7, dims=2))
    assert torch.equal(v_in[:, :, 14:], video[:, :, 14:])          # guide tail untouched
    assert torch.equal(seen["mask"][:, :, :14], torch.roll(mask[:, :, :14], 7, dims=2))
    assert torch.equal(seen["mask"][:, :, 14:], mask[:, :, 14:])   # tail mask untouched
    # audio has no tail: rolls fully by its own content length (0.5 * 32 = 16)
    a_in = seen["input"][..., video.numel():].reshape(ASHAPE)
    assert torch.equal(a_in, torch.roll(audio, 16, dims=2))
    assert torch.equal(out, packed)                                # unrolled on the way out


def test_wrapper_audio_memory_tail_stays_pinned():
    """A trailing mask=0 run on the audio stream (JoyAI memory) is treated as pinned."""
    a_tail = 4
    video, audio, packed = _packed()
    amask = torch.ones(1, 1, 32, 16)
    amask[:, :, -a_tail:] = 0.0
    seen = {}

    def fake_apply(x, ts, **c):
        seen["input"], seen["amask"] = x, c.get("audio_denoise_mask")
        return x * 1.0

    w = make_loop_temporal_wrapper(None)
    out = w(fake_apply, _args(0.909, packed, extra_c={"audio_denoise_mask": amask}))
    a_in = seen["input"][..., video.numel():].reshape(ASHAPE)
    # audio content 28 frames -> frac 0.5 -> shift 14; tail canonical
    assert torch.equal(a_in[:, :, :28], torch.roll(audio[:, :, :28], 14, dims=2))
    assert torch.equal(a_in[:, :, 28:], audio[:, :, 28:])
    assert torch.equal(seen["amask"], amask)  # all-ones content rolls invisibly; zeros pinned
    assert torch.equal(out, packed)


def test_wrapper_guide_tail_reduces_content_gate():
    """A clip whose content region (T - tail) is under the minimum stays canonical."""
    shape = (1, 128, 5, 4, 6)
    x = torch.rand(1, 1, math.prod(shape[1:]))
    seen = {}

    def fake_apply(x_, ts, **c):
        seen["input"] = x_
        return x_ * 1.0

    w = make_loop_temporal_wrapper(None)
    w(fake_apply, {"input": x, "timestep": torch.tensor([0.7]),
                   "c": {"latent_shapes": [torch.Size(shape)],
                         "keyframe_idxs": _keyframe_idxs_for(2)}})  # content 3 < 4
    assert torch.equal(seen["input"], x)


def test_wrapper_resets_shift_sequence_on_new_scene():
    video, _, packed = _packed()
    seen = {}

    def fake_apply(x, ts, **c):
        seen["input"] = x
        return x * 1.0

    w = make_loop_temporal_wrapper(None)
    w(fake_apply, _args(0.909, packed))  # frac 0.5
    w(fake_apply, _args(0.725, packed))  # frac 0.25
    w(fake_apply, _args(0.99, packed))   # sigma jumps back up: new scene
    w(fake_apply, _args(0.909, packed))  # sequence restarts at 0.5
    assert torch.equal(seen["input"][..., : video.numel()].reshape(VSHAPE), torch.roll(video, 8, dims=2))


def test_wrapper_unpacked_single_stream():
    seen = {}

    def fake_apply(x, ts, **c):
        seen["input"] = x
        return x * 1.0

    five_d = torch.arange(math.prod(VSHAPE), dtype=torch.float32).reshape(VSHAPE)
    w = make_loop_temporal_wrapper(None)
    out = w(fake_apply, {"input": five_d, "timestep": torch.tensor([0.7]), "c": {}})
    assert torch.equal(seen["input"], torch.roll(five_d, 8, dims=2))
    assert torch.equal(out, five_d)


def test_wrapper_chain_inner_sees_rolled_caller_sees_canonical():
    _, _, packed = _packed()
    seen = {}

    def fake_apply(x, ts, **c):
        return x * 1.0

    def inner(apply_fn, args):
        seen["inner_input"] = args["input"]
        return apply_fn(args["input"], args["timestep"], **args.get("c", {}))

    w = make_loop_temporal_wrapper(inner)
    out = w(fake_apply, _args(0.725, packed))
    assert not torch.equal(seen["inner_input"], packed)
    assert torch.equal(out, packed)
