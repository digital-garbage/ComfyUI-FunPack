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


def test_wrapper_gates_plateau_guides_and_tiny_clips():
    video, _, packed = _packed()
    seen = {}

    def fake_apply(x, ts, **c):
        seen["input"] = x
        return x * 1.0

    w = make_loop_temporal_wrapper(None)
    w(fake_apply, _args(0.99, packed))  # near-noise plateau
    assert torch.equal(seen["input"], packed)
    w(fake_apply, _args(0.725, packed, extra_c={"keyframe_idxs": torch.zeros(1, 1)}))
    assert torch.equal(seen["input"], packed)
    w(fake_apply, _args(0.725, packed, extra_c={"guide_attention_entries": [object()]}))
    assert torch.equal(seen["input"], packed)

    tiny_shape = (1, 128, 2, 4, 6)
    tiny = torch.rand(1, 1, math.prod(tiny_shape[1:]))
    w2 = make_loop_temporal_wrapper(None)
    w2(fake_apply, {"input": tiny, "timestep": torch.tensor([0.7]),
                    "c": {"latent_shapes": [torch.Size(tiny_shape)]}})
    assert torch.equal(seen["input"], tiny)


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
