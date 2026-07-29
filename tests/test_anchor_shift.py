"""anchor_shift — "fake t2v": run with the real untouched i2v anchor, then delete it.

The point of the feature is that the anchor is NOT weakened on the way in (ALG blurs it and
loses character detail; Best-FaceID overlap tokens approximate it and lose some too) — it is
pinned at full strength for pass 1 and then removed from the clip entirely. So the invariants
worth testing are about the shift itself:

1. The schedule is cut into two passes that each still have a real step, or refused outright.
2. The slide drops the head WITHOUT shortening the clip — no extra frames are ever generated.
3. The pin is dropped with the frame it belonged to (pass 2 must run unpinned).
4. Audio is reset when asked, since pass 1's audio was formed against frames that moved.
"""
import sys
import types
from pathlib import Path

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


# The standard LTX distilled schedule: plateau, then structure at 0.909, then detail.
SCHEDULE = torch.tensor([1.0, 0.975, 0.909, 0.725, 0.422, 0.0])


def _node():
    return samplers.FunPackLTXAVSceneChainSampler()


def _latent(frames=10, audio=True):
    video = torch.arange(frames, dtype=torch.float32).view(1, 1, frames, 1, 1).repeat(1, 4, 1, 2, 2)
    latent = {"samples": video, "noise_mask": torch.zeros_like(video)}
    if audio:
        latent["_audio"] = torch.ones(1, 4, frames * 2)
    return latent


# ── schedule splitting ──────────────────────────────────────────────────────

def test_split_cuts_at_the_requested_sigma_and_continues_by_default():
    first, second, cut_at, restart_at = _node()._anchor_shift_split_sigmas(SCHEDULE, 0.909, 0.0)
    # float32: the schedule stores 0.909 as 0.90899997, so compare loosely.
    assert abs(cut_at - 0.909) < 1e-5 and abs(restart_at - 0.909) < 1e-5
    # pass 1 ends ON the cut sigma; pass 2 resumes from it, so no step is run twice.
    assert [round(v, 3) for v in first.tolist()] == [1.0, 0.975, 0.909]
    assert [round(v, 3) for v in second.tolist()] == [0.909, 0.725, 0.422, 0.0]


def test_restart_sigma_above_the_cut_rewinds_pass_two():
    first, second, cut_at, restart_at = _node()._anchor_shift_split_sigmas(SCHEDULE, 0.725, 0.975)
    assert abs(cut_at - 0.725) < 1e-5 and abs(restart_at - 0.975) < 1e-5
    assert [round(v, 3) for v in second.tolist()] == [0.975, 0.909, 0.725, 0.422, 0.0]


def test_split_refuses_when_a_pass_would_have_no_steps():
    n = _node()
    # Threshold at/above the first sigma: pass 1 would be a single point, no step.
    assert n._anchor_shift_split_sigmas(SCHEDULE, 1.0, 0.0) is None
    # Threshold below everything in the schedule: never reached.
    assert n._anchor_shift_split_sigmas(SCHEDULE, 0.001, 0.0) is None
    # One step either side is legitimate, even on a minimal 3-sigma schedule.
    assert n._anchor_shift_split_sigmas(torch.tensor([1.0, 0.5, 0.0]), 0.5, 0.0) is not None
    # ...but cutting at the FINAL sigma leaves pass 2 a single point, no step.
    assert n._anchor_shift_split_sigmas(torch.tensor([1.0, 0.5, 0.0]), 0.0, 0.0) is None


def test_split_refuses_a_degenerate_schedule():
    n = _node()
    assert n._anchor_shift_split_sigmas(torch.tensor([1.0, 0.0]), 0.5, 0.0) is None
    assert n._anchor_shift_split_sigmas(None, 0.9, 0.0) is None


# ── the slide ───────────────────────────────────────────────────────────────

def test_slide_drops_the_head_without_shortening_the_clip():
    latent = _latent(frames=10)
    shifted, dropped = _node()._anchor_shift_latent(latent, 2, "extend_last", False)
    assert dropped == 2
    v = shifted["samples"]
    # Length preserved — the feature must never cost extra generated frames.
    assert v.shape[2] == 10
    # Frames 2..9 slid to 0..7; the anchor (frame 0) is gone from the clip entirely.
    assert [float(v[0, 0, i, 0, 0]) for i in range(8)] == [2., 3., 4., 5., 6., 7., 8., 9.]
    # ...and the freed tail continues from the last real frame.
    assert [float(v[0, 0, i, 0, 0]) for i in (8, 9)] == [9., 9.]


def test_wrap_tail_puts_the_dropped_head_at_the_end():
    shifted, _ = _node()._anchor_shift_latent(_latent(frames=10), 2, "wrap", False)
    v = shifted["samples"]
    assert [float(v[0, 0, i, 0, 0]) for i in (8, 9)] == [0., 1.]


def test_empty_tail_zeroes_the_freed_region():
    shifted, _ = _node()._anchor_shift_latent(_latent(frames=10), 3, "empty", False)
    v = shifted["samples"]
    assert torch.equal(v[:, :, -3:], torch.zeros_like(v[:, :, -3:]))
    assert float(v[0, 0, 0, 0, 0]) == 3.0


def test_slide_drops_the_pin_with_the_frame_it_belonged_to():
    latent = _latent(frames=10)
    assert "noise_mask" in latent
    shifted, _ = _node()._anchor_shift_latent(latent, 1, "extend_last", False)
    # Pass 2 must run unpinned — the pinned frame no longer exists.
    assert "noise_mask" not in shifted


def test_slide_never_consumes_the_whole_clip():
    shifted, dropped = _node()._anchor_shift_latent(_latent(frames=4), 99, "extend_last", False)
    assert dropped == 3  # clamped to frames - 1
    assert shifted["samples"].shape[2] == 4


def test_zero_drop_is_a_no_op():
    shifted, dropped = _node()._anchor_shift_latent(_latent(frames=6), 0, "extend_last", True)
    assert dropped == 0
    assert torch.equal(shifted["samples"], _latent(frames=6)["samples"])


def test_source_latent_is_never_mutated():
    latent = _latent(frames=8)
    before = latent["samples"].clone()
    _node()._anchor_shift_latent(latent, 3, "extend_last", True)
    assert torch.equal(latent["samples"], before)
