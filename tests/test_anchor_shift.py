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

def test_continue_mode_stops_at_the_cut_and_resumes_from_that_state():
    first, second, cut_at, restart_at, resume = _node()._anchor_shift_split_sigmas(
        SCHEDULE, 0.909, 0.0)
    # float32: the schedule stores 0.909 as 0.90899997, so compare loosely.
    assert abs(cut_at - 0.909) < 1e-5 and abs(restart_at - 0.909) < 1e-5
    # pass 1 ends ON the cut sigma; pass 2 resumes from it, so no step is run twice.
    assert [round(v, 3) for v in first.tolist()] == [1.0, 0.975, 0.909]
    assert [round(v, 3) for v in second.tolist()] == [0.909, 0.725, 0.422, 0.0]
    # resume=True is load-bearing: pass 2 receives a MID-TRAJECTORY latent, and comfy's
    # CONST scaling (x = s*noise + (1-s)*latent) would otherwise treat it as a clean image
    # and re-noise it — scaling the picture to (1-s) under a fresh s of noise. That is the
    # bug that made the first live run come out under-denoised.
    assert resume is True


def test_rewind_mode_runs_pass_one_to_completion_first():
    """A re-noise to a HIGHER sigma is only meaningful on a clean latent, so rewind mode
    must hand pass 2 a finished clip — not a mid-trajectory one."""
    first, second, cut_at, restart_at, resume = _node()._anchor_shift_split_sigmas(
        SCHEDULE, 0.725, 0.975)
    assert abs(cut_at - 0.725) < 1e-5 and abs(restart_at - 0.975) < 1e-5
    assert [round(v, 3) for v in first.tolist()] == [1.0, 0.975, 0.909, 0.725, 0.422, 0.0]
    assert [round(v, 3) for v in second.tolist()] == [0.975, 0.909, 0.725, 0.422, 0.0]
    assert resume is False  # clean input -> re-noising is correct here


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


# ── the t2v guard ───────────────────────────────────────────────────────────

def test_pinned_frames_counts_the_anchor_prefix():
    n = _node()
    frames = 6
    video = torch.zeros(1, 4, frames, 2, 2)
    mask = torch.ones(1, 4, frames, 2, 2)
    mask[:, :, :2] = 0.0  # two pinned anchor frames
    assert n._anchor_pinned_frames({"samples": video, "noise_mask": mask}) == 2


def test_pinned_frames_is_zero_for_a_real_t2v_scene():
    """No anchor image attached — Easy Gen's default. Shifting would drop real content,
    so the caller must skip: this is the 'fake t2v' trick and it needs an anchor to fake with."""
    n = _node()
    video = torch.zeros(1, 4, 6, 2, 2)
    assert n._anchor_pinned_frames({"samples": video, "noise_mask": torch.ones(1, 4, 6, 2, 2)}) == 0
    assert n._anchor_pinned_frames({"samples": video}) == 0  # no mask at all
    assert n._anchor_pinned_frames({"samples": None}) == 0   # unreadable -> refuse, don't raise


# ── the re-noise bug this feature shipped with, locked down ─────────────────

@pytest.fixture
def capture_sample_custom():
    """Stub comfy.sample so we can see exactly what noise/latent the sampler receives.

    Snapshot/restore, because these comfy stubs are SHARED module objects across every test
    module — leaving mutations behind changes other files' results depending on collection
    order (the same trap test_detailing.py hit with comfy.nested_tensor).
    """
    mod = sys.modules["comfy.sample"]
    comfy_mod = sys.modules["comfy"]
    saved = (getattr(mod, "prepare_noise", None), getattr(mod, "sample_custom", None),
             getattr(comfy_mod, "sample", None))
    seen = {}

    def prepare_noise(samples, seed, *a, **k):
        return torch.full_like(samples, 7.0)  # unmistakably "fresh noise"

    def sample_custom(model, noise, cfg, sampler, sigmas, pos, neg, latent_image, **kw):
        seen["noise"] = noise
        seen["latent_image"] = latent_image
        return latent_image

    mod.prepare_noise = prepare_noise
    mod.sample_custom = sample_custom
    sys.modules["comfy"].sample = mod  # attribute access, not just sys.modules
    yield seen
    for name, val in (("prepare_noise", saved[0]), ("sample_custom", saved[1])):
        if val is None:
            delattr(mod, name)
        else:
            setattr(mod, name, val)
    if saved[2] is None:
        delattr(comfy_mod, "sample")
    else:
        comfy_mod.sample = saved[2]


def _const_noise_scaling(sigma, noise, latent_image):
    """comfy's CONST rule, verbatim: x = s*noise + (1-s)*latent_image."""
    return sigma * noise + (1.0 - sigma) * latent_image


def test_continue_from_state_makes_const_scaling_an_identity(capture_sample_custom):
    """The real fix. Pass 2 gets a MID-TRAJECTORY latent; handing it as both terms makes
    comfy's scaling collapse to x, so sampling resumes exactly where pass 1 stopped."""
    seen = capture_sample_custom
    state = torch.randn(1, 4, 5, 2, 2)
    _node()._sample_chunk(object(), object(), torch.tensor([0.725, 0.422, 0.0]), 1, 1.0,
                          [], [], {"samples": state}, continue_from_state=True)
    assert torch.equal(seen["noise"], seen["latent_image"])
    resumed = _const_noise_scaling(0.725, seen["noise"], seen["latent_image"])
    assert torch.allclose(resumed, state, atol=1e-6)


def test_without_continue_from_state_the_latent_is_renoised(capture_sample_custom):
    """The other mode, and the shape of the original bug: fresh noise is mixed in and the
    picture is scaled to (1-sigma). Correct for a CLEAN input, wrong mid-trajectory."""
    seen = capture_sample_custom
    clean = torch.randn(1, 4, 5, 2, 2)
    _node()._sample_chunk(object(), object(), torch.tensor([0.725, 0.422, 0.0]), 1, 1.0,
                          [], [], {"samples": clean}, continue_from_state=False)
    assert not torch.equal(seen["noise"], seen["latent_image"])
    entered = _const_noise_scaling(0.725, seen["noise"], seen["latent_image"])
    assert not torch.allclose(entered, clean, atol=1e-3)
