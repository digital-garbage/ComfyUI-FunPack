"""anchor_shift — "fake t2v": run with the real untouched i2v anchor, then delete it.

The point of the feature is that the anchor is NOT weakened on the way in (ALG blurs it and
loses character detail; Best-FaceID overlap tokens approximate it and lose some too) — it is
pinned at full strength for pass 1 and then removed from the clip entirely. So the invariants
worth testing are about the shift itself:

0. Pass 1 stops at anchor_shift_sigma — ALWAYS. That cut is the feature; no other setting
   may quietly override it (an earlier fix let the restart sigma do exactly that, which
   made the shift sigma inert without saying so).
1. The schedule is cut into two passes that each still have a real step, or refused outright.
1b. A restart above the cut re-noises the half-denoised latent back up to it correctly —
   rebuilding x at the higher sigma, not re-noising it as if it were a finished image.
2. The slide drops the head WITHOUT shortening the clip — no extra frames are ever generated.
3. The pin is dropped with the frame it belonged to (pass 2 must run unpinned).
4. Audio is reset when asked, since pass 1's audio was formed against frames that moved.
"""
import math
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
    first, second, cut_at, restart_at, resume, start_idx = _node()._anchor_shift_split_sigmas(
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


def test_rewind_mode_still_stops_pass_one_at_the_cut():
    """The cut sigma is the whole feature — rewind must NOT override it. Pass 1 stops
    where the user asked, and the restart only decides where pass 2 re-enters."""
    first, second, cut_at, restart_at, resume, start_idx = _node()._anchor_shift_split_sigmas(
        SCHEDULE, 0.725, 0.975)
    assert abs(cut_at - 0.725) < 1e-5 and abs(restart_at - 0.975) < 1e-5
    assert [round(v, 3) for v in first.tolist()] == [1.0, 0.975, 0.909, 0.725]
    assert [round(v, 3) for v in second.tolist()] == [0.975, 0.909, 0.725, 0.422, 0.0]
    # resume=False means the caller must re-noise 0.725 -> 0.975 before pass 2 runs.
    assert resume is False


def test_rewind_refuses_a_restart_below_the_cut():
    """Pass 2 can be handed a NOISIER latent, never a cleaner one — there is no way to
    remove noise from the state pass 1 stopped at."""
    assert _node()._anchor_shift_split_sigmas(SCHEDULE, 0.909, 0.422) is None


def test_restart_landing_on_the_cut_is_just_continue():
    """A restart that snaps to the cut's own step has nothing to re-noise."""
    _, _, cut_at, restart_at, resume, _ = _node()._anchor_shift_split_sigmas(
        SCHEDULE, 0.909, 0.909)
    assert abs(cut_at - restart_at) < 1e-5
    assert resume is True


# ── the sampler must still see the REAL schedule ────────────────────────────

# The user's own 9-step schedule, used here because the boundary walk has to be right on
# the schedule they actually run, not on the toy one above.
USER_SCHEDULE = torch.tensor(
    [1.0, 0.955, 0.893, 0.812, 0.715, 0.603, 0.482, 0.241, 0.121, 0.0])


def test_cut_is_the_step_whose_next_sigma_crosses_the_threshold():
    """Walk the real schedule: the last pre-shift step is the one whose NEXT sigma has
    crossed the shift point. At shift 0.482 the steps at 1.000 and 0.955 both have a next
    sigma still above it and proceed; the step at 0.603 has next=0.482, so it is the last."""
    first, second, cut_at, restart_at, resume, start_idx = _node()._anchor_shift_split_sigmas(
        USER_SCHEDULE, 0.482, 0.812)
    assert [round(v, 3) for v in first.tolist()] == [1.0, 0.955, 0.893, 0.812, 0.715, 0.603, 0.482]
    assert abs(cut_at - 0.482) < 1e-5
    # Pass 2 re-enters at 0.812, which is index 3 of the real schedule.
    assert start_idx == 3
    assert abs(restart_at - 0.812) < 1e-5
    assert [round(v, 3) for v in second.tolist()] == [0.812, 0.715, 0.603, 0.482, 0.241, 0.121, 0.0]
    assert resume is False


def test_schedule_view_hands_the_sampler_the_whole_schedule():
    """Without this the sampler measures its phases from the SLICE: the final-correction
    window reopens at the end of pass 1, the AB2 ramp restarts, and the velocity-bias
    ratio is taken against the slice's own first sigma instead of the schedule's."""
    slice_ = USER_SCHEDULE[3:]
    sched, offset = samplers._schedule_view(
        slice_, {"sigmas": USER_SCHEDULE, "offset": 3})
    assert sched is USER_SCHEDULE and offset == 3
    # Concretely: 9 real steps, not the 6 the slice would have claimed.
    assert len(sched) - 1 == 9 and len(slice_) - 1 == 6


def test_schedule_view_falls_back_to_the_sigmas_it_was_given():
    """Every normal single-call run passes no context — `sigmas` IS the whole schedule."""
    for ctx in (None, {}, {"sigmas": None}, {"sigmas": torch.tensor([1.0])}, "nonsense"):
        sched, offset = samplers._schedule_view(USER_SCHEDULE, ctx)
        assert sched is USER_SCHEDULE and offset == 0


def test_schedule_view_never_returns_a_negative_offset():
    sched, offset = samplers._schedule_view(
        USER_SCHEDULE, {"sigmas": USER_SCHEDULE, "offset": -5})
    assert offset == 0


# ── the mid-trajectory re-noise ─────────────────────────────────────────────

class _FakeNested(list):
    """Minimal stand-in for comfy's NestedTensor: enough for _is_nested + unbind."""
    is_nested = True

    def unbind(self):
        return list(self)


@pytest.fixture
def renoise_stubs():
    """Deterministic noise (a constant 7.0) plus a usable NestedTensor.

    Snapshot/restore for the same reason the fixture below does it: these comfy stubs are
    shared module objects, and mutations left behind change other modules' results.
    """
    mod = sys.modules["comfy.sample"]
    comfy_mod = sys.modules["comfy"]
    nested_mod = sys.modules["comfy.nested_tensor"]
    saved = (getattr(mod, "prepare_noise", None), getattr(comfy_mod, "sample", None),
             nested_mod.NestedTensor, getattr(comfy_mod, "nested_tensor", None))

    def prepare_noise(samples, seed, *a, **k):
        if getattr(samples, "is_nested", False):
            return _FakeNested([torch.full_like(t, 7.0) for t in samples.unbind()])
        return torch.full_like(samples, 7.0)

    mod.prepare_noise = prepare_noise
    comfy_mod.sample = mod
    comfy_mod.nested_tensor = nested_mod  # attribute access, not just sys.modules
    nested_mod.NestedTensor = _FakeNested
    yield
    if saved[0] is None:
        delattr(mod, "prepare_noise")
    else:
        mod.prepare_noise = saved[0]
    if saved[1] is None:
        delattr(comfy_mod, "sample")
    else:
        comfy_mod.sample = saved[1]
    nested_mod.NestedTensor = saved[2]
    if saved[3] is None:
        delattr(comfy_mod, "nested_tensor")
    else:
        comfy_mod.nested_tensor = saved[3]


def test_renoise_to_the_same_sigma_is_an_identity(renoise_stubs):
    """b == a must add nothing: alpha=1, beta=0. This is what lets continue mode and
    rewind mode be one formula instead of two special cases."""
    latent = {"samples": torch.full((1, 4, 6, 2, 2), 3.0)}
    out = _node()._anchor_shift_renoise(latent, 0.725, 0.725, seed=7)
    assert torch.allclose(out["samples"], latent["samples"], atol=1e-6)


def test_renoise_from_zero_matches_comfy_const_scaling(renoise_stubs):
    """a == 0 (a finished latent) must reproduce comfy's own x = b*noise + (1-b)*x0, so
    the formula strictly generalises the img2img re-noise rather than replacing it."""
    b = 0.975
    x0 = torch.full((1, 4, 6, 2, 2), 3.0)
    out = _node()._anchor_shift_renoise({"samples": x0}, 0.0, b, seed=11)
    expected = _const_noise_scaling(b, torch.full_like(x0, 7.0), x0)
    assert torch.allclose(out["samples"], expected, atol=1e-6)


def test_renoise_upward_rebuilds_the_state_at_the_higher_sigma(renoise_stubs):
    """The bug this feature shipped with, fixed properly. Pass 1 stops at 0.603 and pass 2
    restarts at 0.893: x0's coefficient must land on (1-b) EXACTLY and the noise term must
    total b — anything else is the under-denoising from the first live run coming back."""
    a, b = 0.603, 0.893
    x0, eps = torch.full((1, 4, 6, 2, 2), 3.0), torch.full((1, 4, 6, 2, 2), -1.5)
    x_a = x0 * (1.0 - a) + eps * a          # a genuine mid-trajectory latent
    out = _node()._anchor_shift_renoise({"samples": x_a}, a, b, seed=3)

    alpha = (1.0 - b) / (1.0 - a)
    beta = math.sqrt(b * b - (a * alpha) ** 2)
    assert abs(alpha * (1.0 - a) - (1.0 - b)) < 1e-9   # picture kept at full strength
    assert abs((alpha * a) ** 2 + beta ** 2 - b * b) < 1e-9  # noise topped back up to b
    assert torch.allclose(out["samples"], x_a * alpha + 7.0 * beta, atol=1e-6)
    # And the naive path is measurably NOT this — the whole point of the fix.
    assert not torch.allclose(out["samples"],
                              _const_noise_scaling(b, torch.full_like(x_a, 7.0), x_a), atol=1e-3)


def test_renoise_starts_a_reset_stream_from_scratch(renoise_stubs):
    """A stream wiped by fresh_audio is not mid-trajectory — topping it up as if it were
    would start it quieter than a from-scratch run does. It gets the full b*noise."""
    b = 0.893
    latent = {"samples": _FakeNested(
        [torch.full((1, 4, 6, 2, 2), 2.0), torch.zeros(1, 4, 12)])}
    out = _node()._anchor_shift_renoise(latent, 0.603, b, seed=5, reset_indices=(1,))
    video, audio = out["samples"].unbind()
    assert torch.allclose(audio, torch.full_like(audio, 7.0 * b), atol=1e-6)
    # ...and the video stream still took the mid-trajectory path, not the a=0 one.
    alpha = (1.0 - b) / (1.0 - 0.603)
    expected = 2.0 * alpha + 7.0 * math.sqrt(b * b - (0.603 * alpha) ** 2)
    assert torch.allclose(video, torch.full_like(video, expected), atol=1e-6)


def test_renoise_never_mutates_its_input(renoise_stubs):
    latent = {"samples": torch.full((1, 4, 6, 2, 2), 3.0)}
    before = latent["samples"].clone()
    _node()._anchor_shift_renoise(latent, 0.603, 0.893, seed=1)
    assert torch.equal(latent["samples"], before)


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


# ── tail=crop: no regrown tail, the clip just gets shorter ──────────────────

def _av_latent(video_frames=10, audio_frames=20):
    """A packed-style AV latent: video [B,C,F,H,W] + audio [B,C,T] at its own rate."""
    video = torch.arange(video_frames, dtype=torch.float32).view(
        1, 1, video_frames, 1, 1).repeat(1, 4, 1, 2, 2)
    audio = torch.arange(audio_frames, dtype=torch.float32).view(1, 1, audio_frames).repeat(1, 4, 1)
    return {"samples": _FakeNested([video, audio]), "noise_mask": torch.zeros_like(video)}


def test_crop_shortens_the_clip_instead_of_regrowing_a_tail(renoise_stubs):
    shifted, dropped = _node()._anchor_shift_latent(_latent(frames=10), 3, "crop", False)
    assert dropped == 3
    # 10 - 3, not 10: nothing is invented at the end, which is the whole point of the mode.
    assert shifted["samples"].shape[2] == 7
    # And what survives is the ORIGINAL frames 3..9, untouched — no blend, no joint.
    assert [int(v) for v in shifted["samples"][0, 0, :, 0, 0]] == [3, 4, 5, 6, 7, 8, 9]


def test_crop_keeps_audio_and_video_describing_the_same_duration(renoise_stubs):
    """Audio timing comes from the audio stream's own index and video's from its own RoPE,
    so cropping them by different amounts of TIME desyncs the clip silently — no error,
    just drifting sound. The kept lengths must stay in proportion."""
    shifted, _ = _node()._anchor_shift_latent(_av_latent(10, 20), 3, "crop", False)
    video, audio = shifted["samples"].unbind()
    assert video.shape[2] == 7 and audio.shape[2] == 14      # 7/10 == 14/20
    # Cropped from the HEAD, matching the video slide — the tail is what survives.
    assert [int(v) for v in audio[0, 0]] == list(range(6, 20))


def test_crop_keeps_proportion_when_the_rate_does_not_divide_evenly(renoise_stubs):
    """The rounding trap: deriving audio's KEPT length from the new video length keeps the
    two in proportion; subtracting a separately-rounded drop can leave them a frame apart."""
    for v_frames, a_frames, drop in ((10, 15, 3), (9, 14, 4), (13, 7, 5), (10, 21, 7)):
        shifted, _ = _node()._anchor_shift_latent(_av_latent(v_frames, a_frames), drop, "crop", False)
        video, audio = shifted["samples"].unbind()
        assert audio.shape[2] == max(1, round(a_frames * video.shape[2] / v_frames))
        assert audio.shape[2] >= 1 and video.shape[2] >= 1


def test_crop_still_resets_audio_when_asked_at_the_cropped_length(renoise_stubs):
    """fresh_audio zeroes the stream AFTER the crop, so the reset length is the new
    duration — zeroing first would leave the clip's audio length describing the old one."""
    shifted, _ = _node()._anchor_shift_latent(_av_latent(10, 20), 4, "crop", True)
    video, audio = shifted["samples"].unbind()
    assert video.shape[2] == 6 and audio.shape[2] == 12
    assert torch.count_nonzero(audio) == 0


def test_the_other_tail_modes_still_keep_the_clip_length(renoise_stubs):
    """crop is an OPTION — the three refilling modes must be untouched by it."""
    for mode in ("extend_last", "wrap", "empty"):
        shifted, _ = _node()._anchor_shift_latent(_av_latent(10, 20), 3, mode, False)
        video, audio = shifted["samples"].unbind()
        assert video.shape[2] == 10 and audio.shape[2] == 20, mode


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
