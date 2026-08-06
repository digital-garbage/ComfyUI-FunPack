"""The MiniMax H3 audio clock: integrating the audio stream on its own flow schedule.

H3 denoises video and audio on two different flow schedules but hands the sampler a single
sigma grid, so comfy's DiT reconciles them by scaling the audio velocity by the slope
between the schedules at the START of each step. That is exact only in the limit; on a
few-step schedule the start-of-step slope badly overshoots the chord it stands in for and
the audio is driven past where its own schedule puts it.

These tests pin the arithmetic (which is the whole mechanism — there is no model call in it)
and the two properties that make it safe to leave wired: it touches the audio region only,
and it collapses to a no-op wherever it cannot be applied.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3 import (  # noqa: E402
    DEFAULT_SHIFT_AUDIO,
    DEFAULT_SHIFT_VIDEO,
    SHIFT_AUDIO_OPTION,
    SHIFT_VIDEO_OPTION,
    audio_clock_factors,
    resolve_sigma_shifts,
    time_shift_sigma,
    time_shift_slope,
)


# The schedule a shift-12 flow model produces for 4 steps, and the audio schedule the DiT
# derives from it. Recomputed here from the closed form rather than pasted, so the fixture
# stays honest if the map ever changes.
FOUR_STEP_VIDEO = [1.0, 0.973, 0.923, 0.8, 0.0]


def audio_grid(sigmas):
    return [time_shift_sigma(s, DEFAULT_SHIFT_VIDEO, DEFAULT_SHIFT_AUDIO) for s in sigmas]


# ── the schedule map ─────────────────────────────────────────────────────────

def test_endpoints_are_shared_by_both_schedules():
    """sigma 0 and 1 mean the same thing on any shift — only the interior is warped."""
    for shift in (3.0, 12.0, 0.5):
        assert time_shift_sigma(0.0, DEFAULT_SHIFT_VIDEO, shift) == pytest.approx(0.0)
        assert time_shift_sigma(1.0, DEFAULT_SHIFT_VIDEO, shift) == pytest.approx(1.0)


def test_mapping_to_the_same_shift_is_the_identity():
    for sigma in (0.1, 0.5, 0.8, 0.973, 1.0):
        assert time_shift_sigma(sigma, 12.0, 12.0) == pytest.approx(sigma)
        assert time_shift_slope(sigma, 12.0, 12.0) == pytest.approx(1.0)


def test_audio_schedule_runs_ahead_of_the_video_one():
    """The whole problem in one assertion: at the same point on the shared base grid the
    audio stream is already much further denoised than the video stream."""
    for sigma in (0.5, 0.8, 0.923, 0.973):
        assert time_shift_sigma(sigma, DEFAULT_SHIFT_VIDEO, DEFAULT_SHIFT_AUDIO) < sigma


def test_slope_is_the_derivative_of_the_map():
    """The slope function must be the actual derivative of the sigma map, since the DiT
    uses one and the correction divides by it."""
    eps = 1e-6
    for sigma in (0.2, 0.5, 0.8, 0.95):
        numeric = (time_shift_sigma(sigma + eps, 12.0, 3.0)
                   - time_shift_sigma(sigma - eps, 12.0, 3.0)) / (2 * eps)
        assert time_shift_slope(sigma, 12.0, 3.0) == pytest.approx(numeric, rel=1e-4)


# ── the per-step correction ──────────────────────────────────────────────────

def test_factors_land_the_audio_exactly_on_its_own_grid():
    """The defining property: applying the factor to the tangent-scaled step reproduces
    the audio's own schedule step exactly, for every step of the schedule."""
    factors = audio_clock_factors(FOUR_STEP_VIDEO)
    audio = audio_grid(FOUR_STEP_VIDEO)
    assert len(factors) == len(FOUR_STEP_VIDEO) - 1
    for i, factor in enumerate(factors):
        slope = time_shift_slope(FOUR_STEP_VIDEO[i], DEFAULT_SHIFT_VIDEO, DEFAULT_SHIFT_AUDIO)
        uncorrected = slope * (FOUR_STEP_VIDEO[i + 1] - FOUR_STEP_VIDEO[i])
        assert uncorrected * factor == pytest.approx(audio[i + 1] - audio[i], rel=1e-9)


def test_every_factor_shortens_the_audio_step_at_the_stock_shifts():
    """With video 12 / audio 3 the map is convex and the tangent is taken at the step's
    high-sigma end, so the uncorrected step always OVERSHOOTS. A factor above 1 here would
    mean the correction is amplifying the very error it exists to remove."""
    for factor in audio_clock_factors(FOUR_STEP_VIDEO):
        assert 0.0 < factor <= 1.0


def test_the_final_step_is_the_worst_one():
    """The step to sigma 0 is both the largest and the most warped — the one that shows up
    as audible distortion. Pinning the number keeps a refactor from quietly softening it."""
    factors = audio_clock_factors(FOUR_STEP_VIDEO)
    assert factors[-1] == pytest.approx(0.40, abs=0.01)
    assert factors[-1] == min(factors)


def test_correction_fades_out_as_steps_get_smaller():
    """Why this is a few-step setting: on a fine schedule the tangent and the chord agree,
    so the factors approach 1 and the mechanism stops changing anything."""
    fine = [1.0 - i / 200.0 for i in range(201)]
    assert max(abs(f - 1.0) for f in audio_clock_factors(fine)) < 0.05
    coarse = audio_clock_factors(FOUR_STEP_VIDEO)
    assert max(abs(f - 1.0) for f in coarse) > 0.4


def test_equal_shifts_make_every_factor_one():
    """Both streams on one schedule: nothing to correct, and the caller uses this to skip."""
    factors = audio_clock_factors(FOUR_STEP_VIDEO, shift_video=8.0, shift_audio=8.0)
    assert factors == pytest.approx([1.0] * (len(FOUR_STEP_VIDEO) - 1))


def test_repeated_sigmas_do_not_divide_by_zero():
    factors = audio_clock_factors([1.0, 1.0, 0.5, 0.5, 0.0])
    assert all(f == f for f in factors)  # no NaN
    assert factors[0] == 1.0 and factors[2] == 1.0


def test_a_schedule_too_short_to_step_yields_no_factors():
    assert audio_clock_factors([1.0]) == []
    assert audio_clock_factors([]) == []


def test_shifts_the_other_way_round_lengthen_the_step():
    """An audio shift ABOVE the video shift inverts the curvature, so the correct factor is
    greater than 1. Clamping that back to 1 would neuter the setting for those shifts."""
    factors = audio_clock_factors(FOUR_STEP_VIDEO, shift_video=3.0, shift_audio=12.0)
    assert max(factors) > 1.0


# ── reading the shifts in force ──────────────────────────────────────────────

def test_shifts_come_from_the_sigma_shift_node_when_it_is_wired():
    options = {"transformer_options": {SHIFT_VIDEO_OPTION: 7.5, SHIFT_AUDIO_OPTION: 2.25}}
    assert resolve_sigma_shifts(options) == (7.5, 2.25)


def test_shifts_fall_back_to_the_models_own_defaults():
    """Without the MiniMaxH3SigmaShift node the DiT uses 12/3, so the correction must too —
    reading 1/1 here would silently disable it on the most common wiring of all."""
    for options in (None, {}, {"transformer_options": {}}, {"transformer_options": None}):
        assert resolve_sigma_shifts(options) == (DEFAULT_SHIFT_VIDEO, DEFAULT_SHIFT_AUDIO)


def test_unusable_shift_values_fall_back_rather_than_raise():
    options = {"transformer_options": {SHIFT_VIDEO_OPTION: "twelve", SHIFT_AUDIO_OPTION: None}}
    assert resolve_sigma_shifts(options) == (DEFAULT_SHIFT_VIDEO, DEFAULT_SHIFT_AUDIO)


# ── applying it to a packed latent ───────────────────────────────────────────

def _clock(factors, video_dims, audio_dims):
    mask = torch.zeros((1, 1, video_dims + audio_dims))
    mask[..., :video_dims] = 1.0
    return factors, mask


def test_only_the_audio_region_is_rescaled():
    from samplers import _audio_clock_step

    clock = _clock([0.4], video_dims=3, audio_dims=2)
    x_old = torch.zeros((1, 1, 5))
    x_new = torch.ones((1, 1, 5))          # displacement of 1.0 everywhere
    out = _audio_clock_step(x_new, x_old, clock, 0)
    assert out[0, 0, :3].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert out[0, 0, 3:].tolist() == pytest.approx([0.4, 0.4])


def test_the_direction_of_the_audio_step_is_preserved():
    """It shortens the step; it must never flip or rotate it."""
    from samplers import _audio_clock_step

    clock = _clock([0.4], video_dims=1, audio_dims=3)
    x_old = torch.zeros((1, 1, 4))
    x_new = torch.tensor([[[1.0, 2.0, -4.0, 0.5]]])
    out = _audio_clock_step(x_new, x_old, clock, 0)
    assert out[0, 0, 1:].tolist() == pytest.approx([0.8, -1.6, 0.2])


def test_each_step_uses_its_own_factor():
    from samplers import _audio_clock_step

    clock = _clock([0.9, 0.5], video_dims=1, audio_dims=1)
    x_old = torch.zeros((1, 1, 2))
    x_new = torch.ones((1, 1, 2))
    assert _audio_clock_step(x_new, x_old, clock, 0)[0, 0, 1].item() == pytest.approx(0.9)
    assert _audio_clock_step(x_new, x_old, clock, 1)[0, 0, 1].item() == pytest.approx(0.5)


def test_without_a_clock_the_step_passes_through_untouched():
    """The off path has to be exactly the old behaviour, not an approximation of it."""
    from samplers import _audio_clock_step

    x_old = torch.zeros((1, 1, 4))
    x_new = torch.rand((1, 1, 4))
    assert torch.equal(_audio_clock_step(x_new, x_old, None, 0), x_new)


def test_a_step_past_the_end_of_the_schedule_passes_through():
    """A sampler that runs more steps than the sigma list describes must not be corrected
    with a factor from the wrong step — it gets none."""
    from samplers import _audio_clock_step

    clock = _clock([0.4], video_dims=1, audio_dims=1)
    x_old = torch.zeros((1, 1, 2))
    x_new = torch.ones((1, 1, 2))
    assert torch.equal(_audio_clock_step(x_new, x_old, clock, 5), x_new)


def test_a_malformed_clock_never_breaks_sampling():
    """This runs inside the step loop; a bad shape must cost the correction, not the run."""
    from samplers import _audio_clock_step

    x_old = torch.zeros((1, 1, 4))
    x_new = torch.ones((1, 1, 4))
    bad = ([0.4], torch.ones((1, 1, 99)))   # mask that does not match x
    assert torch.equal(_audio_clock_step(x_new, x_old, bad, 0), x_new)


# ── setup refuses rather than pretending ─────────────────────────────────────

def test_setup_is_skipped_entirely_when_the_toggle_is_off():
    from samplers import _audio_clock_setup

    assert _audio_clock_setup(object(), None, [1.0, 0.0], False) is None


def test_setup_declines_off_h3_instead_of_raising():
    """The toggle is reachable on any pipeline, so a non-H3 model has to be an ordinary
    'not running' rather than an exception out of the middle of a generation."""
    from samplers import _audio_clock_setup

    assert _audio_clock_setup(object(), torch.zeros((1, 1, 4)), [1.0, 0.5, 0.0], True) is None


# ── reaching the sampler from the Chain Sampler node ─────────────────────────

def test_every_listed_sampler_actually_accepts_the_option():
    """_sample_chunk pushes h3_audio_clock into extra_options for exactly the functions in
    this list, and comfy's KSAMPLER splats extra_options as kwargs — so a name in the list
    that does not take the kwarg is a TypeError mid-generation, not a quiet no-op."""
    import inspect

    from samplers import _AUDIO_CLOCK_SAMPLERS

    assert _AUDIO_CLOCK_SAMPLERS, "the list must not be empty or the toggle can never run"
    for fn in _AUDIO_CLOCK_SAMPLERS:
        assert "h3_audio_clock" in inspect.signature(fn).parameters, fn.__name__


def test_the_option_defaults_to_off_in_every_sampler():
    """These functions are called directly by users wiring the sampler nodes, without the
    Chain Sampler ever setting the key. Defaulting to anything but off would turn the
    correction on for people who never asked for it."""
    import inspect

    from samplers import _AUDIO_CLOCK_SAMPLERS

    for fn in _AUDIO_CLOCK_SAMPLERS:
        assert inspect.signature(fn).parameters["h3_audio_clock"].default is False, fn.__name__


def test_the_chain_sampler_exposes_the_toggle_and_defaults_it_off():
    from samplers import FunPackLTXAVSceneChainSampler

    spec = FunPackLTXAVSceneChainSampler.INPUT_TYPES()["optional"]["h3_audio_clock"]
    assert spec[0] == "BOOLEAN"
    assert spec[1]["default"] is False


# ── wrapping a sampler we do not own ─────────────────────────────────────────

class FakeSampler:
    """Stands in for a KSAMPLER from KSamplerSelect."""
    def __init__(self, fn, extra_options=None):
        self.sampler_function = fn
        self.extra_options = extra_options or {}
        self.inpaint_options = {}


class RecordingDenoiser:
    """A model_k stand-in: records (x, sigma) per call and returns a fixed x0."""
    def __init__(self, x0):
        self.x0 = x0
        self.calls = []
        self.inner_model = "sentinel"      # samplers reach through for this

    def __call__(self, x, sigma, **kwargs):
        self.calls.append((x.clone(), float(sigma.flatten()[0])))
        return self.x0.clone()


def _proxy(inner, factors, video_dims, audio_dims, sigmas):
    from samplers import _AudioClockDenoiser

    return _AudioClockDenoiser(inner, _clock(factors, video_dims, audio_dims), sigmas)


def test_the_proxy_corrects_the_prediction_not_the_step():
    """The in-loop version scales the displacement; from outside, the only lever is the
    denoised value. Scaling the audio part of (x - denoised) is the same correction, because
    the euler update is x*r + denoised*(1-r) — so this has to match exactly."""
    from samplers import _audio_clock_step

    x = torch.tensor([[[10.0, 10.0]]])          # [video, audio]
    x0 = torch.tensor([[[0.0, 0.0]]])
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    proxy = _proxy(RecordingDenoiser(x0), [0.4, 0.4], 1, 1, sigmas)
    corrected = proxy(x, torch.tensor([1.0]))

    r = 0.5 / 1.0                                # sigma_next / sigma
    got = x * r + corrected * (1 - r)            # what an euler sampler will do
    want = _audio_clock_step(x * r + x0 * (1 - r), x, _clock([0.4], 1, 1), 0)
    assert torch.allclose(got, want, atol=1e-6)


def test_the_proxy_walks_the_schedule_one_step_per_call():
    x = torch.zeros((1, 1, 2))
    sigmas = torch.tensor([1.0, 0.7, 0.3, 0.0])
    inner = RecordingDenoiser(torch.ones((1, 1, 2)))
    proxy = _proxy(inner, [0.2, 0.5, 0.9], 1, 1, sigmas)
    seen = [proxy(x, torch.tensor([float(s)]))[0, 0, 1].item() for s in (1.0, 0.7, 0.3)]
    # audio component of (denoised - x) scaled by that step's factor
    assert seen == pytest.approx([0.2, 0.5, 0.9])


def test_the_proxy_stands_down_on_a_sampler_that_evaluates_twice_per_step():
    """A Heun corrector call lands at sigmas[i+1] — the same signature as the next step's
    predictor. Rather than guess, the proxy must detect the mismatch and stop correcting."""
    x = torch.zeros((1, 1, 2))
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    proxy = _proxy(RecordingDenoiser(torch.ones((1, 1, 2))), [0.3, 0.3], 1, 1, sigmas)
    first = proxy(x, torch.tensor([1.0]))                 # predictor of step 0 — corrected
    assert first[0, 0, 1].item() == pytest.approx(0.3)
    mid = proxy(x, torch.tensor([0.75]))                  # an off-schedule eval
    assert mid[0, 0, 1].item() == pytest.approx(1.0)      # uncorrected
    after = proxy(x, torch.tensor([0.5]))                 # and it stays off
    assert after[0, 0, 1].item() == pytest.approx(1.0)


def test_the_proxy_is_transparent_for_everything_else():
    """Samplers read and write attributes on the denoiser (latent_image, inner_model);
    the proxy must not hide them or the sampler breaks."""
    inner = RecordingDenoiser(torch.zeros((1, 1, 2)))
    proxy = _proxy(inner, [0.5], 1, 1, torch.tensor([1.0, 0.0]))
    assert proxy.inner_model == "sentinel"
    proxy.latent_image = "written-through"
    assert inner.latent_image == "written-through"


def test_wrapping_preserves_the_samplers_own_options():
    from samplers import _audio_clock_wrap_sampler

    original = FakeSampler(lambda *a, **k: None, extra_options={"eta": 0.5})
    wrapped = _audio_clock_wrap_sampler(original)
    assert wrapped.extra_options == {"eta": 0.5}
    assert wrapped.sampler_function is not original.sampler_function


def test_a_sampler_with_nothing_to_wrap_is_declined():
    """Not every SAMPLER is KSAMPLER-shaped; guessing at internals is worse than declining."""
    from samplers import _audio_clock_wrap_sampler

    assert _audio_clock_wrap_sampler(object()) is None


def test_the_wrapped_sampler_runs_the_original_untouched_off_h3():
    """Off H3 the setup declines, and the original sampler must then see the REAL model
    object, not a proxy wrapped around it."""
    from samplers import _audio_clock_wrap_sampler

    seen = {}

    def fake_sampler_fn(model, x, sigmas, extra_args=None, callback=None, disable=None, **kw):
        seen["model"] = model
        return x

    inner = RecordingDenoiser(torch.zeros((1, 1, 2)))
    wrapped = _audio_clock_wrap_sampler(FakeSampler(fake_sampler_fn))
    out = wrapped.sampler_function(inner, torch.zeros((1, 1, 2)), torch.tensor([1.0, 0.0]))
    assert seen["model"] is inner          # no proxy inserted
    assert out.shape == (1, 1, 2)


def test_a_perfect_predictor_lands_the_audio_exactly_when_euler_steps_it():
    """End-to-end against a real euler loop, with an analytically known answer.

    Rectified flow is LINEAR along the true path, so a perfect x0 predictor integrated on the
    RIGHT grid must land exactly on x0 — any residue is the schedule error this feature
    removes. This reproduces the measured 85% -> 0% at 4 steps that the tooltip quotes, and
    is the one assertion that would catch the correction being silently inverted or dropped.
    """
    shift_v, shift_a = DEFAULT_SHIFT_VIDEO, DEFAULT_SHIFT_AUDIO
    x0 = torch.tensor([[[3.0, -2.0]]])                  # [video, audio]
    eps = torch.tensor([[[0.5, 1.5]]])
    span = float((eps - x0)[0, 0, 1].abs())

    steps = 4
    ts = [1.0 - i / steps for i in range(steps + 1)]
    sigmas = torch.tensor([shift_v * t / (1.0 + (shift_v - 1.0) * t) for t in ts])

    def model(x, sigma):
        """comfy's denoised = x - sigma_v * model_output, audio pre-scaled by the slope."""
        sv = float(sigma)
        v = eps - x0
        out = torch.empty_like(x)
        out[..., 0] = v[..., 0]
        out[..., 1] = time_shift_slope(sv, shift_v, shift_a) * v[..., 1]
        return x - sv * out

    def start():
        sv = float(sigmas[0])
        sa = time_shift_sigma(sv, shift_v, shift_a)
        x = torch.empty_like(x0)
        x[..., 0] = (1 - sv) * x0[..., 0] + sv * eps[..., 0]
        x[..., 1] = (1 - sa) * x0[..., 1] + sa * eps[..., 1]
        return x

    def euler(corrected):
        from samplers import _AudioClockDenoiser

        x = start()
        proxy = _AudioClockDenoiser(
            model, (audio_clock_factors(sigmas, shift_v, shift_a), torch.tensor([[[1.0, 0.0]]])),
            sigmas) if corrected else None
        for i in range(len(sigmas) - 1):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            denoised = proxy(x, sigma) if corrected else model(x, sigma)
            x = x + (x - denoised) / sigma * (sigma_next - sigma)
        return x

    off, on = euler(False), euler(True)
    # video is a plain flow on its own grid — right either way
    assert abs(float(off[0, 0, 0]) - float(x0[0, 0, 0])) < 1e-5
    assert abs(float(on[0, 0, 0]) - float(x0[0, 0, 0])) < 1e-5
    # audio: badly wrong without the clock, exact with it
    assert float((off[0, 0, 1] - x0[0, 0, 1]).abs()) / span > 0.5
    assert float((on[0, 0, 1] - x0[0, 0, 1]).abs()) / span < 1e-5


def test_the_documented_sampler_buckets_do_not_overlap():
    """The tooltip promises euler is exact and the multistep family is not. If a name ever
    lands in both lists the guidance the user reads is self-contradictory."""
    from samplers import _AUDIO_CLOCK_EXACT_ON, _AUDIO_CLOCK_HARMED

    assert not set(_AUDIO_CLOCK_EXACT_ON) & set(_AUDIO_CLOCK_HARMED)
    assert "sample_euler" in _AUDIO_CLOCK_EXACT_ON
    assert "sample_res_multistep" in _AUDIO_CLOCK_HARMED


def test_the_toggle_stays_behind_the_second_pass_sigmas_socket():
    """The builder maps a reference workflow's widget values positionally, so a new WIDGET
    has to be appended after the last widget but before the trailing SIGMAS socket — which
    is an input, not a widget, and shifts nothing."""
    from samplers import FunPackLTXAVSceneChainSampler

    names = list(FunPackLTXAVSceneChainSampler.INPUT_TYPES()["optional"])
    assert names.index("h3_audio_clock") == names.index("second_pass_sigmas") - 1
