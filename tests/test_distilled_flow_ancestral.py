"""Distilled Flow's ancestral noise is rectified-flow correct on H3 and LTXAV.

`s_noise` used to take the FULL deterministic step to sigma_next and then add
sqrt(sigma^2 - sigma_next^2) of noise on top. Both halves are wrong for a flow model: that
variance formula is the VP/eps one, and adding noise without shortening the step leaves the
latent noisier at sigma_next than the schedule says it should be. So `s_noise` never behaved
like `euler_ancestral` at any value, and the sampler could not reproduce the result the stock
ancestral sampler gives on H3.

The correct step lands SHORT (sigma_down) and renoises with flow matching's alpha rescaling —
the same arithmetic `_sample_const_rf_full` already used, and the same as comfy's
`sample_euler_ancestral_RF`.
"""
import inspect
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import samplers


def rf_step(x, denoised, sigma, sigma_next, eta, noise):
    """The formulation now in the sampler, transcribed for arithmetic checks."""
    downstep_ratio = 1 + (sigma_next / sigma - 1) * eta
    sigma_down = sigma_next * downstep_ratio
    alpha_ip1, alpha_down = 1 - sigma_next, 1 - sigma_down
    ratio = sigma_down / sigma
    x_anc = ratio * x + (1 - ratio) * denoised
    renoise = (sigma_next ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2) ** 0.5
    return (alpha_ip1 / alpha_down) * x_anc + noise * renoise


def test_the_sampler_uses_the_rf_arithmetic():
    src = inspect.getsource(samplers.sample_funpack_distilled_flow)
    for token in ("downstep_ratio = 1 + (sigma_next / sigma - 1) * s_noise",
                  "sigma_down = sigma_next * downstep_ratio",
                  "alpha_ip1 = 1 - sigma_next",
                  "renoise = (sigma_next ** 2"):
        assert token in src


def test_it_matches_the_hybrid_samplers_existing_rf_step():
    """The RF step already existed for CONST models — this is the same arithmetic, not a
    second opinion about it."""
    hybrid = inspect.getsource(samplers._sample_const_rf_full)
    flow = inspect.getsource(samplers.sample_funpack_distilled_flow)
    for token in ("downstep_ratio = 1 + (sigma_next / sigma - 1)",
                  "sigma_down = sigma_next * downstep_ratio",
                  "alpha_ip1 = 1 - sigma_next"):
        assert token in hybrid and token in flow


def test_eta_zero_is_the_deterministic_step():
    """No stochasticity means the ancestral branch must not run at all."""
    x, den, s, sn = 3.0, 1.0, 0.8, 0.5
    assert rf_step(x, den, s, sn, 0.0, 999.0) == pytest.approx(sn / s * x + (1 - sn / s) * den)


def test_eta_one_converts_the_whole_step():
    """At eta=1 sigma_down collapses to sigma_next^2/sigma — comfy's euler_ancestral_RF."""
    s, sn = 0.8, 0.5
    downstep_ratio = 1 + (sn / s - 1) * 1.0
    assert sn * downstep_ratio == pytest.approx(sn * sn / s)


def test_more_eta_lands_shorter():
    """The point of the fix: noise REPLACES deterministic progress rather than adding to it."""
    s, sn = 0.8, 0.5
    downs = [sn * (1 + (sn / s - 1) * e) for e in (0.0, 0.25, 0.5, 1.0)]
    assert downs == sorted(downs, reverse=True)
    assert downs[0] == pytest.approx(sn)          # eta=0 lands exactly on sigma_next


def test_the_old_formula_overshot_the_schedule():
    """Why it never matched: the old path kept the full step AND added noise, so the latent
    carried more variance at sigma_next than the schedule defines."""
    s, sn, eta = 0.8, 0.5, 0.15
    old_surplus = eta * (s ** 2 - sn ** 2) ** 0.5
    assert old_surplus > 0                         # added on top of an already-complete step
    new_down = sn * (1 + (sn / s - 1) * eta)
    assert new_down < sn                           # the new one gives ground back first


def test_audio_never_gets_ancestral_noise():
    src = inspect.getsource(samplers.sample_funpack_distilled_flow)
    assert "x = _video_only(x_anc, x_det, video_mask)" in src


def test_the_eps_path_is_untouched_for_non_flow_models():
    """LTXV and anything else that is not CONST keeps the behaviour it was tuned with."""
    src = inspect.getsource(samplers.sample_funpack_distilled_flow)
    assert "elif s_noise > 0.0 and float(sigma_next) > 0:" in src
    assert "sigma_up = math.sqrt(max(0.0, float(sigma.item()) ** 2" in src


def test_const_detection_never_raises():
    assert samplers._rf_ancestral(object()) is False
    assert samplers._rf_ancestral(None) is False


def test_const_detection_finds_a_flow_model(monkeypatch):
    """The suite stubs comfy.model_sampling, which has no CONST — so install one. That the
    stub lacks it is itself covered: the detector must return False, not raise."""
    import comfy.model_sampling as msmod
    class CONST: pass
    monkeypatch.setattr(msmod, "CONST", CONST, raising=False)
    class Flow(CONST): pass
    model = types.SimpleNamespace(
        inner_model=types.SimpleNamespace(
            inner_model=types.SimpleNamespace(model_sampling=Flow())))
    assert samplers._rf_ancestral(model) is True
    other = types.SimpleNamespace(
        inner_model=types.SimpleNamespace(
            inner_model=types.SimpleNamespace(model_sampling=object())))
    assert samplers._rf_ancestral(other) is False


def test_the_range_reaches_one_so_euler_ancestral_is_expressible():
    w = samplers.FunPackDistilledFlowSampler.INPUT_TYPES()["required"]["s_noise"][1]
    assert w["max"] == 1.0
    assert w["default"] == 0.0
    assert "BEHAVIOUR CHANGED" in w["tooltip"]
