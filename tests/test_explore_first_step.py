"""_select_best_seed: branch-then-commit seed search at the first step only.

See samplers.py's docstring on _select_best_seed for the mechanism. This isolates it from
the full sample() harness in test_scene_chain_sampler.py deliberately: that file's
fake_sample_custom never touches model.model_options["model_function_wrapper"] at all (it
computes its stand-in result directly from the inputs), so a full-sample() test would never
actually exercise the observer-wrapper capture this method depends on -- it would silently
pass while testing nothing. The fakes here call the installed wrapper directly, the way a
real one-step comfy.sample.sample_custom call would.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

import _comfy_stubs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_comfy_stubs.install_module("comfy.sample", prepare_noise=lambda *a, **k: None,
                            sample_custom=lambda *a, **k: None)

from samplers import FunPackLTXAVSceneChainSampler  # noqa: E402

SEED_OFFSET = 999983  # must match the offset _select_best_seed derives candidate seeds with


class _FakeValueFn:
    """Ranks a captured tensor by its own mean -- deterministic, easy to reason about,
    standing in for the real trained MLP. Real value functions are nn.Modules living on
    the CPU (LatentValueFunction.load uses map_location='cpu'); parameters() stands in for
    that so _select_best_seed's device-placement path (next(value_fn.parameters()).device)
    has something real to find, exactly like production."""
    def is_ready(self):
        return True

    def compress(self, x):
        return x

    def forward(self, x):
        return x.mean()

    def parameters(self):
        return iter([torch.zeros(1)])


class _NotReadyValueFn(_FakeValueFn):
    def is_ready(self):
        return False


@pytest.fixture
def refiner():
    return FunPackLTXAVSceneChainSampler.__new__(FunPackLTXAVSceneChainSampler)


def _fake_model():
    return types.SimpleNamespace(model_options={})


def _install_fakes(monkeypatch, denoised_for_seed, prepare_noise_calls=None, sample_calls=None):
    def fake_prepare_noise(samples, seed, noise_inds=None):
        if prepare_noise_calls is not None:
            prepare_noise_calls.append(seed)
        return torch.zeros_like(samples)

    def fake_sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image,
                           noise_mask=None, callback=None, disable_pbar=False, seed=None):
        if sample_calls is not None:
            sample_calls.append(seed)
        wrapper = model.model_options.get("model_function_wrapper")
        denoised = denoised_for_seed(seed, latent_image)
        if wrapper is not None:
            wrapper(lambda x, t, **c: denoised, {"input": noise, "timestep": torch.tensor([1.0])})
        return latent_image

    monkeypatch.setattr(sys.modules["comfy.sample"], "prepare_noise", fake_prepare_noise, raising=False)
    monkeypatch.setattr(sys.modules["comfy.sample"], "sample_custom", fake_sample_custom, raising=False)


def test_the_highest_scoring_candidate_seed_wins(monkeypatch, refiner):
    model = _fake_model()
    sample_calls = []
    # the fake model's "prediction" is literally the seed's own value broadcast into the
    # latent -- the fake value_fn ranks by mean, so the LARGEST seed must always win.
    _install_fakes(monkeypatch, lambda seed, latent: torch.full_like(latent, float(seed)),
                   sample_calls=sample_calls)

    latent = {"samples": torch.zeros(1, 4, 2, 2, 2)}
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    winner = refiner._select_best_seed(
        model, object(), sigmas, seed=100, cfg=1.0, positive=[], negative=[],
        latent=latent, n_candidates=3, value_fn=_FakeValueFn())

    assert winner == 100 + 2 * SEED_OFFSET
    assert sample_calls == [100, 100 + SEED_OFFSET, 100 + 2 * SEED_OFFSET]
    # the observer wrapper must be stripped after every candidate, success or failure --
    # the REAL run that follows must never inherit it.
    assert model.model_options.get("model_function_wrapper") is None


def test_a_lower_seed_can_win_when_it_scores_higher(monkeypatch, refiner):
    """Not just 'largest seed wins' by construction -- the WINNING seed is whichever one the
    scorer actually preferred, which here is deliberately the FIRST (lowest) candidate."""
    model = _fake_model()

    def denoised_for_seed(seed, latent):
        # seed 100 (candidate 0) scores highest; the other two score lower.
        value = 10.0 if seed == 100 else 1.0
        return torch.full_like(latent, value)

    _install_fakes(monkeypatch, denoised_for_seed)
    latent = {"samples": torch.zeros(1, 4, 2, 2, 2)}
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    winner = refiner._select_best_seed(
        model, object(), sigmas, seed=100, cfg=1.0, positive=[], negative=[],
        latent=latent, n_candidates=3, value_fn=_FakeValueFn())
    assert winner == 100


def test_a_value_function_that_is_not_ready_is_a_pure_no_op(monkeypatch, refiner):
    model = _fake_model()
    calls = []
    _install_fakes(monkeypatch, lambda seed, latent: torch.full_like(latent, float(seed)),
                   sample_calls=calls)
    latent = {"samples": torch.zeros(1, 4, 2, 2, 2)}
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    winner = refiner._select_best_seed(
        model, object(), sigmas, seed=100, cfg=1.0, positive=[], negative=[],
        latent=latent, n_candidates=3, value_fn=_NotReadyValueFn())
    assert winner == 100
    assert calls == []          # no candidate work was ever attempted


def test_none_value_function_is_a_pure_no_op(refiner):
    latent = {"samples": torch.zeros(1, 4, 2, 2, 2)}
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    winner = refiner._select_best_seed(
        _fake_model(), object(), sigmas, seed=100, cfg=1.0, positive=[], negative=[],
        latent=latent, n_candidates=3, value_fn=None)
    assert winner == 100


def test_a_schedule_too_short_to_slice_one_step_is_a_no_op(refiner):
    latent = {"samples": torch.zeros(1, 4, 2, 2, 2)}
    winner = refiner._select_best_seed(
        _fake_model(), object(), torch.tensor([1.0]), seed=100, cfg=1.0, positive=[],
        negative=[], latent=latent, n_candidates=3, value_fn=_FakeValueFn())
    assert winner == 100


def test_fewer_than_two_candidates_is_a_no_op(refiner):
    latent = {"samples": torch.zeros(1, 4, 2, 2, 2)}
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    winner = refiner._select_best_seed(
        _fake_model(), object(), sigmas, seed=100, cfg=1.0, positive=[], negative=[],
        latent=latent, n_candidates=1, value_fn=_FakeValueFn())
    assert winner == 100


def test_a_candidate_whose_sample_custom_call_raises_is_skipped_not_fatal(monkeypatch, refiner):
    model = _fake_model()

    def fake_prepare_noise(samples, seed, noise_inds=None):
        return torch.zeros_like(samples)

    def fake_sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image,
                           noise_mask=None, callback=None, disable_pbar=False, seed=None):
        if seed == 100:
            raise RuntimeError("candidate 0 boom")
        wrapper = model.model_options.get("model_function_wrapper")
        denoised = torch.full_like(latent_image, float(seed))
        if wrapper is not None:
            wrapper(lambda x, t, **c: denoised, {"input": noise, "timestep": torch.tensor([1.0])})
        return latent_image

    monkeypatch.setattr(sys.modules["comfy.sample"], "prepare_noise", fake_prepare_noise, raising=False)
    monkeypatch.setattr(sys.modules["comfy.sample"], "sample_custom", fake_sample_custom, raising=False)

    latent = {"samples": torch.zeros(1, 4, 2, 2, 2)}
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    winner = refiner._select_best_seed(
        model, object(), sigmas, seed=100, cfg=1.0, positive=[], negative=[],
        latent=latent, n_candidates=3, value_fn=_FakeValueFn())
    # candidate 0 (seed=100) raised and is excluded from scoring entirely -- the winner
    # must be the better of the two SURVIVING candidates, not the seed that crashed, and
    # the crash must not propagate (the real run must still get to happen).
    assert winner == 100 + 2 * SEED_OFFSET
    # the wrapper is still restored to None even after the raising candidate.
    assert model.model_options.get("model_function_wrapper") is None


def test_every_candidate_failing_falls_back_to_the_original_seed(monkeypatch, refiner):
    model = _fake_model()

    def fake_prepare_noise(samples, seed, noise_inds=None):
        return torch.zeros_like(samples)

    def fake_sample_custom(*a, **k):
        raise RuntimeError("every candidate boom")

    monkeypatch.setattr(sys.modules["comfy.sample"], "prepare_noise", fake_prepare_noise, raising=False)
    monkeypatch.setattr(sys.modules["comfy.sample"], "sample_custom", fake_sample_custom, raising=False)

    latent = {"samples": torch.zeros(1, 4, 2, 2, 2)}
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    winner = refiner._select_best_seed(
        model, object(), sigmas, seed=100, cfg=1.0, positive=[], negative=[],
        latent=latent, n_candidates=3, value_fn=_FakeValueFn())
    assert winner == 100
    assert model.model_options.get("model_function_wrapper") is None


def test_only_a_two_element_probe_schedule_is_used_regardless_of_the_real_schedule_length(monkeypatch, refiner):
    """The real run's full sigma schedule (dozens of steps) must never be handed to the
    throwaway candidate calls -- only a 2-element slice (exactly one step) may reach
    sample_custom, which is the entire cost guarantee this feature makes."""
    model = _fake_model()
    seen_lengths = []

    def fake_prepare_noise(samples, seed, noise_inds=None):
        return torch.zeros_like(samples)

    def fake_sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image,
                           noise_mask=None, callback=None, disable_pbar=False, seed=None):
        seen_lengths.append(int(sigmas.numel()))
        wrapper = model.model_options.get("model_function_wrapper")
        denoised = torch.full_like(latent_image, float(seed))
        if wrapper is not None:
            wrapper(lambda x, t, **c: denoised, {"input": noise, "timestep": torch.tensor([1.0])})
        return latent_image

    monkeypatch.setattr(sys.modules["comfy.sample"], "prepare_noise", fake_prepare_noise, raising=False)
    monkeypatch.setattr(sys.modules["comfy.sample"], "sample_custom", fake_sample_custom, raising=False)

    latent = {"samples": torch.zeros(1, 4, 2, 2, 2)}
    sigmas = torch.linspace(1.0, 0.0, steps=40)   # a realistic, long real-run schedule
    refiner._select_best_seed(
        model, object(), sigmas, seed=100, cfg=1.0, positive=[], negative=[],
        latent=latent, n_candidates=3, value_fn=_FakeValueFn())
    assert seen_lengths == [2, 2, 2]
