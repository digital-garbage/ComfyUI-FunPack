"""Seed search: scores one real first step per candidate, keeps the best, learns nothing from probes."""

import types

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    from modules.system.taste import store
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")


SHAPES = [torch.Size([1, 4, 2, 4, 4]), torch.Size([1, 3, 5])]
N_VIDEO = 4 * 2 * 4 * 4


class Model:
    """What SAMPLER_SAMPLE is handed: x0 = x, packed like H3's."""

    def __init__(self):
        self.options = []
        self.inner_model = types.SimpleNamespace(
            latent_shapes=SHAPES,
            model_sampling=types.SimpleNamespace(sigma_max=1.0, noise_scaling=lambda s, n, l, m: n))

    def __call__(self, x, sigma, model_options=None, seed=None):
        self.options.append(model_options)
        return x


class Executor:
    def __init__(self):
        import comfy.samplers
        self.class_obj = comfy.samplers.KSAMPLER(lambda *a, **k: None)
        self.noise = None

    def __call__(self, model_wrap, sigmas, extra_args, callback, noise, *rest):
        self.noise = noise
        return noise


def _load(tiny):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    patched, _ = FunPackLoadModifiers.execute(tiny.patcher, {
        "taste": {"key": "fox"}, "explore_first_step": {"enabled": True, "candidates": 4}}).result
    sample = [w for ws in patched.wrappers[WrappersMP.SAMPLER_SAMPLE].values() for w in ws][-1]
    for ws in patched.wrappers[WrappersMP.OUTER_SAMPLE].values():
        for o in ws:
            o(lambda: None)
    return sample


def _teach():
    """Liked first steps were bright, disliked dark."""
    from modules.system.taste import store, value
    for i in range(12):
        sign = 1.0 if i % 2 else -1.0
        store.capture("fox", "first_step",
                      {"first": value.describe(sign + 0.2 * torch.randn(1, 4, 2, 4, 4))},
                      prompt_id=f"t{i}")
        store.rate(f"t{i}", "liked" if sign > 0 else "disliked")


def test_until_it_has_learned_the_seed_is_untouched_and_nothing_extra_runs(tiny_h3):
    sample = _load(tiny_h3)
    model, ex = Model(), Executor()
    noise = torch.randn(1, 1, N_VIDEO + 15)
    sample(ex, model, torch.tensor([1.0, 0.0]), {"model_options": {}, "seed": 5}, None, noise,
           torch.zeros_like(noise), None, True)
    assert ex.noise is noise and not model.options


def test_it_keeps_the_candidate_the_judge_scores_best_and_marks_every_try_a_probe(tiny_h3):
    from core import dit_hooks
    from modules.system.taste import value
    _teach()
    sample = _load(tiny_h3)
    model, ex = Model(), Executor()
    noise = torch.randn(1, 1, N_VIDEO + 15)
    sample(ex, model, torch.tensor([1.0, 0.0]), {"model_options": {}, "seed": 5}, None, noise,
           torch.zeros_like(noise), None, True)
    assert len(model.options) == 4
    assert all(dit_hooks.probing(o["transformer_options"]) for o in model.options)

    from modules.system.taste import Handle
    judge = Handle("fox").judge("first_step", "first")
    score = lambda n: judge.score(n[..., :N_VIDEO].reshape(SHAPES[0]))
    candidates = [noise] + [torch.randn(noise.shape, generator=torch.Generator().manual_seed(5 + 7919 * k))
                            for k in range(1, 4)]
    assert torch.equal(ex.noise, max(candidates, key=score))
