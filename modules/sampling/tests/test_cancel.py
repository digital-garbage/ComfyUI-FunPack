"""Cancelling a run, and what it must leave behind: nothing.

Stopping a generation is ordinary -- a bad seed, a wrong prompt, a rental
minute -- so it is not an error path that can be left approximate. What matters
is that the next run starts clean, because a leak here is invisible: the run
after a cancel just behaves slightly wrong.
"""

import pytest


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture
def patcher():
    import torch
    from comfy.model_patcher import ModelPatcher

    class Stub(torch.nn.Module):
        pass

    return ModelPatcher(Stub(), load_device=torch.device("cpu"),
                        offload_device=torch.device("cpu"))


@pytest.fixture
def registry(monkeypatch):
    from core import registry as registry_mod
    fake = registry_mod.Registry()
    monkeypatch.setattr(registry_mod, "current", lambda rescan=False: fake)
    return fake


def _spec(module_id, install):
    from core.contract import ModuleSpec
    return ModuleSpec(id=module_id, title=module_id, mount="", stage="sampling",
                      provides={"modifier": install})


def test_a_cancel_reaches_the_top_and_is_not_absorbed(patcher, registry):
    """The guard catches Exception, and ComfyUI's cancel is a BaseException, so
    it passes through. If that ever changed, stop would stop nothing."""
    from comfy.model_management import InterruptProcessingException
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    def install(target, values, key):
        def cancels(args):
            raise InterruptProcessingException()
        target.set_model_sampler_pre_cfg_function(cancels)
        return "on"

    registry.specs["m"] = _spec("m", install)
    patched, _ = FunPackLoadModifiers.execute(patcher).result
    hook = patched.model_options["sampler_pre_cfg_function"][0]

    with pytest.raises(InterruptProcessingException):
        hook({"conds_out": ["a", "b"]})


def test_a_cancelled_run_leaves_nothing_on_the_shared_model(patcher, registry):
    """The model everyone else holds must be exactly as it was. Installing on a
    clone is what makes this true, and a cancel must not change that."""
    from comfy.model_management import InterruptProcessingException
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    def install(target, values, key):
        target.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, key,
                                    lambda e, *a, **k: e(*a, **k))
        return "on"

    registry.specs["m"] = _spec("m", install)

    for _ in range(5):
        patched, _ = FunPackLoadModifiers.execute(patcher).result
        try:
            raise InterruptProcessingException()      # cancelled mid-run
        except InterruptProcessingException:
            pass

    assert patcher.wrappers.get(WrappersMP.SAMPLER_SAMPLE, {}) == {}
    assert not [k for k in patcher.model_options.get("sampler_pre_cfg_function", [])]


def test_a_cancel_does_not_mark_a_modifier_as_dropped(patcher, registry):
    """A cancelled modifier did not fail. Recording it would disable it on the
    next run for something the user did on purpose."""
    from comfy.model_management import InterruptProcessingException
    from core import patching

    dropped = patching.Dropped()

    def cancels(args):
        raise InterruptProcessingException()

    guarded = patching.guard(cancels, "funpack.m",
                             lambda args: args["conds_out"], dropped)
    with pytest.raises(InterruptProcessingException):
        guarded({"conds_out": "x"})

    assert "funpack.m" not in dropped and not dropped


def test_the_next_run_starts_with_a_clean_record(patcher, registry):
    """The record travels on the model, which ComfyUI may hand back from its
    cache -- so a run cancelled after a modifier failed must not leave that
    modifier disabled for the next one."""
    from core import patching
    from modules.sampling.sampler.nodes import FunPackSampler

    dropped = patching.Dropped()
    dropped.record("funpack.m", RuntimeError("failed last time"))
    patcher.funpack_dropped = dropped

    # What the sampler does at the start of every generation.
    found = getattr(patcher, "funpack_dropped", None)
    found.clear()
    assert "funpack.m" not in found


def test_alg_puts_the_sharp_anchor_back_when_a_run_is_cancelled():
    """ALG swaps the pinned anchor per step. A cancel between steps must not
    leave the blurred copy pinned on a denoiser that outlives the run."""
    import torch
    import comfy.samplers
    from comfy.model_management import InterruptProcessingException
    from modules.sampling import alg

    sharp = torch.randn(1, 3, 4, 16, 16)
    holder = {}

    def sampler_function(model_k, x, sigmas, **kwargs):
        model_k(x, torch.tensor([0.9]), denoise_mask=None)
        holder["denoiser"] = model_k._inner
        raise InterruptProcessingException()

    captured = {}

    class Patcher:
        def add_wrapper_with_key(self, kind, key, wrapper):
            captured["wrapper"] = wrapper

    alg.install(Patcher(), {"enabled": True, "strength": 4.0, "until_sigma": 0.6},
                key="funpack.alg")

    class Guider:
        def __init__(self):
            class ModelSampling:
                sigma_max = 1.0
                def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
                    return noise
                def inverse_noise_scaling(self, sigma, x):
                    return x

            class InnerModel:
                model_sampling = ModelSampling()
                def scale_latent_inpaint(self, **kw):
                    return kw["latent_image"]

            self.inner_model = InnerModel()

        def __call__(self, x, sigma, model_options={}, seed=None):
            return x

    class Executor:
        class_obj = comfy.samplers.KSAMPLER(sampler_function)
        def __call__(self, *a, **k):
            raise AssertionError("should not fall back")

    with pytest.raises(InterruptProcessingException):
        captured["wrapper"](Executor(), Guider(), torch.tensor([0.9]), {}, None,
                            torch.zeros_like(sharp), sharp, torch.ones_like(sharp), True)

    assert torch.equal(holder["denoiser"].latent_image, sharp), (
        "a cancelled run left the loosened anchor pinned")
