"""A modifier installed on the model is reached by ComfyUI's own sampling path.

This is the architectural claim made checkable: modifiers attach to the MODEL, so
no sampler has to cooperate and therefore no sampler can own them. If this test
holds, wiring the modified model into ComfyUI's own SamplerCustomAdvanced runs
every per-step modifier.

It uses comfy's real functions -- prepare_model_patcher, get_all_wrappers,
WrapperExecutor -- rather than asserting on our own dict. Checking that we wrote
a key we chose would prove nothing about whether anything reads it.
"""

import pytest


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture
def registry(monkeypatch):
    from core import registry as registry_mod
    from core.contract import ModuleSpec

    fake = registry_mod.Registry()
    monkeypatch.setattr(registry_mod, "current", lambda rescan=False: fake)
    return fake, ModuleSpec


def test_the_wrapper_a_modifier_installs_is_executed_by_comfys_sampler_path(registry):
    import torch
    import comfy.patcher_extension as ext
    import comfy.sampler_helpers
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    fake, ModuleSpec = registry
    ran = []

    def install(target, values, key):
        def wrapper(executor, *args, **kwargs):
            ran.append(key)
            return executor(*args, **kwargs)

        target.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, key, wrapper)
        return "on"

    fake.specs["m"] = ModuleSpec(id="m", title="m", mount="",
                                 provides={"modifier": install})

    class Stub(torch.nn.Module):
        pass

    patcher = ModelPatcher(Stub(), load_device=torch.device("cpu"),
                           offload_device=torch.device("cpu"))
    patched, _status = FunPackLoadModifiers.execute(patcher).result

    # Exactly what CFGGuider.inner_sample does: the patcher's wrappers are merged
    # into model_options, then the sampler's own sample() is wrapped in them.
    model_options = {"transformer_options": {}}
    comfy.sampler_helpers.prepare_model_patcher(patched, {}, model_options)

    def sample(*args, **kwargs):
        return "sampled"

    executor = ext.WrapperExecutor.new_class_executor(
        sample, None,
        ext.get_all_wrappers(WrappersMP.SAMPLER_SAMPLE, model_options, is_model_options=True))

    assert executor.execute() == "sampled"
    assert ran == ["funpack.m"], (
        "the modifier was installed somewhere ComfyUI's sampling path does not read")


def test_an_unmodified_model_carries_none_of_our_wrappers(registry):
    """The control. Without it the test above could pass on a machine where
    something else installs a wrapper under any key at all."""
    import torch
    import comfy.sampler_helpers
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrappersMP

    class Stub(torch.nn.Module):
        pass

    patcher = ModelPatcher(Stub(), load_device=torch.device("cpu"),
                           offload_device=torch.device("cpu"))
    model_options = {"transformer_options": {}}
    comfy.sampler_helpers.prepare_model_patcher(patcher, {}, model_options)

    installed = model_options["transformer_options"].get("wrappers", {}).get(
        WrappersMP.SAMPLER_SAMPLE, {})
    assert not [k for k in installed if str(k).startswith("funpack.")]
