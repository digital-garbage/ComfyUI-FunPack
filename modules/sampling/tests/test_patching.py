"""Tag and strip, for BOTH shapes of hook ComfyUI offers.

Keyed wrappers can be removed by name. The function lists on model_options
cannot: `set_model_sampler_pre_cfg_function` appends an anonymous callable and
records nothing about who added it. That second shape is how the accumulation bug
came back a third time -- a model run through the loader twice carried two copies
and applied the effect at double strength while reporting once.
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


def test_a_tagged_pre_cfg_hook_is_removed(patcher):
    from core import patching

    def hook(args):
        return args["conds_out"]

    patcher.set_model_sampler_pre_cfg_function(patching.tag(hook, "funpack.x"))
    assert len(patcher.model_options["sampler_pre_cfg_function"]) == 1

    assert patching.strip(patcher, "funpack.") == 1
    assert patcher.model_options["sampler_pre_cfg_function"] == []


def test_an_untagged_hook_is_left_alone(patcher):
    """Someone else's hook is not ours to remove, and removing it would be a
    worse fault than the one being fixed."""
    from core import patching

    def theirs(args):
        return args["conds_out"]

    patcher.set_model_sampler_pre_cfg_function(theirs)
    assert patching.strip(patcher, "funpack.") == 0
    assert patcher.model_options["sampler_pre_cfg_function"] == [theirs]


def test_a_differently_namespaced_hook_is_left_alone(patcher):
    from core import patching

    def other(args):
        return args["conds_out"]

    patcher.set_model_sampler_pre_cfg_function(patching.tag(other, "otherpack.y"))
    assert patching.strip(patcher, "funpack.") == 0


def test_something_that_cannot_be_tagged_is_refused_rather_than_installed():
    """An untaggable hook could never be removed, so installing it would build
    in the accumulation this exists to prevent."""
    from core import patching
    with pytest.raises(TypeError, match="could never be removed"):
        patching.tag(len, "funpack.x")


def test_chaining_the_loader_does_not_stack_a_pre_cfg_modifier(patcher):
    """The end-to-end version, through the real node."""
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    values = {"sharpen": {"enabled": True, "amount": 0.4, "radius": 5}}
    once = FunPackLoadModifiers.execute(patcher, settings=values).result[0]
    twice = FunPackLoadModifiers.execute(once, settings=values).result[0]
    thrice = FunPackLoadModifiers.execute(twice, settings=values).result[0]

    def count(model):
        return len(model.model_options.get("sampler_pre_cfg_function", []))

    assert count(once) == count(twice) == count(thrice) == 1
    assert count(patcher) == 0, "the shared model was modified"
