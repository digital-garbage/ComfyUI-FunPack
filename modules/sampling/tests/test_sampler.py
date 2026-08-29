"""The sampler: it samples, and it hosts what it said it could host.

v4's became a hub -- 8183 lines and a `_sample_chunk` that grew six ALG
arguments. These tests are mostly about the shape that prevents that: one call
site, a context object rather than a widening signature, and a sampler that
never names a modifier.
"""

import pytest


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture
def registry(monkeypatch):
    from core import registry as registry_mod
    fake = registry_mod.Registry()
    monkeypatch.setattr(registry_mod, "current", lambda rescan=False: fake)
    return fake


def _spec(module_id, make, hooks=(), requires=(), settings=None):
    from core.contract import ModuleSpec
    return ModuleSpec(id=module_id, title=module_id, mount="", stage="sampling",
                      hooks=list(hooks), requires=list(requires),
                      settings=settings or {}, provides={"sampler_modifier": make})


class _Modifier:
    """The shape a sampler modifier has: active(ctx), process(ctx, latent)."""

    def __init__(self, when=lambda ctx: True):
        self.seen = []
        self.when = when

    def active(self, ctx):
        return self.when(ctx)

    def process(self, ctx, latent):
        self.seen.append(ctx.index)
        return latent


# --- the schedule ----------------------------------------------------------

def test_full_denoise_asks_for_exactly_the_steps_requested(comfyui, monkeypatch):
    import comfy.samplers
    import torch
    from modules.sampling.sampler.nodes import FunPackSampler

    seen = {}

    def fake(model_sampling, scheduler, steps):
        seen["steps"] = steps
        return torch.linspace(1.0, 0.0, steps + 1)

    monkeypatch.setattr(comfy.samplers, "calculate_sigmas", fake)

    class Model:
        def get_model_object(self, name):
            return None

    out = FunPackSampler._sigmas(Model(), "normal", 6, 1.0)
    assert seen["steps"] == 6 and len(out) == 7


def test_partial_denoise_is_the_tail_of_a_longer_schedule(comfyui, monkeypatch):
    """denoise 0.5 means the second half of a 20-step curve, not a 10-step
    curve. They are different curves and swapping them changes the picture."""
    import comfy.samplers
    import torch
    from modules.sampling.sampler.nodes import FunPackSampler

    seen = {}

    def fake(model_sampling, scheduler, steps):
        seen["steps"] = steps
        return torch.linspace(1.0, 0.0, steps + 1)

    monkeypatch.setattr(comfy.samplers, "calculate_sigmas", fake)

    class Model:
        def get_model_object(self, name):
            return None

    out = FunPackSampler._sigmas(Model(), "normal", 10, 0.5)
    assert seen["steps"] == 20, "the curve was not computed over the full length"
    assert len(out) == 11, "the tail was not trimmed to the requested steps"


def test_zero_denoise_asks_for_nothing(comfyui):
    from modules.sampling.sampler.nodes import FunPackSampler

    class Model:
        def get_model_object(self, name):
            return None

    assert len(FunPackSampler._sigmas(Model(), "normal", 10, 0.0)) == 0


# --- hosting modifiers -----------------------------------------------------

def test_a_modifier_sees_every_step_through_one_call_site(registry, comfyui):
    import torch
    from core import chain as chain_mod, patching
    from modules.sampling.sampler.nodes import ACCEPTS, FunPackSampler

    modifier = _Modifier()
    registry.specs["m"] = _spec("m", lambda values, accepts: modifier, hooks=["latent"])

    chain, notes = FunPackSampler._chain(_TraitlessModel(), None, patching.Dropped())
    assert chain.ids == ["m"] and "m: on" in notes

    # Drive it the way the wrapped sampler function would.
    for index in range(3):
        chain.process(chain_mod.Step(index=index, sigma=1.0, sigmas=None, total=3), "latent")
    assert modifier.seen == [0, 1, 2]


def test_a_modifier_can_sit_out_a_step(registry, comfyui):
    from core import chain as chain_mod, patching
    from modules.sampling.sampler.nodes import FunPackSampler

    modifier = _Modifier(when=lambda ctx: ctx.index % 2 == 0)
    registry.specs["m"] = _spec("m", lambda values, accepts: modifier, hooks=["latent"])
    chain, _ = FunPackSampler._chain(_TraitlessModel(), None, patching.Dropped())

    for index in range(4):
        chain.process(chain_mod.Step(index=index, sigma=1.0, sigmas=None, total=4), "l")
    assert modifier.seen == [0, 2]


def test_a_modifier_wanting_a_hook_this_sampler_lacks_is_absent(registry, comfyui):
    """The sampler keeps the veto. A modifier that needs a second pass cannot be
    usefully run by a sampler that makes one."""
    from core import patching
    from modules.sampling.sampler.nodes import FunPackSampler

    registry.specs["greedy"] = _spec("greedy", lambda values, accepts: _Modifier(),
                                     hooks=["second_pass"])
    chain, notes = FunPackSampler._chain(_TraitlessModel(), None, patching.Dropped())

    assert chain.ids == []
    assert any("needs second_pass" in note for note in notes)


def test_a_modifier_needing_an_absent_trait_is_absent(registry, comfyui):
    from core import patching
    from modules.sampling.sampler.nodes import FunPackSampler

    registry.specs["audio"] = _spec("audio", lambda values, accepts: _Modifier(),
                                    requires=["audio_stream"])
    chain, notes = FunPackSampler._chain(_TraitlessModel(), None, patching.Dropped())
    assert chain.ids == []
    assert any("audio_stream" in note for note in notes)


def test_one_failing_modifier_does_not_end_the_run(registry, comfyui):
    from core import chain as chain_mod, patching
    from modules.sampling.sampler.nodes import FunPackSampler

    class Breaks:
        def active(self, ctx):
            return True

        def process(self, ctx, latent):
            raise RuntimeError("boom")

    good = _Modifier()
    registry.specs["a_breaks"] = _spec("a_breaks", lambda v, a: Breaks())
    registry.specs["b_good"] = _spec("b_good", lambda v, a: good)

    dropped = patching.Dropped()
    chain, _ = FunPackSampler._chain(_TraitlessModel(), None, dropped)

    for index in range(5):
        out = chain.process(chain_mod.Step(index=index, sigma=1.0, sigmas=None, total=5), "l")
        assert out == "l"

    assert good.seen == [0, 1, 2, 3, 4], "a failing sibling stopped a healthy modifier"
    assert "funpack.a_breaks" in dropped


def test_bad_settings_are_refused_before_sampling(registry, comfyui):
    from core import patching
    from modules.sampling.sampler.nodes import FunPackSampler

    registry.specs["m"] = _spec("m", lambda v, a: _Modifier(), settings={
        "strength": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "label": "S"},
    })
    with pytest.raises(RuntimeError, match="above its maximum"):
        FunPackSampler._chain(_TraitlessModel(), {"m": {"strength": 9.0}}, patching.Dropped())


def test_something_that_is_not_a_modifier_is_named_not_run(registry, comfyui):
    from core import patching
    from modules.sampling.sampler.nodes import FunPackSampler

    registry.specs["m"] = _spec("m", lambda v, a: object())
    chain, notes = FunPackSampler._chain(_TraitlessModel(), None, patching.Dropped())
    assert chain.ids == []
    assert any("not a sampler modifier" in note for note in notes)


class _TraitlessModel:
    """A model core can read nothing from, so trait filtering is exercised
    without needing weights."""

    model = None
