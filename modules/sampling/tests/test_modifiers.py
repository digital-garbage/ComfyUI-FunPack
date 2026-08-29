"""Modifiers: chosen, filtered, installed, and never accumulating.

Run against real ComfyUI ModelPatcher objects. No weights are needed: a patcher
around an empty nn.Module carries the same wrapper machinery a real one does.
"""

import pytest


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Everything here touches comfy."""


@pytest.fixture
def patcher():
    import torch
    from comfy.model_patcher import ModelPatcher

    class Stub(torch.nn.Module):
        pass

    return ModelPatcher(Stub(), load_device=torch.device("cpu"),
                        offload_device=torch.device("cpu"))


def _spec(module_id, install, requires=(), settings=None, stage="sampling"):
    from core.contract import ModuleSpec
    return ModuleSpec(id=module_id, title=module_id, mount="", stage=stage,
                      requires=list(requires), settings=settings or {},
                      provides={"modifier": install})


@pytest.fixture
def registry(monkeypatch):
    """A registry holding only what a test puts in it."""
    from core import registry as registry_mod

    class Fake:
        def __init__(self):
            self.specs = {}
            self.failed = []

    fake = Fake()
    monkeypatch.setattr(registry_mod, "current", lambda rescan=False: fake)
    return fake


# --- installing ------------------------------------------------------------

def test_a_modifier_installs_on_the_model_and_is_reported(registry, patcher):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    def install(target, values, key):
        target.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, key, lambda e, *a, **k: e(*a, **k))
        return "on"

    registry.specs["m"] = _spec("m", install)
    out = FunPackLoadModifiers.execute(patcher)
    model, status = out.result

    assert "m: on" in status
    assert any("funpack.m" in keys for keys in model.wrappers.get(WrappersMP.SAMPLER_SAMPLE, {}))


def test_installing_never_touches_the_model_everyone_else_holds(registry, patcher):
    """The fault this makes impossible: v4 registered hooks on shared blocks and
    removed them only on a scene change, so every run stacked another set and
    only a restart cleared them. Installing on a clone means the original is
    untouched, so there is nothing to accumulate and nothing to clean up."""
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    def install(target, values, key):
        target.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, key, lambda e, *a, **k: e(*a, **k))
        return "on"

    registry.specs["m"] = _spec("m", install)

    before = len(patcher.wrappers.get(WrappersMP.SAMPLER_SAMPLE, {}))
    for _ in range(50):
        FunPackLoadModifiers.execute(patcher)

    after = patcher.wrappers.get(WrappersMP.SAMPLER_SAMPLE, {})
    assert len(after) == before == 0, (
        f"fifty runs left {len(after)} wrapper(s) on the shared model")


def test_a_modifier_that_says_it_is_off_installs_nothing(registry, patcher):
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    def install(target, values, key):
        return None                      # the module's own decision, not core's

    registry.specs["m"] = _spec("m", install)
    _model, status = FunPackLoadModifiers.execute(patcher).result
    assert "0 modifier(s) applied" in status


def test_one_broken_modifier_does_not_stop_the_others(registry, patcher):
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    def explodes(target, values, key):
        raise RuntimeError("boom")

    def works(target, values, key):
        return "on"

    registry.specs["a_broken"] = _spec("a_broken", explodes)
    registry.specs["b_works"] = _spec("b_works", works)

    _model, status = FunPackLoadModifiers.execute(patcher).result
    assert "b_works: on" in status
    assert "failed to install" in status and "boom" in status


def test_modifiers_install_in_declared_order(registry, patcher):
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    from core.contract import ModuleSpec

    seen = []

    def make(name):
        def install(target, values, key):
            seen.append(name)
            return "on"
        return install

    # "first" declares that it runs before "second", so filesystem/alphabetical
    # order must not decide this.
    registry.specs["zzz_first"] = ModuleSpec(
        id="zzz_first", title="", mount="", before=["aaa_second"],
        provides={"modifier": make("zzz_first")})
    registry.specs["aaa_second"] = ModuleSpec(
        id="aaa_second", title="", mount="", provides={"modifier": make("aaa_second")})

    FunPackLoadModifiers.execute(patcher)
    assert seen == ["zzz_first", "aaa_second"]


# --- filtering by what the model actually is -------------------------------

def test_a_modifier_needing_an_absent_trait_is_absent_and_says_why(registry, patcher):
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    def install(target, values, key):
        raise AssertionError("an incompatible modifier was installed")

    registry.specs["needs_audio"] = _spec("needs_audio", install, requires=["audio_stream"])
    _model, status = FunPackLoadModifiers.execute(patcher).result

    assert "needs_audio: needs audio_stream" in status
    assert "0 modifier(s) applied" in status


def test_a_modifier_declaring_no_traits_runs_on_anything(registry, patcher):
    """Narrowing, not opting in -- the property that lets a modifier reach models
    FunPack has never heard of."""
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    registry.specs["plain"] = _spec("plain", lambda t, v, key: "on")
    _model, status = FunPackLoadModifiers.execute(patcher).result
    assert "plain: on" in status


# --- the settings payload --------------------------------------------------

def test_values_reach_the_modifier_that_declared_them(registry, patcher):
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    got = {}

    def install(target, values, key):
        got.update(values)
        return "on"

    registry.specs["m"] = _spec("m", install, settings={
        "strength": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "label": "S"},
    })
    FunPackLoadModifiers.execute(patcher, settings={"m": {"strength": 0.8}})
    assert got == {"strength": 0.8}


def test_a_modifier_with_no_values_gets_its_declared_defaults(registry, patcher):
    """A headless run must match a run with the panel open, because both read the
    same declaration."""
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    got = {}
    registry.specs["m"] = _spec("m", lambda t, v, key: got.update(v) or "on", settings={
        "strength": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "label": "S"},
    })
    FunPackLoadModifiers.execute(patcher)
    assert got == {"strength": 0.5}
