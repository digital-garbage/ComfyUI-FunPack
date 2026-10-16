"""Traits: what is read off a model, and what a model's own module contributes.

The split under test is the one the architecture depends on -- core reads only
facts that need no knowledge of any particular model, and everything else is
announced. If core ever learned a model's name, supporting the next one would
mean editing core, which is the whole thing this avoids.
"""

import pytest

from core.contract import ModuleSpec
from core.traits import contributed, has_block, missing_for, split, traits_of, universal


class _LatentFormat:
    def __init__(self, dims, tdr=1):
        self.latent_dimensions = dims
        self.temporal_downscale_ratio = tdr


class _Kind:
    def __init__(self, name):
        self.name = name


class _Model:
    """The shape traits_of() reads: patcher -> model -> model_config."""

    def __init__(self, dims=3, tdr=4, kind="FLOW"):
        outer = self

        class Inner:
            model_type = _Kind(kind)

            class model_config:
                latent_format = _LatentFormat(dims, tdr)

        self.model = Inner()
        del outer


def test_latent_rank_is_read_not_guessed():
    assert "temporal_latent" in universal(_Model(dims=3))
    assert "spatial_latent" in universal(_Model(dims=2))
    assert "waveform_latent" in universal(_Model(dims=1))


def test_the_prediction_type_is_named():
    assert "predict_flow" in universal(_Model(kind="FLOW"))
    assert "predict_eps" in universal(_Model(kind="EPS"))


def test_an_unreadable_model_yields_nothing_rather_than_raising():
    class Bare:
        model = None

    assert traits_of(Bare()) == []


def test_a_module_contributes_traits_core_could_not_read():
    spec = ModuleSpec(id="m", title="M", mount="", traits=lambda model: ["audio_stream"])
    assert "audio_stream" in traits_of(_Model(), [spec])


def test_a_provider_that_raises_does_not_silence_the_others():
    def explodes(model):
        raise RuntimeError("boom")

    bad = ModuleSpec(id="bad", title="B", mount="", traits=explodes)
    good = ModuleSpec(id="good", title="G", mount="", traits=lambda m: ["audio_stream"])

    found = traits_of(_Model(), [bad, good])
    assert "audio_stream" in found, "one bad provider disabled every other module"
    assert "temporal_latent" in found


def test_a_provider_that_raises_is_not_reported_as_one_that_never_loaded():
    """The log line, not only the surviving traits.

    A traits provider checks whether the model is its own FIRST, so reaching the
    exception means it loaded, recognised the model and then broke. Saying "did
    not load" sends a reader after an import error that never happened -- and
    losing a model's traits quietly narrows which modifiers are considered
    compatible, so the log is the only thing pointing at the cause.
    """
    from core import log

    def explodes(model):
        raise RuntimeError("adaln lookup exploded")

    log._reset()
    traits_of(_Model(), [ModuleSpec(id="bad", title="B", mount="", traits=explodes)])

    said = log.history()
    assert said, "a provider broke and nothing was said"
    assert not any("did not load" in r["message"] for r in said), \
        [r["message"] for r in said]
    assert any("failed while" in r["message"] for r in said), [r["message"] for r in said]
    assert said[-1]["level"] == log.WARNING


def test_declaring_no_traits_means_compatible_with_everything():
    # Narrowing, not opting in -- the property that lets a modifier reach models
    # FunPack knows nothing about.
    plain = ModuleSpec(id="plain", title="P", mount="")
    compatible, incompatible = split([plain], [])
    assert compatible == [plain] and incompatible == []


def test_declaring_a_trait_excludes_the_models_that_lack_it():
    picky = ModuleSpec(id="picky", title="P", mount="", requires=["audio_stream"])
    compatible, incompatible = split([picky], ["temporal_latent"])
    assert incompatible == [picky]
    assert missing_for(picky, ["temporal_latent"]) == ["audio_stream"]


def test_a_structural_probe_finds_a_block_by_class_name():
    class Target:
        pass

    class Root:
        def modules(self):
            return [self, Target()]

    class Model:
        class model:
            diffusion_model = Root()

    # The question a modifier actually has: does this model have the thing I
    # touch -- not "is this the model I was written for".
    assert has_block(Model(), "Target")
    assert not has_block(Model(), "SomethingElse")


def test_a_probe_on_a_model_with_no_module_tree_is_false_not_fatal():
    class Model:
        model = None

    assert has_block(Model(), "Anything") is False


# --- against ComfyUI's real model configs ---------------------------------

@pytest.fixture(scope="module")
def real_configs(comfyui):
    import comfy.supported_models as sm
    return sm.models


def test_every_real_model_config_can_be_read(real_configs):
    # The point is coverage, not a particular answer: whatever ships upstream,
    # reading it must not raise and must not invent anything.
    known = {"temporal_latent", "spatial_latent", "waveform_latent",
             "temporal_compression"}
    for cfg in real_configs:
        model = _Model.__new__(_Model)

        class Inner:
            model_type = None
            model_config = cfg

        model.model = Inner()
        found = universal(model)
        assert found <= known, f"{cfg.__name__} produced an unexpected trait: {found - known}"


def test_traits_describe_the_tensor_not_an_assumed_modality(real_configs):
    """QwenImage is an IMAGE model with a 3-D latent. LTXV is a video model with
    a 3-D latent. They must read the same, because the latent really is the same
    shape -- claiming otherwise would be FunPack guessing at modality, which
    ComfyUI itself does not classify."""
    by_name = {cfg.__name__: cfg for cfg in real_configs}
    qwen, ltxv = by_name.get("QwenImage"), by_name.get("LTXV")
    if qwen is None or ltxv is None:
        pytest.skip("upstream renamed these configs")

    def read(cfg):
        model = _Model.__new__(_Model)

        class Inner:
            model_type = None
            model_config = cfg

        model.model = Inner()
        return universal(model)

    assert "temporal_latent" in read(qwen)
    assert "temporal_latent" in read(ltxv)


def test_a_bare_probe_name_collides_across_architectures():
    """`Attention`, `MLP`, `Block` are not distinctive. A bare name matches any
    of them, which is why a probe that needs to be sure must qualify."""

    class Root:
        def modules(self):
            # A class named Attention, but from an unrelated architecture.
            other = type("Attention", (), {})
            other.__module__ = "somewhere.else.entirely"
            return [self, other()]

    class Model:
        class model:
            diffusion_model = Root()

    assert has_block(Model(), "Attention")           # the collision, unqualified
    assert not has_block(Model(), "comfy.ldm.minimax.model.Attention")


def test_a_qualified_probe_matches_its_own_architecture():
    class Root:
        def modules(self):
            target = type("AdalnProj", (), {})
            target.__module__ = "comfy.ldm.minimax.model"
            return [self, target()]

    class Model:
        class model:
            diffusion_model = Root()

    assert has_block(Model(), "comfy.ldm.minimax.model.AdalnProj")
    assert has_block(Model(), "minimax.model.AdalnProj")     # tail match
    assert has_block(Model(), "AdalnProj")                   # distinctive enough
    assert not has_block(Model(), "other.pack.AdalnProj")
