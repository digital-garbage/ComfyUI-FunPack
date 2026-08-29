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
