"""The empty latent: derived where it can be, claimed where it cannot.

The derivation is checked against ComfyUI's REAL model configs and against the
shapes core's own nodes produce, because a latent of the wrong shape does not
fail here -- it fails at the first sampling step, or worse, samples something
subtly wrong.
"""

import pytest


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """These import comfy and the node modules."""


class _Model:
    """The shape the node reads: patcher -> model -> model_config."""

    def __init__(self, config):
        class Inner:
            model_config = config
        self.model = Inner()


def _config_named(name):
    import comfy.supported_models as sm
    for cfg in sm.models:
        if cfg.__name__ == name:
            return cfg
    pytest.skip(f"upstream no longer ships {name}")


@pytest.mark.parametrize("name,width,height,length,expected", [
    # channels, then the spatial dims the format's own ratio implies.
    ("SDXL", 1024, 1024, 1, (1, 4, 128, 128)),
    ("WAN21_T2V", 512, 512, 81, (1, 16, 21, 64, 64)),
    ("LTXV", 768, 512, 97, (1, 128, 13, 16, 24)),
])
def test_the_derived_shape_matches_the_models_own_format(name, width, height, length, expected):
    from modules.latent.empty.nodes import derive
    out = derive(_Model(_config_named(name)), width, height, length, 1)
    assert tuple(out["samples"].shape) == expected


def test_ltx_matches_what_comfys_own_node_produces():
    """The strongest available check short of sampling: the same numbers core's
    EmptyLTXVLatentVideo computes, from a completely different route."""
    import torch
    from modules.latent.empty.nodes import derive

    width, height, length = 768, 512, 97
    theirs = torch.zeros([1, 128, ((length - 1) // 8) + 1, height // 32, width // 32])
    ours = derive(_Model(_config_named("LTXV")), width, height, length, 1)

    assert tuple(ours["samples"].shape) == tuple(theirs.shape)
    assert ours["downscale_ratio_spacial"] == 32


def test_an_image_model_gets_an_image_latent_with_no_family_branch():
    """The not-video-only property, stated as a test: nothing in the node asks
    what kind of model this is."""
    from modules.latent.empty.nodes import derive
    out = derive(_Model(_config_named("SDXL")), 512, 512, 99, 1)
    assert out["samples"].ndim == 4, "a length was supplied and grew a time axis anyway"


def test_a_tiny_canvas_never_yields_a_zero_sized_latent():
    """height // 32 floors to 0 on a small canvas, and a zero-sized latent fails
    a long way from here."""
    from modules.latent.empty.nodes import derive
    out = derive(_Model(_config_named("LTXV")), 16, 16, 1, 1)
    assert all(dim >= 1 for dim in out["samples"].shape)


def test_a_model_that_publishes_no_latent_format_says_so():
    from modules.latent.empty.nodes import derive

    class Bare:
        class model:
            model_config = None

    with pytest.raises(RuntimeError, match="latent format"):
        derive(Bare(), 64, 64, 1, 1)


def test_the_ratios_reported_are_the_ratios_used():
    """The sampler rescales when these disagree with the model's own, so
    reporting anything other than what we built at causes a silent resize."""
    from modules.latent.empty.nodes import derive
    cfg = _config_named("WAN21_T2V")
    out = derive(_Model(cfg), 512, 512, 81, 1)
    assert out["downscale_ratio_spacial"] == cfg.latent_format.spacial_downscale_ratio
    assert out["downscale_ratio_temporal"] == cfg.latent_format.temporal_downscale_ratio


# --- the claimed path -----------------------------------------------------

def test_a_model_module_can_claim_a_latent_core_could_not_derive():
    """H3's latent is a NestedTensor whose video channel count is not the
    latent_channels its format reports. Deriving it is not possible; claiming it
    is the whole point of a model-support module."""
    from modules.models import minimax_h3

    built = minimax_h3.empty_latent(_H3Model(), width=1344, height=768, length=124)
    samples = built["samples"]
    assert getattr(samples, "is_nested", False), "not a nested latent"

    video, audio = samples.unbind()
    assert video.shape[1] == 24, "used latent_channels instead of the video channel count"
    assert audio.shape[1] == 32
    assert video.shape[-2:] == (768 // 16, 1344 // 16)


def test_the_h3_module_declines_models_that_are_not_h3():
    """A provider that claimed everything would take the generic path away from
    every other model."""
    from modules.models import minimax_h3
    assert minimax_h3.empty_latent(_Model(_config_named("SDXL")), 512, 512, 1) is None
    assert tuple(minimax_h3.traits(_Model(_config_named("SDXL")))) == ()


class _H3Model:
    """A model tree carrying H3's own classes, so the structural probe fires for
    the reason it would fire on a real one."""

    def __init__(self):
        import comfy.ldm.minimax.model as h3

        class Root:
            def modules(self_inner):
                return [_FakeOf(h3.MiniMaxH3Model), _FakeOf(h3.AdalnProj)]

        class Inner:
            diffusion_model = Root()
            model_config = None

        self.model = Inner()


class _FakeOf:
    """An object reporting another class's identity, so no weights are needed."""

    def __init__(self, cls):
        self.__class__ = type(cls.__name__, (object,), {"__module__": cls.__module__})


# --- execute(), through the real registry ---------------------------------
#
# The dispatch between "claimed" and "derived" is the whole point of this node
# and had no coverage at all: the tests above only ever called derive() and the
# provider directly, never the node that chooses between them.

def test_execute_uses_the_real_registry_and_lets_h3_claim_its_own_latent():
    from modules.latent.empty.nodes import FunPackEmptyLatent

    out = FunPackEmptyLatent.execute(_H3Model(), width=1344, height=768, length=124)
    latent, status = out.result

    assert "model_minimax_h3" in status, f"H3 did not claim its latent: {status!r}"
    video, _audio = latent["samples"].unbind()
    assert video.shape[1] == 24


def test_execute_derives_for_a_model_nothing_claims():
    from modules.latent.empty.nodes import FunPackEmptyLatent

    out = FunPackEmptyLatent.execute(_Model(_config_named("SDXL")), 512, 512, 1)
    latent, status = out.result
    assert "latent format" in status
    assert tuple(latent["samples"].shape) == (1, 4, 64, 64)


def test_a_provider_that_breaks_on_its_own_model_stops_the_node(monkeypatch):
    """The fault this replaced: a claiming provider that raised was treated like
    "not my model", so the node fell through and derived a shape that is wrong
    for exactly the model the provider existed to handle -- and said it worked."""
    from modules.latent.empty import nodes
    from modules.models import minimax_h3

    def broken(model, **kw):
        if not minimax_h3.is_h3(model):
            return None
        raise RuntimeError("upstream moved temporal_shape")

    monkeypatch.setattr(minimax_h3, "empty_latent", broken)
    # The registry holds the function object, so patch what it hands out too.
    from core import registry as registry_mod
    spec = registry_mod.current().specs["model_minimax_h3"]
    monkeypatch.setitem(spec.provides, "empty_latent", broken)

    with pytest.raises(RuntimeError, match="Refusing to substitute"):
        nodes.FunPackEmptyLatent.execute(_H3Model(), width=1344, height=768, length=124)


def test_the_derivation_really_is_wrong_for_h3(monkeypatch):
    """Why the refusal above matters, in numbers. H3's grid is not linear, so the
    generic formula is not close -- it is 16% short, with nothing downstream
    reporting it."""
    from comfy_extras.nodes_minimax_h3 import temporal_shape
    from modules.latent.empty.nodes import derive

    _frames, real_latent_t, _audio_t = temporal_shape(124)

    class H3Format:
        latent_channels = 32
        latent_dimensions = 3
        spacial_downscale_ratio = 16
        temporal_downscale_ratio = 4

    class Cfg:
        latent_format = H3Format()

    derived = derive(_Model(Cfg()), 1344, 768, 124, 1)
    assert derived["samples"].shape[2] != real_latent_t, (
        "if these ever agree, the refusal above is no longer load-bearing")


def test_wan22_matches_what_comfys_own_node_produces():
    """Wan 2.2 changes both the channel count and the spatial ratio from 2.1,
    and neither number appears anywhere in FunPack -- they are read off the
    format. If the derivation were carrying its own copy, this is where it would
    disagree."""
    import torch
    from modules.latent.empty.nodes import derive

    width, height, length = 832, 480, 81
    theirs = torch.zeros([1, 48, ((length - 1) // 4) + 1, height // 16, width // 16])
    ours = derive(_Model(_config_named("WAN22_T2V")), width, height, length, 1)
    assert tuple(ours["samples"].shape) == tuple(theirs.shape)


def test_every_spatial_model_comfyui_ships_gets_a_latent_matching_its_own_format():
    """The whole catalogue, not a list somebody maintained.

    What this proves and what it does not, said plainly. It proves the
    derivation agrees with what each model PUBLISHES, for every config upstream
    ships including ones added after this was written. It does not prove the
    published format is the whole truth: MiniMax H3 reports 32 channels for a
    video branch that has 24, and passes this sweep while being wrong. A model
    whose real latent differs from its format needs a module, and H3's own tests
    are where that is checked -- so passing here is a floor, not a certificate.
    """
    import comfy.supported_models as sm
    from modules.latent.empty.nodes import derive

    checked = 0
    for cfg in sm.models:
        fmt = getattr(cfg, "latent_format", None)
        if fmt is None:
            continue
        rank = getattr(fmt, "latent_dimensions", 2)
        if rank not in (2, 3):
            continue
        out = derive(_Model(cfg), 832, 480, 81, 1)
        shape = tuple(out["samples"].shape)
        assert len(shape) == rank + 2, f"{cfg.__name__}: {shape} is not rank {rank}"
        assert all(dim >= 1 for dim in shape), f"{cfg.__name__}: {shape} has an empty axis"
        assert shape[1] == int(getattr(fmt, "latent_channels", 4)), (
            f"{cfg.__name__}: {shape[1]} channels, format says "
            f"{getattr(fmt, 'latent_channels', 4)}")
        checked += 1

    assert checked > 80, f"only {checked} model configs were reached; upstream moved"


def test_a_one_axis_latent_is_refused_rather_than_guessed():
    """Every rank-1 family upstream ships, by name at the time of writing and by
    enumeration after it.

    The length of a single-axis latent is in the NODE that builds it, never in
    the format: core turns seconds into samples with a 44100 rate and a 2048 hop
    that latent_format does not carry, and the same rank means tokens of geometry
    for Hunyuan3D. Deriving `length` samples gave a StableAudio latent of 1 where
    core's own default is 1024 -- three orders of magnitude short, reported as
    success, which is the shape of failure this node exists to refuse.
    """
    import comfy.supported_models as sm
    from modules.latent.empty.nodes import derive

    families = [cfg for cfg in sm.models
                if getattr(getattr(cfg, "latent_format", None), "latent_dimensions", 2) == 1]
    assert {c.__name__ for c in families} >= {"StableAudio", "ACEStep15", "Hunyuan3Dv2"}, (
        [c.__name__ for c in families])

    for cfg in families:
        with pytest.raises(RuntimeError, match="not derivable"):
            derive(_Model(cfg), 832, 480, 81, 1)
