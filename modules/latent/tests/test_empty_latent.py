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
