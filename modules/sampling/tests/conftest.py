"""Stub models for the sampling tests.

A patcher whose model announces nothing offers no traits, and every modifier
that declares one is then filtered out -- so a test using a bare stub would pass
by installing nothing at all and prove the opposite of what it says. The stubs
here announce a latent rank, which is the trait modifiers actually narrow on.
"""

import pytest


def patcher_reporting(latent_dimensions):
    import torch
    from comfy.model_patcher import ModelPatcher

    class Format:
        pass

    class Config:
        pass

    class Stub(torch.nn.Module):
        pass

    stub = Stub()
    fmt = Format()
    fmt.latent_dimensions = latent_dimensions
    config = Config()
    config.latent_format = fmt
    stub.model_config = config
    return ModelPatcher(stub, load_device=torch.device("cpu"),
                        offload_device=torch.device("cpu"))


@pytest.fixture
def patcher(comfyui):
    """A model with a two-axis latent: what an image model reports."""
    return patcher_reporting(2)


@pytest.fixture
def temporal_patcher(comfyui):
    """A model with a time axis."""
    return patcher_reporting(3)
