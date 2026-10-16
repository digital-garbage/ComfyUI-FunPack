"""compute_dtype, carried to the layer that would fail.

This branch had no coverage at all, and the bug in it was a straight port from
v4: setting the compute dtype and then clearing `force_cast_weights` casts the
model's INPUT without casting its WEIGHTS, so the first Linear of the first
sampling step raises a dtype mismatch. It survives on constrained hardware only
because the low-VRAM path turns the cast back on by itself.

No model weights are needed: the failure is in ComfyUI's real ops and patcher,
both of which run fine on placeholder-free stubs.
"""

import pytest


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """These import comfy directly."""


def _load_with(monkeypatch, compute_dtype, model):
    import comfy.sd
    import comfy.utils
    import folder_paths
    from modules.loaders.diffusion_model import nodes

    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda kind, name: f"/fake/{name}")
    monkeypatch.setattr(comfy.utils, "load_torch_file", lambda p, **kw: ({}, None))
    monkeypatch.setattr(comfy.sd, "load_diffusion_model_state_dict", lambda sd, **kw: model)
    return nodes.FunPackDiffusionModelLoader.execute(
        model_name="m.safetensors", weight_dtype="default",
        compute_dtype=compute_dtype, attention="default")


def test_a_chosen_compute_dtype_leaves_the_weight_cast_enabled(monkeypatch):
    """The invariant. ComfyUI sets force_cast_weights inside
    set_model_compute_dtype on purpose -- it is what casts each layer's weights
    to match the input. Clearing it is what breaks sampling."""
    import torch
    import comfy.model_patcher

    class Stub(torch.nn.Module):
        pass

    patcher = comfy.model_patcher.ModelPatcher(
        Stub(), load_device=torch.device("cpu"), offload_device=torch.device("cpu"))

    _load_with(monkeypatch, "bf16", patcher)

    assert patcher.model_options is not None
    assert patcher.force_cast_weights is True, (
        "the weight cast was disabled, so the model will cast its input to "
        "bf16 and leave the weights alone")
    assert patcher.object_patches.get("manual_cast_dtype") is torch.bfloat16


def test_default_compute_dtype_touches_nothing(monkeypatch):
    import torch
    import comfy.model_patcher

    class Stub(torch.nn.Module):
        pass

    patcher = comfy.model_patcher.ModelPatcher(
        Stub(), load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    before = patcher.force_cast_weights

    _load_with(monkeypatch, "default", patcher)

    assert patcher.force_cast_weights == before
    assert "manual_cast_dtype" not in patcher.object_patches


def test_the_mismatch_this_prevents_is_real():
    """What the first sampling step does when the input is cast and the weights
    are not. Pinned so the invariant above has a stated consequence rather than
    just a boolean."""
    import torch
    import comfy.ops

    # The ops ComfyUI picks when a model needs no manual cast at construction --
    # the ordinary full-VRAM load.
    layer = comfy.ops.disable_weight_init.Linear(8, 8)
    assert layer.comfy_cast_weights is False

    layer.weight = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.float32))
    layer.bias = torch.nn.Parameter(torch.zeros(8, dtype=torch.float32))

    # _apply_model casts the activation to manual_cast_dtype.
    activation = torch.randn(2, 8, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="same dtype"):
        layer(activation)
