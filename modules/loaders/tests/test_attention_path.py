"""The attention override, carried all the way to a real attention call.

This is the seam the loader is most likely to get silently wrong: a bad override
loads perfectly and then fails at the first attention call, which is sampling.
None of it needs model weights, so there is no excuse for not running it.
"""

import pytest


@pytest.fixture(scope="module")
def torch_and_attention(comfyui):
    import torch
    import comfy.ldm.modules.attention as attention
    return torch, attention


@pytest.fixture
def qkv(torch_and_attention):
    torch, _ = torch_and_attention
    torch.manual_seed(0)
    heads, dim = 2, 16
    shape = (1, 8, heads * dim)
    return torch.randn(shape), torch.randn(shape), torch.randn(shape), heads


def test_every_backend_this_machine_offers_survives_the_override(torch_and_attention, qkv):
    """The override is handed the WRAPPED function plus the wrapper's own kwargs
    (`_inside_attn_wrapper` among them). If the unwrapped backend chokes on any
    of that, choosing a backend breaks sampling and nothing else."""
    torch, attention = torch_and_attention
    from modules.loaders.common import attention_override

    q, k, v, heads = qkv
    reference = attention.get_attention_function("pytorch")(q, k, v, heads=heads)

    for name in sorted(attention.REGISTERED_ATTENTION_FUNCTIONS):
        override = attention_override(name)
        assert override is not None, f"{name} is registered but produced no override"
        out = override(attention.get_attention_function(name), q, k, v, heads=heads,
                       transformer_options={}, _inside_attn_wrapper=True)
        assert out.shape == reference.shape, f"{name} changed the output shape"


def test_an_installed_override_is_actually_reached_by_comfys_dispatch(torch_and_attention, qkv):
    """Proves the KEY the loader writes is the key comfy reads. A typo here is
    invisible: the model loads, sampling runs, and the chosen backend is simply
    never used."""
    torch, attention = torch_and_attention
    from modules.loaders.common import attention_override

    q, k, v, heads = qkv
    inner = attention_override("pytorch")
    reached = []

    def spy(func, *args, **kwargs):
        reached.append(True)
        return inner(func, *args, **kwargs)

    # Exactly the dict the diffusion loader writes into model_options.
    transformer_options = {"optimized_attention_override": spy}
    out = attention.optimized_attention(q, k, v, heads=heads,
                                        transformer_options=transformer_options)

    assert reached, "the override was installed where nothing reads it"
    plain = attention.optimized_attention(q, k, v, heads=heads, transformer_options={})
    assert torch.allclose(out, plain, atol=1e-5), (
        "routing through the override changed the result")


def test_the_loader_writes_the_override_where_dispatch_looks(comfyui, monkeypatch):
    """The loader's own installation step, without loading a model: the key name
    and its nesting under model_options are what this pins."""
    import comfy.sd
    import comfy.utils
    import folder_paths
    from comfy.ldm.modules import attention as attn
    from modules.loaders.diffusion_model import nodes

    class FakeModel:
        def __init__(self):
            self.model_options = {}

    def wrapped(*args, **kwargs):
        raise AssertionError("called the wrapped function, re-entering wrap_attn")

    wrapped.__wrapped__ = lambda *a, **k: "attended"
    monkeypatch.setattr(attn, "get_attention_function", lambda name, default: wrapped)
    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda kind, name: f"/fake/{name}")
    monkeypatch.setattr(comfy.utils, "load_torch_file", lambda p, **kw: ({}, None))
    monkeypatch.setattr(comfy.sd, "load_diffusion_model_state_dict",
                        lambda sd, **kw: FakeModel())

    out = nodes.FunPackDiffusionModelLoader.execute(
        model_name="m.safetensors", weight_dtype="default",
        compute_dtype="default", attention="pytorch")

    model = out.result[0]
    installed = model.model_options["transformer_options"]["optimized_attention_override"]
    assert callable(installed)
    assert installed(wrapped, "q") == "attended"


def test_a_model_comfy_could_not_identify_fails_at_the_loader(comfyui, monkeypatch):
    """comfy returns None instead of raising, and a None model reaches the
    sampler before anything complains."""
    import comfy.sd
    import comfy.utils
    import folder_paths
    from modules.loaders.diffusion_model import nodes

    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda kind, name: f"/fake/{name}")
    monkeypatch.setattr(comfy.utils, "load_torch_file", lambda p, **kw: ({}, None))
    monkeypatch.setattr(comfy.sd, "load_diffusion_model_state_dict", lambda sd, **kw: None)

    with pytest.raises(RuntimeError, match="Could not detect a diffusion model"):
        nodes.FunPackDiffusionModelLoader.execute(
            model_name="actually_a_vae.safetensors", weight_dtype="default",
            compute_dtype="default", attention="default")
