"""Every weight dtype, compute dtype and attention the Diffusion Model Loader offers, on a real (tiny) H3 file.

The file is written from ComfyUI's own `model_base.MiniMaxH3` at a few thousand parameters, so ComfyUI's real
detection and loading path reads it. Each combination must come back as a model, not an exception.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture(scope="module")
def tiny_file(comfyui, tmp_path_factory):
    import comfy.supported_models as sm
    import folder_paths
    import safetensors.torch as st
    torch.manual_seed(0)
    config = sm.MiniMaxH3(dict(
        image_model="minimax_h3", hidden_size=64, num_layers=4, token_refiner_num_layers=1, num_attention_heads=2,
        attention_head_dim=32, ffn_hidden_size=64, text_dim=48, timestep_input_dim=32, time_embed_hidden_size=64,
        time_embed_dim=32, rope_inv_freq_len=4))
    config.set_inference_dtype(torch.float32, None)
    base = config.get_model({}, device=torch.device("cpu"))
    for p in base.diffusion_model.parameters():
        torch.nn.init.normal_(p, std=0.05)
    folder = tmp_path_factory.mktemp("unet")
    st.save_file({k: v.contiguous() for k, v in base.diffusion_model.state_dict().items()}, str(folder / "tiny_h3.safetensors"))
    folder_paths.add_model_folder_path("diffusion_models", str(folder))
    return "tiny_h3.safetensors"


def _choices():
    from modules.loaders.common import COMPUTE_DTYPES, WEIGHT_DTYPES, attention_choices
    return list(WEIGHT_DTYPES), list(COMPUTE_DTYPES), list(attention_choices())


def pytest_generate_tests(metafunc):
    if "combo" in metafunc.fixturenames:
        w, c, a = _choices() if _importable() else ([], [], [])
        metafunc.parametrize("combo", [(x, y, z) for x in w for y in c for z in a], ids=lambda t: "-".join(t))


def _importable():
    try:
        from modules.loaders.common import COMPUTE_DTYPES  # noqa: F401
        return True
    except Exception:                                       # noqa: BLE001 -- collection must not fail without ComfyUI
        return False


def test_every_dtype_and_attention_combination_loads(tiny_file, combo):
    from modules.loaders.diffusion_model.nodes import FunPackDiffusionModelLoader
    weight, compute, attention = combo
    model, status = FunPackDiffusionModelLoader.execute(tiny_file, weight, compute, attention).result
    assert type(model.model).__name__ == "MiniMaxH3"
    assert weight in status
