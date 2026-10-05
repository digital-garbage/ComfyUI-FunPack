"""The SD1.5 module: recognising the file, and a pipeline that builds into a real graph."""

import asyncio
from pathlib import Path

import pytest

UNET = "model.diffusion_model.input_blocks.0.0.weight"
CLIP_L = "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"


@pytest.fixture(scope="module")
def registered(comfyui):
    import nodes as comfy_nodes

    async def load():
        await comfy_nodes.init_extra_nodes(init_custom_nodes=False)
        await comfy_nodes.load_custom_node(str(Path(__file__).resolve().parents[4]), module_parent="custom_nodes")

    asyncio.run(load())
    return comfy_nodes


def test_sd1_is_recognised_and_sd2_sdxl_h3_are_not(comfyui):
    from modules.models import sd15
    assert sd15.detect({UNET, CLIP_L})
    assert sd15.probe_traits({UNET, CLIP_L}) == ["spatial_latent", "predict_eps"]
    assert not sd15.detect({UNET, "cond_stage_model.model.token_embedding.weight"})        # SD2 (OpenCLIP)
    assert not sd15.detect({UNET, "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight"})  # SDXL
    assert not sd15.detect({"video_patch_proj.weight", "audio_patch_proj.weight"})       # H3
    assert sd15.probe_traits({UNET}) == []


def test_the_pipeline_builds_a_real_graph_and_is_offered(registered):
    from core import graph, routes
    from modules.models.sd15 import text_to_image

    _prompt, problems = graph.build(text_to_image())
    assert [p for p in problems if "ckpt_name" not in p and "is '', which is not one of" not in p] == []
    assert "sd15_text_to_image" in {p["id"] for _s, make in routes.modules().providers("pipeline_presets") for p in make()}
