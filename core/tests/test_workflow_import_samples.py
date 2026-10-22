"""Workflow import against ComfyUI's own sample workflows and its real nodes.

The unit tests in test_workflow_import.py use stand-in nodes. This file reads the workflows ComfyUI ships as
templates (the `comfyui-workflow-templates` package, part of ComfyUI's own requirements) and converts each with the
real node schemas, so a node, a dynamic input or a widget layout the stand-ins do not imagine is met here.

What it checks is structure: every node is known, every wire fits, every typed-in value is in range. A model or
other file the template names is not installed on a test machine, so a "which is not one of" about a file is the
one thing allowed through.
"""

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("comfyui_workflow_templates_json")

from core import graph, workflow_import  # noqa: E402

FILE_WORDS = re.compile(r"\.(safetensors|png|jpg|jpeg|mp4|webp|ckpt|pt|gguf|wav|mp3|flac|mov|glb|bin|pth|json)'|pixel_space|kitchen")
CLOUD = re.compile(r"Gemini|Grok|ByteDance|Kling|OpenAI|Eleven|Seedream")        # partner nodes: dynamic forms of their own

VIDEO = re.compile(r"^(video_|ltx|template_ltx|text_to_video|image_to_video|hunyuan_video|wan|template_image_speech)")


@pytest.fixture(scope="module")
def schemas(comfyui):
    import asyncio
    import nodes as comfy_nodes

    async def load():
        await comfy_nodes.init_extra_nodes(init_custom_nodes=False)
        await comfy_nodes.load_custom_node(str(Path(__file__).resolve().parents[2]), module_parent="custom_nodes")

    asyncio.run(load())
    return graph.from_comfyui()


def _templates():
    import comfyui_workflow_templates_json as pack
    folder = Path(pack.__file__).parent / "templates"
    return sorted(p for p in folder.glob("*.json") if VIDEO.match(p.stem))


def _convert(path, schemas):
    wf = json.loads(path.read_text(encoding="utf-8"))
    out = workflow_import.convert(wf, schemas)
    return out, graph.build(out["slots"], schemas)[1]


def test_there_are_video_templates_to_test():
    assert len(_templates()) > 30


@pytest.mark.parametrize("path", _templates(), ids=lambda p: p.stem)
def test_a_video_template_imports_and_every_node_wire_and_value_fits(path, schemas):
    out, problems = _convert(path, schemas)
    assert out["slots"], "nothing imported"
    left = [p for p in problems if not FILE_WORDS.search(p) and not CLOUD.search(p)]
    # video_ltx2_i2v_lora ships with a subgraph output whose link is missing from the file itself
    if path.stem == "video_ltx2_i2v_lora":
        left = [p for p in left if "SaveVideo needs 'video'" not in p]
    assert not left, left[:5]


@pytest.mark.parametrize("name", ["video_minimax_h3_t2v", "video_minimax_h3_i2v", "video_minimax_h3_r2v", "video_ltx2_5_t2v",
                                  "video_ltx2_3_i2v", "video_wan2_2_14B_t2v", "text_to_video_wan"])
def test_the_prompt_and_seed_of_the_main_video_templates_get_a_control(name, schemas):
    folder = _templates()[0].parent
    out, _ = _convert(folder / f"{name}.json", schemas)
    assert "prompt" in out["bound"] and "seed" in out["bound"], out["bound"]


def test_r2v_offers_exactly_the_inputs_the_node_builds(schemas):
    got = set(schemas.inputs("MiniMaxH3ReferenceToVideo"))
    names = lambda prefix, n: {f"{prefix}{i}" for i in range(n)}               # noqa: E731
    assert {f"ref_images.ref_image_{i}" for i in range(9)} <= got and "ref_images.ref_image_9" not in got
    assert {f"ref_videos.ref_video_{i}" for i in range(3)} <= got and "ref_videos.ref_video_3" not in got
    assert {f"ref_video_audios.ref_video_audio_{i}" for i in range(3)} <= got and "ref_video_audios.ref_video_audio_3" not in got
    assert {f"ref_audios.ref_audio_{i}" for i in range(3)} <= got and "ref_audios.ref_audio_3" not in got
    assert names


def test_autogrow_inputs_of_other_nodes_are_named_as_comfyui_names_them(schemas):
    assert "images.image0" in schemas.inputs("BatchImagesNode")                 # numbered from 0, whatever the minimum
    assert "values.a" in schemas.inputs("ComfyMathExpression") and "values.z" in schemas.inputs("ComfyMathExpression")
    assert schemas.wire_type("ComfyMathExpression", "values.a") == "FLOAT,INT,BOOLEAN"       # an INT may feed it
    assert "images.image0" in schemas.inputs("ReferenceMultiLatent")            # the wrapped input is an optional one there


def test_a_widget_that_also_takes_a_wire_accepts_either_type(schemas):
    assert "INT" in schemas.wire_type("LTXVEmptyLatentAudio", "frame_rate")
