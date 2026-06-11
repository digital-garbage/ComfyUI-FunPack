"""Tests for ComfyUI workflow import (parse + apply bindings)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from movie_editor.backend import workflow_import  # noqa: E402

OI = {
    "CLIPTextEncode": {
        "input": {"required": {"text": ["STRING", {"multiline": True}], "clip": ["CLIP"]}},
        "output": ["CONDITIONING"],
        "output_name": ["CONDITIONING"],
    },
    "KSampler": {
        "input": {"required": {
            "seed": ["INT", {"default": 0}],
            "model": ["MODEL"],
            "positive": ["CONDITIONING"],
            "negative": ["CONDITIONING"],
            "latent_image": ["LATENT"],
        }},
        "output": ["LATENT"],
        "output_name": ["LATENT"],
    },
    "LoadImage": {
        "input": {"required": {"image": [["a.png", "b.png"]]}},
        "output": ["IMAGE", "MASK"],
        "output_name": ["IMAGE", "MASK"],
    },
    "EmptyLatentImage": {
        "input": {"required": {"width": ["INT", {"default": 512}], "height": ["INT", {"default": 512}], "batch_size": ["INT", {"default": 1}]}},
        "output": ["LATENT"],
        "output_name": ["LATENT"],
    },
    "VAEDecode": {
        "input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}},
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
    },
    "VHS_VideoCombine": {
        "input": {"required": {"images": ["IMAGE"], "frame_rate": ["FLOAT", {"default": 30}]}},
        "output": ["VHS_FILENAMES"],
        "output_name": ["Filenames"],
    },
    "ImageScale": {
        "input": {"required": {"image": ["IMAGE"], "width": ["INT"], "height": ["INT"]}},
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
    },
}

UI_WORKFLOW = {
    "nodes": [
        {"id": 1, "type": "CLIPTextEncode", "widgets_values": ["hello world"], "inputs": [], "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [10]}]},
        {"id": 2, "type": "CLIPTextEncode", "widgets_values": ["bad"], "inputs": [], "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [11]}]},
        {"id": 3, "type": "LoadImage", "widgets_values": ["a.png"], "inputs": [], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [12]}]},
        {"id": 4, "type": "ImageScale", "widgets_values": [512, 512], "inputs": [
            {"name": "image", "type": "IMAGE", "link": 12},
        ], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [13]}]},
        {"id": 5, "type": "VAEDecode", "widgets_values": [], "inputs": [
            {"name": "samples", "type": "LATENT", "link": None},
            {"name": "vae", "type": "VAE", "link": None},
        ], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}]},
        {"id": 6, "type": "VHS_VideoCombine", "widgets_values": [30], "inputs": [
            {"name": "images", "type": "IMAGE", "link": 13},
        ], "outputs": [{"name": "Filenames", "type": "VHS_FILENAMES", "links": []}]},
    ],
    "links": [
        [10, 1, 0, 99, 0, "CONDITIONING"],
        [11, 2, 0, 99, 1, "CONDITIONING"],
        [12, 3, 0, 4, 0, "IMAGE"],
        [13, 4, 0, 6, 0, "IMAGE"],
    ],
    "extra": {"workflow_name": "Test workflow"},
}


API_WORKFLOW = {
    "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "pos"}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "neg"}},
    "3": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["8", 0]}},
    "8": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
    "9": {"class_type": "EmptyLatentImage", "inputs": {"width": 768, "height": 512, "batch_size": 1}},
}


def test_parse_ui_workflow_nodes_and_links():
    parsed = workflow_import.parse_workflow(UI_WORKFLOW, OI)
    assert parsed["format"] == "ui"
    assert parsed["name"] == "Test workflow"
    assert parsed["node_count"] == 6
    assert parsed["link_count"] >= 2
    ids = {s["node_class"] for s in parsed["slots"]}
    assert "CLIPTextEncode" in ids
    assert "LoadImage" in ids
    scale = next(s for s in parsed["slots"] if s["node_class"] == "ImageScale")
    assert scale["input_sources"].get("image", "").startswith("out:w3:")


def test_suggest_bindings():
    parsed = workflow_import.parse_workflow(UI_WORKFLOW, OI)
    sug = parsed["suggestions"]
    assert sug.get("prompt", "").startswith("link:")
    assert sug.get("negative_prompt", "").startswith("link:")
    assert sug.get("timeline_image", "").startswith("source:w4:")


def test_apply_bindings_disable_core():
    parsed = workflow_import.parse_workflow(UI_WORKFLOW, OI)
    bindings = {
        "prompt": parsed["suggestions"].get("prompt", ""),
        "negative_prompt": parsed["suggestions"].get("negative_prompt", ""),
        "video_output": parsed["suggestions"].get("video_output", ""),
    }
    config = workflow_import.apply_bindings(parsed, bindings, OI)
    assert config["disable_core"] is True
    assert len(config["slots"]) == 6
    assert config["workflow_import"]["name"] == "Test workflow"
    prompt_links = [l for l in config["links"] if l.get("editor_key") == "prompt"]
    assert prompt_links and prompt_links[0]["source"] == "editor"
    scale = next(s for s in config["slots"] if s["node_class"] == "ImageScale")
    # ImageScale receives LoadImage via preserved internal link, not timeline override
    assert scale["input_sources"].get("image", "").startswith("out:w3:")


def test_apply_timeline_binding():
    parsed = workflow_import.parse_workflow(UI_WORKFLOW, OI)
    scale = next(s for s in parsed["slots"] if s["node_class"] == "ImageScale")
    bindings = {"timeline_image": f"source:{scale['id']}:image"}
    config = workflow_import.apply_bindings(parsed, bindings, OI)
    scale2 = next(s for s in config["slots"] if s["node_class"] == "ImageScale")
    assert scale2["input_sources"].get("image") == "timeline"


def test_parse_api_workflow():
    parsed = workflow_import.parse_workflow(API_WORKFLOW, OI)
    assert parsed["format"] == "api"
    assert parsed["node_count"] == 5
    decode = next(s for s in parsed["slots"] if s["node_class"] == "VAEDecode")
    assert "samples" in decode["input_sources"]


def test_apply_workflow_end_to_end():
    config = workflow_import.apply_workflow(
        API_WORKFLOW,
        {"prompt": "link:w1:text", "width": "link:w9:width"},
        OI,
    )
    assert config["disable_core"] is True
    assert any(l.get("editor_key") == "width" for l in config["links"])
