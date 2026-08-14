"""Tests for ComfyUI workflow import (parse + apply bindings)."""
import re
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


# ── subgraphs ────────────────────────────────────────────────────────────────
# A subgraph instance is a node whose type is a uuid from definitions.subgraphs. Taking
# that uuid for a class name loses every node inside it and queues a class ComfyUI has
# never heard of. ComfyUI's own MiniMax H3 templates ship this way.

def _subgraph_workflow():
    """Outer: LoadImage -> [subgraph] -> SaveImage. Inner: CLIPTextEncode -> KSampler.

    The instance's promoted `text` says "outer"; the inner node still says "stale", which
    is what a real template looks like once the promoted widget has been edited.
    """
    return {
        "nodes": [
            {"id": 1, "type": "LoadImage", "widgets_values": ["a.png"],
             "inputs": [], "outputs": [{"name": "IMAGE", "links": [10]}]},
            {"id": 2, "type": "sub-uuid", "widgets_values": ["outer"],
             "inputs": [{"name": "img", "type": "IMAGE", "link": 10}],
             "outputs": [{"name": "LATENT", "links": [11]}]},
            {"id": 3, "type": "SaveImage", "widgets_values": ["out"],
             "inputs": [{"name": "images", "type": "IMAGE", "link": 11}], "outputs": []},
        ],
        "links": [[10, 1, 0, 2, 0, "IMAGE"], [11, 2, 0, 3, 0, "IMAGE"]],
        "definitions": {"subgraphs": [{
            "id": "sub-uuid", "name": "inner",
            "inputs": [{"name": "img", "type": "IMAGE"}, {"name": "text", "type": "STRING"}],
            "outputs": [{"name": "LATENT", "type": "LATENT"}],
            "nodes": [
                {"id": 20, "type": "CLIPTextEncode", "widgets_values": ["stale"],
                 "inputs": [{"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": 31},
                            {"name": "clip", "type": "CLIP", "link": None}],
                 "outputs": [{"name": "CONDITIONING", "links": [32]}]},
                {"id": 21, "type": "KSampler", "widgets_values": [7],
                 "inputs": [{"name": "positive", "type": "CONDITIONING", "link": 32},
                            {"name": "latent_image", "type": "LATENT", "link": 30}],
                 "outputs": [{"name": "LATENT", "links": [33]}]},
            ],
            "links": [
                {"id": 30, "origin_id": -10, "origin_slot": 0, "target_id": 21, "target_slot": 1, "type": "IMAGE"},
                {"id": 31, "origin_id": -10, "origin_slot": 1, "target_id": 20, "target_slot": 0, "type": "STRING"},
                {"id": 32, "origin_id": 20, "origin_slot": 0, "target_id": 21, "target_slot": 0, "type": "CONDITIONING"},
                {"id": 33, "origin_id": 21, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "LATENT"},
            ],
        }]},
    }


def _classes(parsed):
    return sorted(s["node_class"] for s in parsed["slots"])


def _pairs(parsed):
    """Link endpoints as Class.socket — the label carries a " #id" suffix to tell two
    nodes of the same class apart, which these tests do not care about."""
    strip = lambda s: re.sub(r" #\S+?\.", ".", s)
    return {(strip(l["from"]), strip(l["to"])) for l in parsed["links"]}


def test_a_subgraph_is_expanded_into_the_nodes_inside_it():
    parsed = workflow_import.parse_workflow(_subgraph_workflow(), OI)
    assert _classes(parsed) == ["CLIPTextEncode", "KSampler", "LoadImage", "SaveImage"]
    assert not any(s["node_class"] == "sub-uuid" for s in parsed["slots"])


def test_the_promoted_widget_wins_over_the_inner_nodes_stale_copy():
    """The instance carries the value the user actually set; the inner node keeps whatever
    it had when the subgraph was authored."""
    parsed = workflow_import.parse_workflow(_subgraph_workflow(), OI)
    enc = next(s for s in parsed["slots"] if s["node_class"] == "CLIPTextEncode")
    assert enc["inputs"]["text"] == "outer"


def test_links_cross_the_subgraph_boundary_in_both_directions():
    parsed = workflow_import.parse_workflow(_subgraph_workflow(), OI)
    pairs = _pairs(parsed)
    assert ("LoadImage.IMAGE", "KSampler.latent_image") in pairs      # outer -> inner
    assert ("KSampler.LATENT", "SaveImage.images") in pairs           # inner -> outer
    assert ("CLIPTextEncode.CONDITIONING", "KSampler.positive") in pairs   # inner -> inner


def test_a_workflow_without_subgraphs_is_untouched():
    """The flattener must be a no-op for every workflow that predates subgraphs."""
    plain = {
        "nodes": [{"id": 1, "type": "LoadImage", "widgets_values": ["a.png"],
                   "inputs": [], "outputs": [{"name": "IMAGE", "links": [1]}]},
                  {"id": 2, "type": "SaveImage", "widgets_values": ["out"],
                   "inputs": [{"name": "images", "type": "IMAGE", "link": 1}], "outputs": []}],
        "links": [[1, 1, 0, 2, 0, "IMAGE"]],
    }
    parsed = workflow_import.parse_workflow(plain, OI)
    assert _classes(parsed) == ["LoadImage", "SaveImage"]
    assert _pairs(parsed) == {("LoadImage.IMAGE", "SaveImage.images")}


def test_an_unconnected_subgraph_input_leaves_the_widget_value_alone():
    """A promoted widget with nothing wired into it must not produce a dangling link."""
    wf = _subgraph_workflow()
    wf["nodes"][1]["inputs"][0]["link"] = None      # nothing feeds `img` any more
    wf["links"] = [l for l in wf["links"] if l[0] != 10]
    parsed = workflow_import.parse_workflow(wf, OI)
    assert "KSampler" in _classes(parsed)
    assert not any(t == "KSampler.latent_image" for _f, t in _pairs(parsed))


# ── Set/Get virtual link nodes ───────────────────────────────────────────────
# KJNodes' Set/Get exist only in web/js/setgetnodes.js — there is no Python class, so the
# frontend resolves them away when building a prompt and emitting one queues a node type
# the backend has never heard of. Big LTX workflows lean on them heavily.

def _setget_workflow():
    """LoadImage -> Set("img") ... Get("img") -> SaveImage, plus a Get with no Set."""
    return {
        "nodes": [
            {"id": 1, "type": "LoadImage", "widgets_values": ["a.png"],
             "inputs": [], "outputs": [{"name": "IMAGE", "links": [1]}]},
            {"id": 2, "type": "SetNode", "widgets_values": ["img"],
             "inputs": [{"name": "value", "type": "IMAGE", "link": 1}],
             "outputs": [{"name": "IMAGE", "links": []}]},
            {"id": 3, "type": "GetNode", "widgets_values": ["img"],
             "inputs": [], "outputs": [{"name": "IMAGE", "links": [2]}]},
            {"id": 4, "type": "SaveImage", "widgets_values": ["out"],
             "inputs": [{"name": "images", "type": "IMAGE", "link": 2}], "outputs": []},
            {"id": 5, "type": "GetNode", "widgets_values": ["nothing-sets-this"],
             "inputs": [], "outputs": [{"name": "IMAGE", "links": [3]}]},
            {"id": 6, "type": "SaveImage", "widgets_values": ["out2"],
             "inputs": [{"name": "images", "type": "IMAGE", "link": 3}], "outputs": []},
        ],
        "links": [[1, 1, 0, 2, 0, "IMAGE"], [2, 3, 0, 4, 0, "IMAGE"], [3, 5, 0, 6, 0, "IMAGE"]],
    }


def test_set_get_nodes_are_resolved_away():
    parsed = workflow_import.parse_workflow(_setget_workflow(), OI)
    assert not [s for s in parsed["slots"] if s["node_class"] in ("SetNode", "GetNode")]


def test_the_link_reconnects_across_the_set_get_pair():
    parsed = workflow_import.parse_workflow(_setget_workflow(), OI)
    assert ("LoadImage.IMAGE", "SaveImage.images") in _pairs(parsed)


def test_a_get_with_no_matching_set_leaves_the_input_unwired():
    """Better an unwired input the user can see than a link to a node that cannot exist."""
    parsed = workflow_import.parse_workflow(_setget_workflow(), OI)
    # exactly one SaveImage is fed; the one behind the orphan Get is not
    fed = [t for _f, t in _pairs(parsed) if t == "SaveImage.images"]
    assert len(fed) == 1


def test_a_workflow_without_set_get_is_untouched():
    plain = {
        "nodes": [{"id": 1, "type": "LoadImage", "widgets_values": ["a.png"],
                   "inputs": [], "outputs": [{"name": "IMAGE", "links": [1]}]},
                  {"id": 2, "type": "SaveImage", "widgets_values": ["out"],
                   "inputs": [{"name": "images", "type": "IMAGE", "link": 1}], "outputs": []}],
        "links": [[1, 1, 0, 2, 0, "IMAGE"]],
    }
    assert _pairs(workflow_import.parse_workflow(plain, OI)) == {("LoadImage.IMAGE", "SaveImage.images")}


def test_a_set_get_chain_terminates():
    """Set -> Get -> Set -> Get is legal; a Get whose Set loops back to it must not hang."""
    wf = {
        "nodes": [
            {"id": 1, "type": "GetNode", "widgets_values": ["a"],
             "inputs": [], "outputs": [{"name": "IMAGE", "links": [1]}]},
            {"id": 2, "type": "SetNode", "widgets_values": ["a"],
             "inputs": [{"name": "value", "type": "IMAGE", "link": 1}], "outputs": []},
            {"id": 3, "type": "GetNode", "widgets_values": ["a"],
             "inputs": [], "outputs": [{"name": "IMAGE", "links": [2]}]},
            {"id": 4, "type": "SaveImage", "widgets_values": ["out"],
             "inputs": [{"name": "images", "type": "IMAGE", "link": 2}], "outputs": []},
        ],
        "links": [[1, 1, 0, 2, 0, "IMAGE"], [2, 3, 0, 4, 0, "IMAGE"]],
    }
    parsed = workflow_import.parse_workflow(wf, OI)     # must return, not spin
    assert _classes(parsed) == ["SaveImage"]
