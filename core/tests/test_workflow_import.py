import pytest

from core import workflow_import as wi
from core.graph import Schemas

NODES = {
    "CLIPTextEncode": {"inputs": {"text": "STRING", "clip": "CLIP"}, "outputs": ["CONDITIONING"]},
    "KSampler": {"inputs": {"model": "MODEL", "positive": "CONDITIONING", "negative": "CONDITIONING", "latent_image": "LATENT",
                            "seed": "INT", "steps": "INT", "cfg": "FLOAT", "sampler_name": "COMBO"}, "outputs": ["LATENT"]},
    "EmptyLatentImage": {"inputs": {"width": "INT", "height": "INT", "batch_size": "INT"}, "outputs": ["LATENT"]},
    "LoadImage": {"inputs": {"image": "COMBO"}, "outputs": ["IMAGE", "MASK"]},
    "VAEDecode": {"inputs": {"samples": "LATENT"}, "outputs": ["IMAGE"]},
    "ImageScale": {"inputs": {"image": "IMAGE", "width": "INT"}, "outputs": ["IMAGE"]},
}
S = Schemas(lambda cls: NODES.get(cls))


def node(i, cls, inputs=(), values=None, **extra):
    return {"id": i, "type": cls, "inputs": [{"name": i[0], "link": i[1], **({"type": i[2]} if len(i) > 2 else {})} for i in inputs], "widgets_values": values or [], **extra}


def link(i, a, aslot, b, bslot, typ="X"):
    return [i, a, aslot, b, bslot, typ]


def ui():
    return {"nodes": [
        node(1, "CLIPTextEncode", [("clip", None)], ["a red fox"]),
        node(2, "CLIPTextEncode", [("clip", None)], ["blurry"]),
        node(3, "Reroute", [("", 10)]),
        node(4, "EmptyLatentImage", [], [768, 512, 1]),
        node(5, "KSampler", [("positive", 11), ("negative", 12), ("latent_image", 13)], [42, "randomize", 20, 7.0, "euler"]),
        node(6, "Note", [], ["hello"]),
    ], "links": [link(10, 1, 0, 3, 0), link(11, 3, 0, 5, 0), link(12, 2, 0, 5, 1), link(13, 4, 0, 5, 2)]}


def test_ui_export_becomes_slots_with_values_and_links_stepping_over_reroutes_and_the_seed_control():
    out = wi.convert(ui(), S)
    by = {s["id"]: s for s in out["slots"]}
    assert set(by) == {"w1", "w2", "w4", "w5"}                                  # Reroute and Note are gone
    assert by["w5"]["inputs"]["positive"] == ["w1", 0] and by["w5"]["inputs"]["negative"] == ["w2", 0]
    assert by["w5"]["inputs"]["seed"] == 42 and by["w5"]["inputs"]["steps"] == 20 and by["w5"]["inputs"]["cfg"] == 7.0
    assert by["w5"]["inputs"]["sampler_name"] == "euler"                       # "randomize" was the control, not a value


def test_the_apps_controls_find_their_nodes():
    out = wi.convert(ui(), S)
    assert out["bound"] == {"prompt": "CLIPTextEncode #1.text", "negative": "CLIPTextEncode #2.text", "seed": "KSampler #5.seed",
                            "width": "EmptyLatentImage #4.width", "height": "EmptyLatentImage #4.height"}
    roles = {s["id"]: s.get("roles") for s in out["slots"]}
    assert roles["w1"] == [{"at": "generation.prompt", "label": "Prompt", "input": "text"}]
    assert roles["w2"][0]["at"] == "project.negative"


def test_api_export_works_and_a_load_image_becomes_the_start_picture():
    api = {"1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
           "2": {"class_type": "ImageScale", "inputs": {"image": ["1", 0], "width": 512}}}
    out = wi.convert(api, S)
    by = {s["id"]: s for s in out["slots"]}
    assert by["w1"]["node"] == "FunPackLoadMedia" and by["w1"]["roles"][0]["at"] == "assets.source_image"
    assert by["w2"]["inputs"]["image"] == ["w1", 0]
    assert out["bound"]["image"] == "FunPackLoadMedia #1.media_id"


def test_a_load_image_whose_mask_is_used_is_left_alone():
    api = {"1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
           "2": {"class_type": "VAEDecode", "inputs": {"samples": ["1", 1]}}}
    assert wi.convert(api, S)["slots"][0]["node"] == "LoadImage"


def test_muted_nodes_drop_out_and_say_what_was_left_unconnected_bypassed_ones_pass_through():
    w = ui()
    w["nodes"][1]["mode"] = 2                                                   # negative prompt muted
    out = wi.convert(w, S)
    assert "negative" not in {s["id"]: s for s in out["slots"]}["w5"]["inputs"]
    assert any("muted, missing or unresolved" in n for n in out["notes"])

    b = {"nodes": [node(1, "LoadImage", [], ["a.png"]),
                   node(2, "ImageScale", [("image", 20, "IMAGE")], [512], mode=4, outputs=[{"type": "IMAGE"}]),
                   node(3, "VAEDecode", [("samples", 21)])],
         "links": [link(20, 1, 0, 2, 0, "IMAGE"), link(21, 2, 0, 3, 0, "IMAGE")]}
    got = {s["id"]: s for s in wi.convert(b, S)["slots"]}
    assert "w2" not in got and got["w3"]["inputs"]["samples"] == ["w1", 0]


def test_set_get_pairs_and_primitive_nodes_are_stepped_over():
    w = {"nodes": [node(1, "EmptyLatentImage", [], [64, 64, 1]),
                   node(2, "SetNode", [("LATENT", 30)], ["lat"]),
                   node(3, "GetNode", [], ["lat"]),
                   node(4, "KSampler", [("latent_image", 31), ("steps", 32)], [1, "fixed", 5, 1.0, "euler"]),
                   node(5, "PrimitiveNode", [], [30])],
         "links": [link(30, 1, 0, 2, 0), link(31, 3, 0, 4, 0), link(32, 5, 0, 4, 1)]}
    got = {s["id"]: s for s in wi.convert(w, S)["slots"]}
    assert set(got) == {"w1", "w4"}
    assert got["w4"]["inputs"]["latent_image"] == ["w1", 0] and got["w4"]["inputs"]["steps"] == 30


def test_a_subgraph_is_opened_and_its_instance_values_win():
    sub = {"id": "sg-1", "name": "Sampler", "nodes": [node(7, "KSampler", [("positive", None), ("steps", None)], [1, "fixed", 5, 1.0, "euler"])],
           "inputs": [{"name": "pos"}, {"name": "steps"}], "outputs": [{"name": "out"}],
           "links": [{"id": 1, "origin_id": wi.SG_IN, "origin_slot": 0, "target_id": 7, "target_slot": 0, "type": "CONDITIONING"},
                     {"id": 2, "origin_id": wi.SG_IN, "origin_slot": 1, "target_id": 7, "target_slot": 1, "type": "INT"},
                     {"id": 3, "origin_id": 7, "origin_slot": 0, "target_id": wi.SG_OUT, "target_slot": 0, "type": "LATENT"}]}
    sub["nodes"][0]["inputs"][1]["widget"] = {"name": "steps"}
    w = {"definitions": {"subgraphs": [sub]},
         "nodes": [node(1, "CLIPTextEncode", [("clip", None)], ["fox"]),
                   {"id": 2, "type": "sg-1", "inputs": [{"name": "pos", "link": 40}, {"name": "steps", "link": None}], "widgets_values": [33]},
                   node(3, "VAEDecode", [("samples", 41)])],
         "links": [link(40, 1, 0, 2, 0), link(41, 2, 0, 3, 0)]}
    got = {s["id"]: s for s in wi.convert(w, S)["slots"]}
    assert got["wsg2_7"]["inputs"]["positive"] == ["w1", 0] and got["wsg2_7"]["inputs"]["steps"] == 33
    assert got["w3"]["inputs"]["samples"] == ["wsg2_7", 0] and got["wsg2_7"]["group"] == "Sampler"


def test_not_installed_nodes_are_kept_and_named_and_a_non_workflow_is_refused():
    out = wi.convert({"1": {"class_type": "Mystery", "inputs": {}}}, S)
    assert out["slots"][0]["node"] == "Mystery" and "Mystery" in out["notes"][0]
    with pytest.raises(ValueError):
        wi.convert({"hello": 1}, S)
    with pytest.raises(ValueError):
        wi.convert({"nodes": [node(1, "Note")], "links": []}, S)
