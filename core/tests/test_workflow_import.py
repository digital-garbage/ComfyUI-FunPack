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
    assert out["bound"] == {"prompt": ["CLIPTextEncode #1.text"], "negative": ["CLIPTextEncode #2.text"], "seed": ["KSampler #5.seed"],
                            "width": ["EmptyLatentImage #4.width"], "height": ["EmptyLatentImage #4.height"]}
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
    assert out["bound"]["image"] == ["FunPackLoadMedia #1.media_id"] and any("start picture" in n for n in out["notes"])


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


NODES["Conditioning2"] = {"inputs": {"positive": "CONDITIONING", "negative": "CONDITIONING"}, "outputs": ["CONDITIONING", "CONDITIONING"]}
NODES["Int"] = {"inputs": {"value": "INT"}, "outputs": ["INT"]}


def test_a_bypassed_or_muted_subgraph_instance_does_not_run():
    sub = {"id": "sg-1", "name": "Upscale", "nodes": [node(7, "ImageScale", [("image", None)], [64])],
           "inputs": [{"name": "img", "type": "IMAGE"}], "outputs": [{"name": "out", "type": "IMAGE"}],
           "links": [{"id": 1, "origin_id": wi.SG_IN, "origin_slot": 0, "target_id": 7, "target_slot": 0, "type": "IMAGE"},
                     {"id": 2, "origin_id": 7, "origin_slot": 0, "target_id": wi.SG_OUT, "target_slot": 0, "type": "IMAGE"}]}
    for mode in (4, 2):
        w = {"definitions": {"subgraphs": [sub]},
             "nodes": [node(1, "LoadImage", [], ["a.png"]),
                       {"id": 2, "type": "sg-1", "mode": mode, "inputs": [{"name": "img", "type": "IMAGE", "link": 50}], "outputs": [{"type": "IMAGE"}], "widgets_values": []},
                       node(3, "VAEDecode", [("samples", 51)])],
             "links": [link(50, 1, 0, 2, 0, "IMAGE"), link(51, 2, 0, 3, 0, "IMAGE")]}
        got = {s["id"]: s for s in wi.convert(w, S)["slots"]}
        assert not any(i.startswith("wsg") for i in got), "the disabled stage is not queued"
        assert (got["w3"]["inputs"].get("samples") == ["w1", 0]) == (mode == 4)          # bypass passes the picture on, mute leaves it unwired


def test_a_primitive_seed_reaches_its_node_through_a_reroute():
    w = {"nodes": [node(1, "KSampler", [("seed", 60)], [0, "fixed", 5, 1.0, "euler"]), node(2, "Reroute", [("", 61)]), node(3, "PrimitiveNode", [], [1234])],
         "links": [link(60, 2, 0, 1, 0), link(61, 3, 0, 2, 0)]}
    assert {s["id"]: s for s in wi.convert(w, S)["slots"]}["w1"]["inputs"]["seed"] == 1234


def test_an_input_the_obvious_node_takes_from_a_wire_is_not_given_to_another_node_and_the_note_says_so():
    api = {"1": {"class_type": "Int", "inputs": {"value": 512}},
           "2": {"class_type": "EmptyLatentImage", "inputs": {"width": ["1", 0], "height": 512, "batch_size": 1}},
           "3": {"class_type": "ImageScale", "inputs": {"image": ["9", 0], "width": 100}}}
    out = wi.convert(api, S)
    assert "width" not in out["bound"] and "height" in out["bound"]
    assert any("Width" in n and "EmptyLatentImage" in n for n in out["notes"])


def test_one_control_drives_every_node_of_the_kind_and_each_sampler_gets_its_own_prompt():
    api = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hi", "clip": ["9", 0]}},
           "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "lo", "clip": ["9", 0]}},
           "3": {"class_type": "KSampler", "inputs": {"positive": ["1", 0], "seed": 1, "steps": 1}},
           "4": {"class_type": "KSampler", "inputs": {"positive": ["2", 0], "seed": 1, "steps": 1}}}
    out = wi.convert(api, S)
    assert out["bound"]["seed"] == ["KSampler #3.seed", "KSampler #4.seed"]
    assert out["bound"]["prompt"] == ["CLIPTextEncode #1.text", "CLIPTextEncode #2.text"]


def test_positive_and_negative_are_not_swapped_when_one_node_carries_both():
    api = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "neg", "clip": ["9", 0]}},
           "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "pos", "clip": ["9", 0]}},
           "3": {"class_type": "Conditioning2", "inputs": {"negative": ["1", 0], "positive": ["2", 0]}},
           "4": {"class_type": "KSampler", "inputs": {"positive": ["3", 0], "negative": ["3", 1], "seed": 1}}}
    out = wi.convert(api, S)
    assert out["bound"]["prompt"] == ["CLIPTextEncode #2.text"] and out["bound"]["negative"] == ["CLIPTextEncode #1.text"]


def test_widget_values_follow_the_sockets_that_are_widgets_not_the_schema_guess():
    NODES["Odd"] = {"inputs": {"text": "STRING", "n": "INT"}, "outputs": []}      # the schema says both are widgets...
    n = {"id": 1, "type": "Odd", "inputs": [{"name": "text", "link": None}, {"name": "n", "link": None, "widget": {"name": "n"}}], "widgets_values": [4]}
    assert wi.convert({"nodes": [n], "links": []}, S)["slots"][0]["inputs"] == {"n": 4}      # ...the export says only n is
    NODES["Plain"] = {"inputs": {"label": "STRING", "mode": "COMBO"}, "outputs": []}
    p = {"id": 1, "type": "Plain", "inputs": [], "widgets_values": ["fixed", "x"]}          # a string that merely looks like a seed control
    assert wi.convert({"nodes": [p], "links": []}, S)["slots"][0]["inputs"] == {"label": "fixed", "mode": "x"}


def test_a_set_inside_one_subgraph_copy_is_not_read_by_the_other_copy():
    w = {"nodes": [node("sg1_1", "EmptyLatentImage", [], [1, 1, 1]), node("sg1_2", "SetNode", [("L", 70)], ["lat"]), node("sg1_3", "GetNode", [], ["lat"]), node("sg1_4", "KSampler", [("latent_image", 71)], [1, "fixed", 1, 1.0, "euler"]),
                   node("sg2_1", "EmptyLatentImage", [], [2, 2, 1]), node("sg2_2", "SetNode", [("L", 72)], ["lat"]), node("sg2_3", "GetNode", [], ["lat"]), node("sg2_4", "KSampler", [("latent_image", 73)], [1, "fixed", 1, 1.0, "euler"])],
         "links": [link(70, "sg1_1", 0, "sg1_2", 0), link(71, "sg1_3", 0, "sg1_4", 0), link(72, "sg2_1", 0, "sg2_2", 0), link(73, "sg2_3", 0, "sg2_4", 0)]}
    got = {s["id"]: s for s in wi.convert(w, S)["slots"]}
    assert got["wsg2_4"]["inputs"]["latent_image"] == ["wsg2_1", 0] and got["wsg1_4"]["inputs"]["latent_image"] == ["wsg1_1", 0]


# ── what the official ComfyUI workflow templates showed (audit, 2026-10-21) ──────────────────────────────

NODES["Resize"] = {"inputs": {"image": "IMAGE", "kind": "COMBO", "kind.width": "INT", "kind.mult": "FLOAT", "method": "COMBO"}, "outputs": ["IMAGE"],
                   "shown_by": {"kind.width": ("kind", ["exact"]), "kind.mult": ("kind", ["scale"])}}
NODES["Sampler2"] = {"inputs": {"seed": "INT", "steps": "INT", "denoise": "FLOAT", "model": "MODEL"}, "outputs": ["LATENT"]}
NODES["Guider"] = {"inputs": {"model": "MODEL", "conditioning": "CONDITIONING"}, "outputs": ["GUIDER"]}
NODES["Save"] = {"inputs": {"samples": "LATENT", "quality": "INT"}, "outputs": [], "required": ["samples", "quality"], "output": True,
                 "defaults": {"quality": 90}}
NODES["Viewer"] = {"inputs": {"images": "IMAGE", "view": "IMAGECOMPARE"}, "outputs": [], "required": ["view"], "output": True, "frontend": {"view"}}


def test_a_widget_value_that_is_a_list_does_not_crash_the_reading_of_the_next_one():
    # a bounding-box editor stores a list where a number's control word would be
    n = {"id": 1, "type": "Sampler2", "inputs": [], "widgets_values": [1, [1, 2], 5]}
    assert wi.convert({"nodes": [n], "links": []}, S)["slots"][0]["inputs"]["seed"] == 1


def test_a_dynamic_combo_reads_its_chosen_widgets_right_after_it_and_leaves_the_next_widget_alone():
    for values, want in ((["exact", 1536, "lanczos"], {"kind": "exact", "kind.width": 1536, "method": "lanczos"}),
                         (["scale", 2.0, "area"], {"kind": "scale", "kind.mult": 2.0, "method": "area"})):
        n = {"id": 1, "type": "Resize", "inputs": [{"name": "image", "link": None}], "widgets_values": values}
        assert wi.convert({"nodes": [n], "links": []}, S)["slots"][0]["inputs"] == want


def test_named_values_win_over_counting_and_control_words_are_not_inputs():
    n = {"id": 1, "type": "Sampler2", "inputs": [], "widgets_values": [9, "fixed", 4, 1.0],
         "widgets_values_named": {"seed": 9, "control_after_generate": "fixed", "steps": 4, "denoise": 1.0}}
    assert wi.convert({"nodes": [n], "links": []}, S)["slots"][0]["inputs"] == {"seed": 9, "steps": 4, "denoise": 1.0}


def test_a_node_inside_a_subgraph_keeps_every_widget_though_its_sockets_list_only_some():
    # the top-level rule (only sockets marked as widgets are widgets) must not apply to widgets the definition never lists as sockets
    n = {"id": 1, "type": "Sampler2", "inputs": [{"name": "steps", "link": None, "widget": {"name": "steps"}}], "widgets_values": [7, "fixed", 4, 0.5]}
    assert wi.convert({"nodes": [n], "links": []}, S)["slots"][0]["inputs"] == {"seed": 7, "steps": 4, "denoise": 0.5}


def test_an_instance_of_a_subgraph_gets_one_value_per_typed_in_input_in_the_definitions_order():
    sub = {"id": "sg-9", "name": "Run", "nodes": [node(7, "Sampler2", [("steps", None), ("seed", None)], [0, "fixed", 1, 1.0])],
           "inputs": [{"name": "model", "type": "MODEL"}, {"name": "seed", "type": "INT"}, {"name": "steps", "type": "INT"}],
           "outputs": [{"name": "out", "type": "LATENT"}],
           "links": [{"id": 1, "origin_id": wi.SG_IN, "origin_slot": 1, "target_id": 7, "target_slot": 1, "type": "INT"},
                     {"id": 2, "origin_id": wi.SG_IN, "origin_slot": 2, "target_id": 7, "target_slot": 0, "type": "INT"},
                     {"id": 3, "origin_id": 7, "origin_slot": 0, "target_id": wi.SG_OUT, "target_slot": 0, "type": "LATENT"}]}
    sub["nodes"][0]["inputs"][0]["widget"] = {"name": "steps"}
    w = {"definitions": {"subgraphs": [sub]}, "nodes": [{"id": 2, "type": "sg-9", "inputs": [{"name": "model", "link": None}], "widgets_values": [123, 30]},
                                                          node(3, "VAEDecode", [("samples", 41)])],
         "links": [link(41, 2, 0, 3, 0)]}
    got = {s["id"]: s for s in wi.convert(w, S)["slots"]}["wsg2_7"]["inputs"]
    assert got["seed"] == 123 and got["steps"] == 30


def test_the_prompt_is_found_through_a_guiders_conditioning_input():
    w = {"nodes": [node(1, "CLIPTextEncode", [("clip", None)], ["a fox"]), node(2, "Guider", [("conditioning", 10)]),
                   node(3, "Save", [("samples", 11)], [80])],
         "links": [link(10, 1, 0, 2, 0)]}
    out = wi.convert(w, S)
    assert out["bound"]["prompt"] == ["CLIPTextEncode #1.text"]


def test_a_widget_an_older_export_never_had_gets_the_nodes_default_and_a_dangling_node_is_left_out():
    w = {"nodes": [node(1, "Sampler2", [], [1, "fixed", 4, 1.0]), node(2, "Save", [("samples", 10)], []),
                   node(3, "Resize", [("image", None)], ["exact", 64, "area"])],      # feeds nothing, takes nothing
         "links": [link(10, 1, 0, 2, 0)]}
    out = wi.convert(w, S)
    assert {s["id"] for s in out["slots"]} == {"w1", "w2"}
    assert {s["id"]: s for s in out["slots"]}["w2"]["inputs"]["quality"] == 90
    assert any("Left out" in n and "Resize #3" in n for n in out["notes"])


def test_a_viewer_only_the_comfyui_page_can_draw_is_not_an_end_and_does_not_block_the_run():
    w = {"nodes": [node(1, "Sampler2", [], [1, "fixed", 4, 1.0]), node(2, "Save", [("samples", 10)], [80]), node(3, "Viewer", [("images", 11)])],
         "links": [link(10, 1, 0, 2, 0)]}
    out = wi.convert(w, S)
    assert {s["id"] for s in out["slots"]} == {"w1", "w2"}


def test_an_end_that_lost_its_feed_stops_the_pruning_instead_of_hiding_the_whole_graph():
    w = {"nodes": [node(1, "Sampler2", [], [1, "fixed", 4, 1.0]), node(2, "Save", [], [80])], "links": []}
    assert {s["id"] for s in wi.convert(w, S)["slots"]} == {"w1", "w2"}


def test_a_bypassed_node_with_an_any_socket_passes_that_input_through():
    NODES["Peek"] = {"inputs": {"source": "*"}, "outputs": ["STRING"]}
    w = {"nodes": [node(1, "CLIPTextEncode", [("clip", None)], ["fox"]),
                   node(2, "Peek", [("source", 20, "*")], mode=4, outputs=[{"type": "STRING"}]),
                   node(3, "CLIPTextEncode", [("text", 21, "STRING")], [""])],
         "links": [link(20, 3, 0, 2, 0, "*"), link(21, 2, 0, 1, 0, "STRING")]}
    got = {s["id"]: s for s in wi.convert(w, S)["slots"]}
    assert "w2" not in got
