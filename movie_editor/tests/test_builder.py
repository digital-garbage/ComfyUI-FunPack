"""Tests for the code graph builder (fixed core + slot wiring + auto-wire)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from movie_editor.backend import builder  # noqa: E402

OI = {
    "FunPackStudio": {
        "input": {"required": {"rating": [["a", "b"], {"default": "a"}],
                               "studio_settings": ["STRING", {"default": "{}"}],
                               "adjustments": ["STRING", {"default": "[]"}]},
                  "optional": {"model": ["MODEL"], "clip": ["CLIP"], "source_image": ["IMAGE"],
                               "latent": ["LATENT"],
                               "positive_prompt": ["STRING", {"forceInput": True}],
                               "negative_prompt": ["STRING", {"forceInput": True}],
                               "refinement_key_input": ["STRING", {"forceInput": True}]}},
        "output": ["MODEL", "CONDITIONING", "CONDITIONING", "INT", "SAMPLER", "SIGMAS",
                   "SAMPLER", "SIGMAS", "IMAGE", "STRING", "STRING", "STRING", "LATENT"],
        "output_name": ["model", "modified_positive", "negative", "seed", "high_pass_sampler",
                        "high_pass_sigmas", "low_pass_sampler", "low_pass_sigmas", "loss_graph",
                        "status", "training_info", "encoded_prompts", "video_latent"]},
    "LTXVConditioning": {"input": {"required": {"positive": ["CONDITIONING"], "negative": ["CONDITIONING"],
                                                "frame_rate": ["FLOAT", {"default": 25}]}},
                         "output": ["CONDITIONING", "CONDITIONING"], "output_name": ["positive", "negative"]},
    "FunPackLTXAVSceneChainSampler": {
        "input": {"required": {"model": ["MODEL"], "vae": ["VAE"], "positive": ["CONDITIONING"],
                               "negative": ["CONDITIONING"], "sampler": ["SAMPLER"], "sigmas": ["SIGMAS"],
                               "seed": ["INT", {"default": 1}], "latent_template": ["LATENT"],
                               "num_frames_per_scene": ["INT", {"default": 97}]}},
        "output": ["LATENT", "IMAGE"], "output_name": ["latent", "images"]},
    "LTXVConcatAVLatent": {"input": {"required": {"video_latent": ["LATENT"], "audio_latent": ["LATENT"]}},
                           "output": ["LATENT"]},
    "LTXVSeparateAVLatent": {"input": {"required": {"av_latent": ["LATENT"]}},
                             "output": ["LATENT", "LATENT"], "output_name": ["video_latent", "audio_latent"]},
    "LTXVAudioVAEDecode": {"input": {"required": {"samples": ["LATENT"], "audio_vae": ["VAE"]}}, "output": ["AUDIO"]},
    "NormalizeAudioLoudness": {"input": {"required": {"audio": ["AUDIO"], "target": ["INT", {"default": -30}]}},
                               "output": ["AUDIO"]},
    "VHS_VideoCombine": {"input": {"required": {"images": ["IMAGE"], "frame_rate": ["FLOAT", {"default": 30}]},
                                   "optional": {"audio": ["AUDIO"]}}, "output": ["VHS_FILENAMES"]},
    "FunPackSaveRefinementLatent": {"input": {"required": {"latent": ["LATENT"], "refinement_key": ["STRING"]}},
                                    "output": ["LATENT", "STRING"]},
    "FunPackRefinementKeyLoader": {"input": {"required": {"key": ["STRING", {"default": "k"}]}},
                                   "output": ["STRING", "STRING"], "output_name": ["refinement_key", "status"]},
    "PrimitiveStringMultiline": {"input": {"required": {"value": ["STRING", {"default": ""}]}}, "output": ["STRING"]},
    "PrimitiveInt": {"input": {"required": {"value": ["INT", {"default": 0}]}}, "output": ["INT"]},
    "PrimitiveFloat": {"input": {"required": {"value": ["FLOAT", {"default": 0.0}]}}, "output": ["FLOAT"]},
    "LTXFloatToInt": {"input": {"required": {"a": ["FLOAT", {"default": 0}]}}, "output": ["INT"]},
    # slot nodes
    "UnetLoader": {"input": {"required": {"unet_name": [["m.safetensors"]]}}, "output": ["MODEL"]},
    "ClipLoader": {"input": {"required": {"clip_name": [["c.safetensors"]]}}, "output": ["CLIP"]},
    "VaeLoader": {"input": {"required": {"vae_name": [["v.safetensors"]]}}, "output": ["VAE"]},
    "LoadImage": {"input": {"required": {"image": [["a.png"]]}}, "output": ["IMAGE", "MASK"]},
    "ImgProc": {"input": {"required": {"length": ["INT", {"default": 97}], "vae": ["VAE"], "image": ["IMAGE"]}},
                "output": ["LATENT", "LATENT", "IMAGE"], "output_name": ["latent", "Latent", "output_image"]},
    "UpscaleModelLoader": {"input": {"required": {"model_name": [["x4.pth"]]}}, "output": ["UPSCALE_MODEL"],
                           "output_name": ["UPSCALE_MODEL"]},
    "ImageUpscaleWithModel": {"input": {"required": {"upscale_model": ["UPSCALE_MODEL"], "image": ["IMAGE"]}},
                              "output": ["IMAGE"], "output_name": ["IMAGE"]},
    "LoraLoader": {"input": {"required": {"model": ["MODEL"], "clip": ["CLIP"],
                                          "lora_name": [["x.safetensors"]],
                                          "strength_model": ["FLOAT", {"default": 1.0}],
                                          "strength_clip": ["FLOAT", {"default": 1.0}]}},
                  "output": ["MODEL", "CLIP"], "output_name": ["MODEL", "CLIP"]},
}

PARAMS = {"prompt": "scene one", "seed": 42, "num_frames_per_scene": 121, "frame_rate": 24}


def test_core_skeleton_links_and_params():
    graph, report = builder.build(OI, {"slots": []}, PARAMS)
    # every fixed-core node emitted with correct class
    for cid, cls in builder.CORE.items():
        assert graph[cid]["class_type"] == cls
    # internal links
    assert graph["sampler"]["inputs"]["model"] == ["studio", 0]
    assert graph["cond"]["inputs"]["positive"] == ["studio", 1]
    assert graph["cond"]["inputs"]["negative"] == ["studio", 2]
    assert graph["concat"]["inputs"]["video_latent"] == ["studio", 12]
    assert graph["sampler"]["inputs"]["latent_template"] == ["concat", 0]
    assert graph["separate"]["inputs"]["av_latent"] == ["sampler", 0]
    assert graph["audiodec"]["inputs"]["samples"] == ["separate", 1]
    assert graph["vhs"]["inputs"]["images"] == ["sampler", 1]
    assert graph["vhs"]["inputs"]["audio"] == ["normaudio", 0]
    # param overrides
    assert graph["pos"]["inputs"]["value"] == "scene one"
    assert graph["frames"]["inputs"]["value"] == 121
    assert graph["fps"]["inputs"]["value"] == 24
    assert graph["sampler"]["inputs"]["seed"] == 42
    # with no slots, the open ports are unsatisfied
    assert any("FunPackStudio" in u and ".model" in u for u in report["unsatisfied"])


def test_explicit_wires_and_autowire():
    models = {"full_control": True, "slots": [
        {"id": "u", "node_class": "UnetLoader", "inputs": {"unet_name": "m.safetensors"},
         "wires": {"MODEL": "port:FunPackStudio.model"}},
        {"id": "c", "node_class": "ClipLoader", "inputs": {}, "wires": {"CLIP": "port:FunPackStudio.clip"}},
        {"id": "v", "node_class": "VaeLoader", "inputs": {}, "wires": {}},
        {"id": "li", "node_class": "LoadImage", "inputs": {}, "wires": {"IMAGE": "node:ip:image"}},
        {"id": "ip", "node_class": "ImgProc", "inputs": {"length": 121},
         "wires": {"latent": "port:FunPackStudio.latent",
                   "Latent": "port:LTXVConcatAVLatent.audio_latent",
                   "output_image": "port:FunPackStudio.source_image"}},
    ]}
    graph, report = builder.build(OI, models, PARAMS)
    # explicit slot->core wires
    assert graph["studio"]["inputs"]["model"] == ["slot_u", 0]
    assert graph["studio"]["inputs"]["clip"] == ["slot_c", 0]
    assert graph["studio"]["inputs"]["latent"] == ["slot_ip", 0]
    assert graph["studio"]["inputs"]["source_image"] == ["slot_ip", 2]
    assert graph["concat"]["inputs"]["audio_latent"] == ["slot_ip", 1]
    # explicit slot->slot wire (LoadImage -> ImgProc.image)
    assert graph["slot_ip"]["inputs"]["image"] == ["slot_li", 0]
    # auto-wire by unique type: single VAE feeds both VAE consumers
    assert graph["sampler"]["inputs"]["vae"] == ["slot_v", 0]
    assert graph["audiodec"]["inputs"]["audio_vae"] == ["slot_v", 0]
    # slot widget value carried through
    assert graph["slot_ip"]["inputs"]["length"] == 121
    # a complete config has nothing unsatisfied / ambiguous
    assert report["unsatisfied"] == []
    assert report["ambiguous"] == []


def test_stale_combo_value_coerced_to_live_choice():
    # "old.safetensors" was renamed/removed; the live spec only has "m.safetensors".
    models = {"full_control": True, "slots": [
        {"id": "u", "node_class": "UnetLoader", "inputs": {"unet_name": "old.safetensors"},
         "wires": {"MODEL": "port:FunPackStudio.model"}},
    ]}
    graph, _ = builder.build(OI, models, PARAMS)
    assert graph["slot_u"]["inputs"]["unet_name"] == "m.safetensors"


def test_linked_inputs_drive_multiple_node_values():
    models = {
        "slots": [
            {"id": "a", "node_class": "ImgProc", "inputs": {"length": 50}, "wires": {}},
            {"id": "b", "node_class": "ImgProc", "inputs": {"length": 50}, "wires": {}},
        ],
        "links": [
            {"id": "L1", "name": "size", "kind": "int", "value": 121,
             "members": [{"slotId": "a", "input": "length"}, {"slotId": "b", "input": "length"}]},
        ],
    }
    graph, _ = builder.build(OI, models, PARAMS)
    assert graph["slot_a"]["inputs"]["length"] == 121
    assert graph["slot_b"]["inputs"]["length"] == 121


def test_editor_driven_link_pulls_from_params():
    models = {
        "slots": [{"id": "a", "node_class": "ImgProc", "inputs": {"length": 50}, "wires": {}}],
        "links": [{"id": "L", "name": "len", "source": "editor", "editor_key": "num_frames_per_scene",
                   "members": [{"slotId": "a", "input": "length"}]}],
    }
    graph, _ = builder.build(OI, models, PARAMS)   # PARAMS num_frames_per_scene = 121
    assert graph["slot_a"]["inputs"]["length"] == 121


def test_media_injects_loadimage_wired_to_target():
    models = {"slots": [{"id": "ip", "node_class": "ImgProc", "inputs": {}, "wires": {}}]}
    media = {"filename": "funpack_movie_x.png", "target": "node:ip:image"}
    graph, report = builder.build(OI, models, PARAMS, media=media)
    assert graph["media_load"]["class_type"] == "LoadImage"
    assert graph["media_load"]["inputs"]["image"] == "funpack_movie_x.png"
    assert graph["slot_ip"]["inputs"]["image"] == ["media_load", 0]


def test_optional_open_ports_do_not_block():
    # loaders + vae + audio handled, no image source: source_image/latent optional -> ok
    models = {"full_control": True, "slots": [
        {"id": "u", "node_class": "UnetLoader", "inputs": {}, "wires": {"MODEL": "port:FunPackStudio.model"}},
        {"id": "c", "node_class": "ClipLoader", "inputs": {}, "wires": {"CLIP": "port:FunPackStudio.clip"}},
        {"id": "v", "node_class": "VaeLoader", "inputs": {}, "wires": {}},
        {"id": "ip", "node_class": "ImgProc", "inputs": {},
         "wires": {"latent": "port:FunPackStudio.latent", "Latent": "port:LTXVConcatAVLatent.audio_latent"}},
        {"id": "li", "node_class": "LoadImage", "inputs": {}, "wires": {"IMAGE": "node:ip:image"}},
    ]}
    graph, report = builder.build(OI, models, PARAMS)
    assert report["blocking"] == []   # source_image/latent optional, everything required satisfied


def test_ambiguous_type_is_reported_not_guessed():
    models = {"slots": [
        {"id": "v1", "node_class": "VaeLoader", "inputs": {}, "wires": {}},
        {"id": "v2", "node_class": "VaeLoader", "inputs": {}, "wires": {}},
    ]}
    graph, report = builder.build(OI, models, PARAMS)
    assert not isinstance(graph["sampler"]["inputs"].get("vae"), list)  # left unwired
    assert any("FunPackLTXAVSceneChainSampler" in a and ".vae" in a for a in report["ambiguous"])


def _loaders():
    return [
        {"id": "u", "node_class": "UnetLoader", "inputs": {}, "wires": {"MODEL": "port:FunPackStudio.model"}},
        {"id": "c", "node_class": "ClipLoader", "inputs": {}, "wires": {"CLIP": "port:FunPackStudio.clip"}},
        {"id": "v", "node_class": "VaeLoader", "inputs": {}, "wires": {}},
    ]


def test_upscale_before_export_wires_cleanly_and_loader_autofeeds():
    # The user's scenario: tap decoded frames -> ImageUpscaleWithModel -> replace export frames,
    # with an UpscaleModelLoader that has no explicit output wire (it should still auto-feed).
    models = {"full_control": True, "slots": _loaders() + [
        {"id": "um", "node_class": "UpscaleModelLoader", "inputs": {}, "wires": {}},
        {"id": "up", "node_class": "ImageUpscaleWithModel", "inputs": {},
         "input_sources": {"image": "core:sampler:1"},          # decoded video frames
         "wires": {"IMAGE": "port:VHS_VideoCombine.images"}},   # replace the export frames
    ]}
    graph, report = builder.build(OI, models, PARAMS)
    assert graph["slot_up"]["inputs"]["image"] == ["sampler", 1]    # decoded frames in
    assert graph["vhs"]["inputs"]["images"] == ["slot_up", 0]       # upscaled frames out to export
    assert graph["slot_up"]["inputs"]["upscale_model"] == ["slot_um", 0]  # loader auto-feeds
    assert builder._find_cycle(graph) is None
    assert not any("dependency cycle" in m.lower() for m in report["blocking"])  # no cycle block


def test_cycle_from_default_source_image_wire_is_dropped():
    # An image_processing node consumes the decoded frames AND carries the role-default wire back
    # to Studio · source_image — that loops. The default wire must be dropped, the user's source kept.
    models = {"full_control": True, "slots": _loaders() + [
        {"id": "ip", "node_class": "ImgProc", "role": "image_processing", "inputs": {},
         "input_sources": {"image": "core:sampler:1", "vae": "out:v:VAE"},
         "wires": {"output_image": "port:FunPackStudio.source_image"}},  # == role default → droppable
    ]}
    graph, report = builder.build(OI, models, PARAMS)
    assert builder._find_cycle(graph) is None                       # no cycle handed to ComfyUI
    assert graph["slot_ip"]["inputs"]["image"] == ["sampler", 1]    # user's source preserved
    src_img = graph["studio"]["inputs"].get("source_image")
    assert not (isinstance(src_img, list) and src_img[0] == "slot_ip")  # looping default dropped
    assert any("dependency cycle" in m.lower() for m in report["auto_wired"])


def test_all_explicit_cycle_is_reported_not_silently_built():
    # Two nodes feeding each other through explicit input sources — nothing droppable, so it must
    # surface as a clear blocking error rather than a graph ComfyUI rejects opaquely.
    models = {"full_control": True, "slots": _loaders() + [
        {"id": "a", "node_class": "ImgProc", "inputs": {},
         "input_sources": {"image": "out:b:output_image", "vae": "out:v:VAE"}, "wires": {}},
        {"id": "b", "node_class": "ImgProc", "inputs": {},
         "input_sources": {"image": "out:a:output_image", "vae": "out:v:VAE"}, "wires": {}},
    ]}
    graph, report = builder.build(OI, models, PARAMS)
    assert any("dependency cycle" in m.lower() for m in report["blocking"])


def test_extract_widgets_handles_control_and_links():
    nd = OI["FunPackLTXAVSceneChainSampler"]
    # seed(+control), num_frames, then the rest are not declared widgets here, so map stops
    vals = [999, "randomize", 97]
    got = builder.extract_widgets(nd, vals)
    assert got["seed"] == 999            # control_after_generate ("randomize") skipped
    assert got["num_frames_per_scene"] == 97

    # forceInput STRINGs are sockets, never widgets
    studio = builder.extract_widgets(OI["FunPackStudio"], ["a", "{\"x\":1}", "[]", "ignored"])
    assert studio == {"rating": "a", "studio_settings": "{\"x\":1}", "adjustments": "[]"}

    # dict widget store (VHS) drops non-input keys
    d = builder.extract_widgets(OI["VHS_VideoCombine"], {"frame_rate": 30, "videopreview": {"x": 1}})
    assert d == {"frame_rate": 30}


def test_movie_editor_scene_ratings_in_studio_settings():
    import json
    params = {
        **PARAMS,
        "studio_inputs": {
            "rating": "__funpack_continue__",
            "_movie_editor_scene_ratings": [
                {"index": 0, "rating": "Perfect"},
                {"index": 2, "rating": "Missing action"},
            ],
        },
    }
    graph, _report = builder.build(OI, {"slots": []}, params)
    settings = json.loads(graph["studio"]["inputs"]["studio_settings"])
    assert settings["refiner"]["split_by_transitions"] is True
    assert settings["refiner"]["movie_editor_scene_ratings"] == [
        {"index": 0, "rating": "Perfect"},
        {"index": 2, "rating": "Missing action"},
    ]
    assert "_movie_editor_scene_ratings" not in graph["studio"]["inputs"]


def test_refinement_key_injected_into_keyloader():
    graph, _ = builder.build(OI, {"slots": []}, {**PARAMS, "refinement_key": "charA"})
    assert graph["keyloader"]["inputs"]["key_name"] == "charA"
    # combo forced to none so the typed key_name wins over any reference-seeded value
    assert graph["keyloader"]["inputs"]["refinement_key"] == "-None-"


def test_refinement_key_defaults_when_absent():
    graph, _ = builder.build(OI, {"slots": []}, PARAMS)
    assert graph["keyloader"]["inputs"]["key_name"] == "default"


def test_bypassed_lora_drops_node_and_passes_model_clip_through():
    models = {"full_control": True, "slots": [
        {"id": "u", "node_class": "UnetLoader", "inputs": {"unet_name": "m.safetensors"}, "wires": {}},
        {"id": "c", "node_class": "ClipLoader", "inputs": {}, "wires": {}},
        {"id": "lora", "node_class": "LoraLoader", "bypassed": True,
         "inputs": {"lora_name": "x.safetensors", "strength_model": 0.8, "strength_clip": 0.8,
                    "model": ["slot_u", 0], "clip": ["slot_c", 0]},
         "wires": {"MODEL": "port:FunPackStudio.model", "CLIP": "port:FunPackStudio.clip"}},
    ]}
    graph, report = builder.build(OI, models, PARAMS)
    # the LoRA node itself is gone
    assert "slot_lora" not in graph
    # Studio's model/clip wire straight to the original loaders, skipping the LoRA
    assert graph["studio"]["inputs"]["model"] == ["slot_u", 0]
    assert graph["studio"]["inputs"]["clip"] == ["slot_c", 0]
    assert any("LoraLoader bypassed" in w for w in report["wired"])


def test_non_bypassed_lora_stays_in_graph():
    models = {"full_control": True, "slots": [
        {"id": "u", "node_class": "UnetLoader", "inputs": {"unet_name": "m.safetensors"}, "wires": {}},
        {"id": "c", "node_class": "ClipLoader", "inputs": {}, "wires": {}},
        {"id": "lora", "node_class": "LoraLoader",
         "inputs": {"lora_name": "x.safetensors", "model": ["slot_u", 0], "clip": ["slot_c", 0]},
         "wires": {"MODEL": "port:FunPackStudio.model", "CLIP": "port:FunPackStudio.clip"}},
    ]}
    graph, _ = builder.build(OI, models, PARAMS)
    assert "slot_lora" in graph
    assert graph["studio"]["inputs"]["model"] == ["slot_lora", 0]
    assert graph["studio"]["inputs"]["clip"] == ["slot_lora", 1]


def test_bypass_on_node_with_no_matching_input_leaves_it_active_and_reports():
    # VaeLoader has no connection_input at all, so there's nothing to pass an output through —
    # bypass must be a documented no-op here, not a crash or a silently dropped node.
    models = {"full_control": True, "slots": [
        {"id": "v", "node_class": "VaeLoader", "bypassed": True, "inputs": {}, "wires": {}},
    ]}
    graph, report = builder.build(OI, models, PARAMS)
    assert "slot_v" in graph
    assert any("bypass needs exactly one input" in u for u in report["unsatisfied"])
