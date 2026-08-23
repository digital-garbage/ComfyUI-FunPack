"""Tests for the code graph builder (fixed core + slot wiring + auto-wire)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from movie_editor.backend import builder  # noqa: E402

OI = {
    "FunPackStudio": {
        "input": {"required": {"rating": [["a", "b"], {"default": "a"}],
                               "studio_settings": ["STRING", {"default": "{}"}],
                               "adjustments": ["STRING", {"default": "[]"}]},
                  "optional": {"model": ["MODEL"], "clip": ["CLIP"], "source_image": ["IMAGE"],
                               "latent": ["LATENT"], "positive_conditioning": ["CONDITIONING"],
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
    "CondLoader": {"input": {"required": {"cond_name": [["c.json"]]}}, "output": ["CONDITIONING"]},
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
    # Real shape of LTXICLoRALoaderModelOnly: a MODEL passthrough plus a FLOAT the graph
    # never reads. strength_model is a widget, so no connection input matches that FLOAT.
    "LTXICLoRALoaderModelOnly": {
        "input": {"required": {"model": ["MODEL"],
                               "lora_name": [["x.safetensors"]],
                               "strength_model": ["FLOAT", {"default": 1.0}]}},
        "output": ["MODEL", "FLOAT"], "output_name": ["model", "latent_downscale_factor"]},
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


def test_run_output_gets_unique_temp_prefix():
    # Each build (= one generation run) must give the VHS preview node its OWN filename_prefix
    # so a later run can't reuse/overwrite an earlier run's temp file — otherwise a previously
    # rendered scene stops playing once the next scene generates. save_output stays False (temp).
    g1, _ = builder.build(OI, {"slots": []}, PARAMS)
    g2, _ = builder.build(OI, {"slots": []}, PARAMS)
    p1 = g1["vhs"]["inputs"]["filename_prefix"]
    p2 = g2["vhs"]["inputs"]["filename_prefix"]
    assert p1.startswith("funpack_preview_")
    assert p2.startswith("funpack_preview_")
    assert p1 != p2  # distinct per run
    assert g1["vhs"]["inputs"]["save_output"] is False


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


def test_wired_positive_conditioning_replaces_the_clip_requirement():
    # Studio can run off a pre-encoded CONDITIONING. With two text encoders installed, CLIP
    # is ambiguous — that must not block generation once positive_conditioning is wired, and
    # the ambiguous CLIP must be left alone rather than auto-wired over the user's own
    # conditioning (Studio prefers CLIP whenever it's connected).
    models = {"full_control": True, "slots": [
        {"id": "u", "node_class": "UnetLoader", "inputs": {}, "wires": {"MODEL": "port:FunPackStudio.model"}},
        {"id": "c1", "node_class": "ClipLoader", "inputs": {}, "wires": {}},
        {"id": "c2", "node_class": "ClipLoader", "inputs": {}, "wires": {}},
        {"id": "v", "node_class": "VaeLoader", "inputs": {}, "wires": {}},
        {"id": "li", "node_class": "LoadImage", "inputs": {}, "wires": {"IMAGE": "node:ip:image"}},
        {"id": "ip", "node_class": "ImgProc", "inputs": {},
         "wires": {"Latent": "port:LTXVConcatAVLatent.audio_latent"}},
        {"id": "cd", "node_class": "CondLoader", "inputs": {},
         "wires": {"CONDITIONING": "port:FunPackStudio.positive_conditioning"}},
    ]}
    graph, report = builder.build(OI, models, PARAMS)
    assert graph["studio"]["inputs"]["positive_conditioning"] == ["slot_cd", 0]
    assert "clip" not in graph["studio"]["inputs"]
    assert report["blocking"] == []
    assert not any(".clip" in m for m in report["ambiguous"])


def test_clip_still_blocks_when_nothing_else_feeds_studio():
    # Without the conditioning wire, an ambiguous CLIP is still a hard stop.
    models = {"full_control": True, "slots": [
        {"id": "u", "node_class": "UnetLoader", "inputs": {}, "wires": {"MODEL": "port:FunPackStudio.model"}},
        {"id": "c1", "node_class": "ClipLoader", "inputs": {}, "wires": {}},
        {"id": "c2", "node_class": "ClipLoader", "inputs": {}, "wires": {}},
        {"id": "v", "node_class": "VaeLoader", "inputs": {}, "wires": {}},
        {"id": "li", "node_class": "LoadImage", "inputs": {}, "wires": {"IMAGE": "node:ip:image"}},
        {"id": "ip", "node_class": "ImgProc", "inputs": {},
         "wires": {"Latent": "port:LTXVConcatAVLatent.audio_latent"}},
    ]}
    _, report = builder.build(OI, models, PARAMS)
    assert any(".clip" in m for m in report["blocking"])


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


def test_multitype_widget_input_does_not_block_and_keeps_its_value():
    # LTXVEmptyLatentAudio.frame_rate arrives as a V3 MultiType widget ("FLOAT,INT" with a
    # widgetType hint). It is a field, not a socket: it must not demand a source (nothing
    # outputs "FLOAT,INT", so the run was blocked) and its value must reach the graph.
    oi = dict(OI)
    oi["LTXVEmptyLatentAudio"] = {
        "input": {"required": {
            "frames_number": ["INT", {"default": 97}],
            "frame_rate": ["FLOAT,INT", {"widgetType": "INT", "default": 25}],
            "audio_vae": ["VAE"],
        }},
        "output": ["LATENT"], "output_name": ["Latent"]}
    models = {"full_control": True, "slots": [
        {"id": "u", "node_class": "UnetLoader", "inputs": {}, "wires": {"MODEL": "port:FunPackStudio.model"}},
        {"id": "c", "node_class": "ClipLoader", "inputs": {}, "wires": {"CLIP": "port:FunPackStudio.clip"}},
        {"id": "v", "node_class": "VaeLoader", "inputs": {},
         "wires": {"VAE": "port:FunPackLTXAVSceneChainSampler.vae"}},
        {"id": "av", "node_class": "VaeLoader", "inputs": {},
         "wires": {"VAE": ["node:ae:audio_vae", "port:LTXVAudioVAEDecode.audio_vae"]}},
        {"id": "ae", "node_class": "LTXVEmptyLatentAudio", "inputs": {"frame_rate": 30},
         "wires": {"Latent": "port:LTXVConcatAVLatent.audio_latent"}},
    ]}
    graph, report = builder.build(oi, models, PARAMS)
    assert report["blocking"] == []
    assert not any("frame_rate" in m for m in report["unsatisfied"] + report["ambiguous"])
    assert graph["slot_ae"]["inputs"]["frame_rate"] == 30       # user value, not a link
    assert graph["slot_ae"]["inputs"]["frames_number"] == 97     # default still emitted
    assert graph["slot_ae"]["inputs"]["audio_vae"] == ["slot_av", 0]


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


def test_bypass_ignores_outputs_nothing_is_wired_to():
    """LTXICLoRALoaderModelOnly emits a FLOAT (latent_downscale_factor) next to its MODEL,
    and nothing in the editor's graph reads it. Demanding a matching input for an output
    that feeds nothing refused a bypass that is unambiguous for every link that exists —
    only CONSUMED outputs need a passthrough."""
    models = {"full_control": True, "slots": [
        {"id": "u", "node_class": "UnetLoader", "inputs": {"unet_name": "m.safetensors"}, "wires": {}},
        {"id": "ic", "node_class": "LTXICLoRALoaderModelOnly", "bypassed": True,
         "inputs": {"lora_name": "x.safetensors", "strength_model": 0.8, "model": ["slot_u", 0]},
         "wires": {"MODEL": "port:FunPackStudio.model"}},
    ]}
    graph, report = builder.build(OI, models, PARAMS)
    assert "slot_ic" not in graph
    assert graph["studio"]["inputs"]["model"] == ["slot_u", 0]
    assert not any("bypass" in b for b in report["blocking"])
    assert any("LTXICLoRALoaderModelOnly bypassed" in w for w in report["wired"])


def test_bypass_on_node_with_no_matching_input_blocks_generation():
    # VaeLoader has no connection_input at all, so there's nothing to pass an output through.
    # A bypass a user explicitly asked for must never be silently ignored — generation blocks
    # with a clear reason instead of quietly leaving the node active.
    models = {"full_control": True, "slots": [
        {"id": "v", "node_class": "VaeLoader", "bypassed": True, "inputs": {}, "wires": {}},
    ]}
    graph, report = builder.build(OI, models, PARAMS)
    assert "slot_v" in graph
    assert any("bypass needs exactly one input" in u for u in report["unsatisfied"])
    assert any("bypass needs exactly one input" in b for b in report["blocking"])


# ── MiniMax H3 family ─────────────────────────────────────────────────────────
# H3 keeps the core's SHAPE but three of its nodes do not apply. The graph has to be
# built for the right family up front: a wrong node class here fails deep inside
# ComfyUI at generation time, where the Models panel can no longer explain it.

H3_OI = dict(OI)
H3_OI["VAEDecodeAudio"] = {"input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}},
                           "output": ["AUDIO"]}
H3_OI["EmptyMiniMaxH3LatentAV"] = {
    "input": {"required": {"width": ["INT", {"default": 1344}], "height": ["INT", {"default": 768}],
                           "length": ["INT", {"default": 124}]}},
    "output": ["LATENT"], "output_name": ["LATENT"]}

H3_MODELS = {
    "model_family": "minimax_h3",
    "slots": [
        {"id": "u", "role": "unet", "node_class": "UnetLoader", "label": "unet"},
        {"id": "c", "role": "clip", "node_class": "ClipLoader", "label": "clip"},
        {"id": "vv", "role": "video_vae", "node_class": "VaeLoader", "label": "video vae",
         "wires": {"VAE": "port:FunPackLTXAVSceneChainSampler.vae"}},
        {"id": "av", "role": "audio_vae", "node_class": "VaeLoader", "label": "audio vae",
         "wires": {"VAE": "port:VAEDecodeAudio.vae"}},
        {"id": "lat", "role": "empty_latent", "node_class": "EmptyMiniMaxH3LatentAV", "label": "av latent",
         "wires": {"LATENT": "port:FunPackLTXAVSceneChainSampler.latent_template"}},
    ],
}


def _classes(graph):
    return {n["class_type"] for n in graph.values()}


def test_h3_family_drops_the_ltx_only_core_nodes():
    graph, report = builder.build(H3_OI, H3_MODELS, {"prompt": "a shot"})
    classes = _classes(graph)
    # LTXVConditioning stamps LTX's frame_rate onto conditioning — H3 has no such field
    assert "LTXVConditioning" not in classes
    # H3's own empty-latent node emits both streams, so there is nothing to concat
    assert "LTXVConcatAVLatent" not in classes
    # LTXVAudioVAEDecode reads an output_sample_rate H3's audio VAE does not define
    assert "LTXVAudioVAEDecode" not in classes
    assert "VAEDecodeAudio" in classes
    # LTXFloatToInt only converted the project's frame rate for user slots. H3's rate is
    # fixed at 24 by the model, so keeping it made an LTX pack a hard requirement of a
    # pipeline that has no use for it.
    assert "LTXFloatToInt" not in classes
    # the shared half is untouched
    assert {"FunPackStudio", "FunPackLTXAVSceneChainSampler", "LTXVSeparateAVLatent"} <= classes
    assert not report["blocking"], report["blocking"]


def test_h3_does_not_offer_the_dropped_frame_rate_converter_as_a_producer():
    """A dropped core node must not be auto-wired into a slot that wants an INT."""
    models = dict(H3_MODELS)
    models["slots"] = H3_MODELS["slots"] + [
        {"id": "ip", "role": "image_processing", "node_class": "IntEater", "label": "eats an int"},
    ]
    oi = dict(H3_OI)
    oi["IntEater"] = {"input": {"required": {"n": ["INT", {"default": 0}]}}, "output": ["IMAGE"]}
    graph, _report = builder.build(oi, models, {"prompt": "a shot"})
    for node in graph.values():
        for value in node["inputs"].values():
            assert not (isinstance(value, list) and value and value[0] == "f2i"), value


def test_h3_wires_studio_conditioning_straight_to_the_sampler():
    graph, _ = builder.build(H3_OI, H3_MODELS, {"prompt": "a shot"})
    sampler = graph["sampler"]["inputs"]
    assert sampler["positive"] == ["studio", 1]
    assert sampler["negative"] == ["studio", 2]
    # audio decode source is asserted in its own test (it reads the sampler latent on H3)


def test_h3_latent_slot_feeds_the_sampler_directly():
    graph, report = builder.build(H3_OI, H3_MODELS, {"prompt": "a shot"})
    assert graph["sampler"]["inputs"]["latent_template"] == ["slot_lat", 0]
    assert graph["audiodec"]["inputs"]["vae"] == ["slot_av", 0]
    assert not report["blocking"], report["blocking"]


def test_h3_without_an_av_latent_slot_blocks_instead_of_generating_a_broken_graph():
    models = {"model_family": "minimax_h3",
              "slots": [s for s in H3_MODELS["slots"] if s["id"] != "lat"]}
    _graph, report = builder.build(H3_OI, models, {"prompt": "a shot"})
    assert any("latent_template" in m for m in report["blocking"]), report


# A core override is saved per core ID, not per node class, so switching family leaves the
# replaced node's input names behind on its successor. The panel lists the NEW node's inputs,
# so the leftover is invisible; emitting it hands ComfyUI a kwarg the node never declared and
# the run dies inside it ("VAEDecodeAudio.execute() got an unexpected keyword argument
# 'audio_vae'"). Guided mode filtered these out as a side effect; full control did not.

def test_a_core_override_from_another_family_is_not_emitted():
    models = dict(H3_MODELS)
    models["full_control"] = True            # guided mode filtered this one out by accident
    models["core_overrides"] = {"audiodec": {"audio_vae": "out:av:VAE"}}
    graph, report = builder.build(H3_OI, models, {"prompt": "a shot"})

    assert "audio_vae" not in graph["audiodec"]["inputs"]
    assert graph["audiodec"]["inputs"]["vae"] == ["slot_av", 0]   # the real wire is untouched
    assert any("has no input 'audio_vae'" in u for u in report["unsatisfied"]), report
    assert not report["blocking"], report["blocking"]   # a leftover is not a reason to refuse


def test_a_valid_core_override_still_applies_under_full_control():
    models = dict(H3_MODELS)
    models["full_control"] = True
    models["core_overrides"] = {"audiodec": {"vae": "out:vv:VAE"}}
    graph, report = builder.build(H3_OI, models, {"prompt": "a shot"})

    assert graph["audiodec"]["inputs"]["vae"] == ["slot_vv", 0]
    assert not report["blocking"], report["blocking"]


def test_the_ltxav_graph_is_unchanged_by_the_family_split():
    """The default family must emit exactly what it emitted before H3 existed."""
    ltx_models = {"slots": [
        {"id": "u", "role": "unet", "node_class": "UnetLoader",
         "wires": {"MODEL": "port:FunPackStudio.model"}},
        {"id": "c", "role": "clip", "node_class": "ClipLoader",
         "wires": {"CLIP": "port:FunPackStudio.clip"}},
        {"id": "vv", "role": "video_vae", "node_class": "VaeLoader",
         "wires": {"VAE": "port:FunPackLTXAVSceneChainSampler.vae"}},
        {"id": "av", "role": "audio_vae", "node_class": "VaeLoader",
         "wires": {"VAE": "port:LTXVAudioVAEDecode.audio_vae"}},
        {"id": "al", "role": "audio_encoder", "node_class": "AudioEnc",
         "wires": {"LATENT": "port:LTXVConcatAVLatent.audio_latent"}},
    ]}
    oi = dict(OI)
    oi["AudioEnc"] = {"input": {"required": {"seconds": ["FLOAT", {"default": 5.0}]}},
                      "output": ["LATENT"], "output_name": ["LATENT"]}
    graph, report = builder.build(oi, ltx_models, {"prompt": "a shot"})
    classes = _classes(graph)
    assert {"LTXVConditioning", "LTXVConcatAVLatent", "LTXVAudioVAEDecode"} <= classes
    assert "VAEDecodeAudio" not in classes
    assert graph["sampler"]["inputs"]["positive"] == ["cond", 0]
    assert graph["sampler"]["inputs"]["latent_template"] == ["concat", 0]
    assert not report["blocking"], report["blocking"]


# ── frame geometry is a property of the model, not a setting ──────────────────
# LTX: 8k+1 frames at the project's fps. H3: 17k+5 frames at a fixed 24 fps. An off-grid
# length on H3 is not a rounding nuisance — the latent node snaps its own length up while
# the sampler is told the raw number, and the run dies on the mismatch.

def test_h3_frame_counts_snap_to_the_17k_plus_5_grid():
    graph, _ = builder.build(H3_OI, H3_MODELS, {"prompt": "a shot", "num_frames_per_scene": 121})
    assert graph["frames"]["inputs"]["value"] == 124
    # a value already on the grid is left exactly as it is
    graph, _ = builder.build(H3_OI, H3_MODELS, {"prompt": "a shot", "num_frames_per_scene": 124})
    assert graph["frames"]["inputs"]["value"] == 124


def test_h3_renders_at_the_models_own_frame_rate():
    graph, _ = builder.build(H3_OI, H3_MODELS, {"prompt": "a shot", "frame_rate": 30})
    assert graph["fps"]["inputs"]["value"] == 24


def test_ltx_frame_geometry_is_untouched():
    graph, _ = builder.build(OI, {"slots": []},
                             {"prompt": "a shot", "num_frames_per_scene": 121, "frame_rate": 30})
    assert graph["frames"]["inputs"]["value"] == 121   # already 8k+1
    assert graph["fps"]["inputs"]["value"] == 30       # LTX has no fixed rate
    graph, _ = builder.build(OI, {"slots": []}, {"prompt": "a shot", "num_frames_per_scene": 100})
    assert graph["frames"]["inputs"]["value"] == 105   # snapped up to 8k+1


def test_a_latent_node_driven_by_project_frames_gets_the_same_snapped_number():
    """A slot widget bound to "Project · Frames" must not receive the raw value while the
    sampler receives the snapped one — that disagreement is the mismatch itself."""
    models = dict(H3_MODELS)
    models["links"] = [{"id": "l1", "source": "editor", "editor_key": "num_frames_per_scene",
                        "members": [{"slotId": "lat", "input": "length"}]}]
    graph, _ = builder.build(H3_OI, models, {"prompt": "a shot", "num_frames_per_scene": 121})
    assert graph["slot_lat"]["inputs"]["length"] == 124
    assert graph["frames"]["inputs"]["value"] == 124


def test_an_unknown_family_falls_back_to_ltxav_rather_than_emitting_nothing():
    assert builder.family_of({"model_family": "hailuo-9000"}) == "ltxav"
    assert builder.family_of({}) == "ltxav"
    assert builder.family_of(None) == "ltxav"
    assert builder.family_of({"model_family": "MiniMax_H3"}) == "minimax_h3"


# ── autogrow list inputs (MiniMax H3 reference nodes) ─────────────────────────

REF_OI = dict(OI)
REF_OI["MiniMaxH3ReferenceToVideo"] = {
    "input": {"required": {
        "clip": ["CLIP"],
        "prompt": ["STRING", {"default": ""}],
        "ref_images": ["COMFY_AUTOGROW_V3", {
            "template": {"input": {"optional": {"ref_image": ["IMAGE", {}]}},
                         "prefix": "ref_image", "min": 0, "max": 4}}],
    }},
    "output": ["CONDITIONING"], "output_name": ["conditioning"]}


def test_autogrow_entries_are_wired_by_their_expanded_names():
    """The API graph carries ref_image0/ref_image1, never the 'ref_images' wrapper — that
    is the shape ComfyUI expands back into the node's list."""
    models = {"slots": [
        {"id": "li", "node_class": "LoadImage", "inputs": {}, "wires": {}},
        {"id": "r", "node_class": "MiniMaxH3ReferenceToVideo", "inputs": {"prompt": "hi"},
         "wires": {},
         "input_sources": {"ref_images.ref_image0": "out:li:IMAGE", "ref_images.ref_image1": "timeline"}},
    ]}
    graph, report = builder.build(REF_OI, models, PARAMS, media={"filename": "scene.png"})
    ins = graph["slot_r"]["inputs"]
    assert ins["ref_images.ref_image0"] == ["slot_li", 0]
    assert ins["ref_images.ref_image1"] == ["media_load", 0]
    assert "ref_images" not in ins                 # the template name itself is never sent
    # an unwired entry is simply absent — never blocking, never an empty placeholder
    assert "ref_images.ref_image2" not in ins and "ref_images.ref_image3" not in ins
    assert not any("ref_image" in m for m in report["blocking"])


def test_a_single_image_producer_is_not_copied_into_every_autogrow_slot():
    """Auto-wire fills a lone unambiguous input; for a LIST input that would duplicate one
    reference across all ten indices. Autogrow entries are explicit-only."""
    models = {"slots": [
        {"id": "li", "node_class": "LoadImage", "inputs": {}, "wires": {}},
        {"id": "r", "node_class": "MiniMaxH3ReferenceToVideo", "inputs": {}, "wires": {}},
    ]}
    graph, report = builder.build(REF_OI, models, PARAMS)
    assert not [k for k in graph["slot_r"]["inputs"] if k.startswith("ref_image")]
    assert not any("ref_image" in m for m in report["unsatisfied"] + report["ambiguous"])


def test_a_node_widget_can_be_driven_by_the_project_prompt():
    """A custom node with its own prompt field takes the run's prompt through a linked
    input, instead of the user keeping two copies of the text in sync by hand."""
    models = {"slots": [{"id": "r", "node_class": "MiniMaxH3ReferenceToVideo",
                         "inputs": {"prompt": "stale text"}, "wires": {}}],
              "links": [{"id": "l1", "source": "editor", "editor_key": "prompt",
                         "members": [{"slotId": "r", "input": "prompt"}]}]}
    graph, _ = builder.build(REF_OI, models, PARAMS)
    assert graph["slot_r"]["inputs"]["prompt"] == "scene one"


def test_a_linked_text_takes_the_expanded_prompt_not_the_raw_one():
    """The node on the other end encodes the string as it arrives — it does none of what
    Studio does to a prompt. So a linked text gets the shortcut/$variable-expanded version,
    while Studio's own port keeps the raw text it expands per scene itself."""
    models = {"slots": [{"id": "r", "node_class": "MiniMaxH3ReferenceToVideo",
                         "inputs": {}, "wires": {}}],
              "links": [{"id": "l1", "source": "editor", "editor_key": "prompt",
                         "members": [{"slotId": "r", "input": "prompt"}]}]}
    params = dict(PARAMS, prompt="/greet $hero", expanded={"prompt": "hello Rin"})
    graph, _ = builder.build(REF_OI, models, params)
    assert graph["slot_r"]["inputs"]["prompt"] == "hello Rin"
    assert graph["pos"]["inputs"]["value"] == "/greet $hero"   # Studio still expands its own


def test_anchor_and_postfix_are_linkable_texts_of_their_own():
    """A node encoding on its own has no idea Studio would wrap each scene in these."""
    models = {"slots": [{"id": "r", "node_class": "MiniMaxH3ReferenceToVideo",
                         "inputs": {}, "wires": {}}],
              "links": [{"id": "l1", "source": "editor", "editor_key": "postfix",
                         "members": [{"slotId": "r", "input": "prompt"}]}]}
    params = dict(PARAMS, expanded={"postfix": "cinematic lighting"})
    graph, _ = builder.build(REF_OI, models, params)
    assert graph["slot_r"]["inputs"]["prompt"] == "cinematic lighting"


def test_a_core_combo_value_missing_on_this_machine_falls_back_instead_of_blocking():
    """ComfyUI validates combo values against the LIVE list and rejects the whole prompt if
    one is stale — so a projector/LoRA left selected on a machine that doesn't have the file
    stopped every run, even with the feature switched off."""
    oi = dict(OI)
    sampler = {k: dict(v) for k, v in oi["FunPackLTXAVSceneChainSampler"].items()
               if k in ("input",)}
    sampler["input"] = {"required": dict(oi["FunPackLTXAVSceneChainSampler"]["input"]["required"])}
    sampler["input"]["required"]["identity_projector"] = [["None"], {"default": "None"}]
    oi["FunPackLTXAVSceneChainSampler"] = {**oi["FunPackLTXAVSceneChainSampler"], **sampler}
    params = dict(PARAMS)
    params["sampler_inputs"] = {"identity_projector": "Best_FaceID_v1.0_ArcFace_Projector.safetensors"}
    graph, report = builder.build(oi, {"slots": []}, params)
    assert graph["sampler"]["inputs"]["identity_projector"] == "None"
    assert any("identity_projector" in m and "not installed" in m for m in report["unsatisfied"])


def test_an_autogrow_entry_saved_under_its_bare_name_still_wires():
    """Configs written before the dotted socket id was known hold "ref_image0". Sending that
    is what produced "execute() got an unexpected keyword argument" — map it back instead of
    letting an old config keep failing every run."""
    models = {"slots": [
        {"id": "li", "node_class": "LoadImage", "inputs": {}, "wires": {"IMAGE": "node:r:ref_image1"}},
        {"id": "r", "node_class": "MiniMaxH3ReferenceToVideo", "inputs": {}, "wires": {},
         "input_sources": {"ref_image0": "out:li:IMAGE"}},
    ]}
    graph, _ = builder.build(REF_OI, models, PARAMS)
    ins = graph["slot_r"]["inputs"]
    assert ins["ref_images.ref_image0"] == ["slot_li", 0]   # via input_sources
    assert ins["ref_images.ref_image1"] == ["slot_li", 0]   # via the mirrored wire
    assert "ref_image0" not in ins and "ref_image1" not in ins


# ── media marked "R" in the bin, wired to node inputs ─────────────────────────

REF_OI["LoadAudio"] = {"input": {"required": {"audio": [["a.wav"]]}}, "output": ["AUDIO"],
                       "output_name": ["AUDIO"]}
REF_OI["LoadVideo"] = {"input": {"required": {"file": [["a.mp4"]]}}, "output": ["VIDEO"],
                       "output_name": ["VIDEO"]}


def _ref_params(*refs):
    p = dict(PARAMS)
    p["references"] = [dict(r, index=i + 1) for i, r in enumerate(refs)]
    return p


def test_a_marked_reference_injects_its_own_loader_and_wires_it():
    models = {"slots": [
        {"id": "r", "node_class": "MiniMaxH3ReferenceToVideo", "inputs": {}, "wires": {},
         "input_sources": {"ref_images.ref_image0": "ref:m1"}},
    ]}
    params = _ref_params({"id": "m1", "kind": "image", "name": "face.png",
                          "filename": "funpack_movie_m1.png"})
    graph, report = builder.build(REF_OI, models, params)
    load = next(n for n, d in graph.items() if d["class_type"] == "LoadImage" and n != "media_load")
    assert graph[load]["inputs"]["image"] == "funpack_movie_m1.png"
    assert graph["slot_r"]["inputs"]["ref_images.ref_image0"] == [load, 0]
    assert not any("reference" in m for m in report["blocking"])


def test_two_sockets_on_one_reference_share_a_single_loader():
    """Decoding the same clip twice would cost real time on every run — and for a video
    reference, that is an ffmpeg decode."""
    models = {"slots": [
        {"id": "r", "node_class": "MiniMaxH3ReferenceToVideo", "inputs": {}, "wires": {},
         "input_sources": {"ref_images.ref_image0": "ref:m1", "ref_images.ref_image1": "ref:m1"}},
    ]}
    params = _ref_params({"id": "m1", "kind": "image", "name": "face.png",
                          "filename": "funpack_movie_m1.png"})
    graph, _ = builder.build(REF_OI, models, params)
    loaders = [n for n, d in graph.items() if d["class_type"] == "LoadImage"]
    assert len(loaders) == 1
    ins = graph["slot_r"]["inputs"]
    assert ins["ref_images.ref_image0"] == ins["ref_images.ref_image1"] == [loaders[0], 0]


# ── numbered reference SLOTS ("Reference image 1") ────────────────────────────
# Wiring a socket to a particular media id means re-opening the node page every time you
# change your mind about which reference goes where. A slot is wired once and re-points
# itself: "Reference image 1" is whatever is marked first among the image references.


def _slot_models(**srcs):
    return {"slots": [
        {"id": "r", "node_class": "MiniMaxH3ReferenceToVideo", "inputs": {}, "wires": {},
         "input_sources": dict(srcs)},
    ]}


def _ref_problems(report):
    """Complaints about the reference wiring only — these fixtures are deliberately partial
    pipelines, so unrelated core inputs are expected to be unsatisfied."""
    return [m for m in report["blocking"] + report["unsatisfied"]
            if "ref" in m.lower()]


def _img(mid, name):
    return {"id": mid, "kind": "image", "name": name, "filename": f"funpack_movie_{mid}.png"}


def test_reference_slot_one_resolves_to_the_first_marked_image():
    graph, report = builder.build(
        REF_OI, _slot_models(**{"ref_images.ref_image0": "ref#image:1"}),
        _ref_params(_img("m1", "a.png"), _img("m2", "b.png")))
    load = graph[graph["slot_r"]["inputs"]["ref_images.ref_image0"][0]]
    assert load["inputs"]["image"] == "funpack_movie_m1.png"
    assert not _ref_problems(report)


def test_reordering_the_marks_repoints_the_same_wiring():
    """The whole point: shuffle references in the bin, touch no node settings."""
    models = _slot_models(**{"ref_images.ref_image0": "ref#image:1"})
    first, second = _img("m1", "a.png"), _img("m2", "b.png")
    g1, _ = builder.build(REF_OI, models, _ref_params(first, second))
    g2, _ = builder.build(REF_OI, models, _ref_params(second, first))
    assert (g1[g1["slot_r"]["inputs"]["ref_images.ref_image0"][0]]["inputs"]["image"]
            == "funpack_movie_m1.png")
    assert (g2[g2["slot_r"]["inputs"]["ref_images.ref_image0"][0]]["inputs"]["image"]
            == "funpack_movie_m2.png")


def test_slots_are_numbered_per_kind():
    """Marking an audio file must not shift the image slots out from under the wiring."""
    audio = {"id": "a1", "kind": "audio", "name": "v.wav", "filename": "funpack_movie_a1.wav"}
    graph, _ = builder.build(
        REF_OI, _slot_models(**{"ref_images.ref_image0": "ref#image:1"}),
        _ref_params(audio, _img("m1", "a.png")))
    load = graph[graph["slot_r"]["inputs"]["ref_images.ref_image0"][0]]
    assert load["inputs"]["image"] == "funpack_movie_m1.png"


def test_an_unmarked_slot_leaves_the_socket_empty_and_says_nothing():
    """An unused reference slot is a normal state, not a setup mistake."""
    graph, report = builder.build(
        REF_OI, _slot_models(**{"ref_images.ref_image0": "ref#image:1",
                                "ref_images.ref_image1": "ref#image:2"}),
        _ref_params(_img("m1", "a.png")))
    ins = graph["slot_r"]["inputs"]
    assert isinstance(ins["ref_images.ref_image0"], list)
    assert not isinstance(ins.get("ref_images.ref_image1"), list)
    assert not _ref_problems(report)


def test_no_references_at_all_is_silent():
    _graph, report = builder.build(
        REF_OI, _slot_models(**{"ref_images.ref_image0": "ref#image:1"}), _ref_params())
    assert not _ref_problems(report)


def test_an_empty_slot_is_not_quietly_filled_with_something_else():
    """Auto-wire must not substitute a different image for the reference you asked for."""
    models = {"slots": [
        {"id": "img", "node_class": "LoadImage", "inputs": {}, "wires": {}},
        {"id": "r", "node_class": "MiniMaxH3ReferenceToVideo", "inputs": {}, "wires": {},
         "input_sources": {"ref_images.ref_image0": "ref#image:1"}},
    ]}
    graph, _ = builder.build(REF_OI, models, _ref_params())
    assert not isinstance(graph["slot_r"]["inputs"].get("ref_images.ref_image0"), list)


def test_a_malformed_slot_says_it_is_malformed_not_that_nothing_is_marked():
    """"No reference marked" sends the user to the Media Bin to fix a value that is wrong in
    the node. The two are different problems and must not read the same."""
    _graph, report = builder.build(
        REF_OI, _slot_models(**{"ref_images.ref_image0": "ref#image:nope"}),
        _ref_params(_img("m1", "a.png")))
    assert any("not a reference slot" in u for u in report["unsatisfied"])
    assert not any("none marked" in w for w in report["wired"])


def test_the_loader_matches_what_the_destination_socket_asks_for():
    """An audio reference feeding an AUDIO socket needs LoadAudio, not LoadImage — the
    loader is chosen by the socket's type, not by guessing from the file."""
    oi = dict(REF_OI)
    oi["AudioNode"] = {"input": {"required": {"clip": ["AUDIO"]}}, "output": ["CONDITIONING"]}
    models = {"slots": [
        {"id": "a", "node_class": "AudioNode", "inputs": {}, "wires": {},
         "input_sources": {"clip": "ref:m2"}},
    ]}
    params = _ref_params({"id": "m2", "kind": "audio", "name": "voice.wav",
                          "filename": "funpack_movie_m2.wav"})
    graph, _ = builder.build(oi, models, params)
    load = next(n for n, d in graph.items() if d["class_type"] == "LoadAudio")
    assert graph[load]["inputs"]["audio"] == "funpack_movie_m2.wav"
    assert graph["slot_a"]["inputs"]["clip"] == [load, 0]


def test_a_reference_whose_file_is_gone_reports_instead_of_wiring_nothing():
    models = {"slots": [
        {"id": "r", "node_class": "MiniMaxH3ReferenceToVideo", "inputs": {}, "wires": {},
         "input_sources": {"ref_images.ref_image0": "ref:missing"}},
    ]}
    graph, report = builder.build(REF_OI, models, _ref_params())
    assert "ref_images.ref_image0" not in graph["slot_r"]["inputs"]
    assert any("no longer in the media bin" in m for m in report["unsatisfied"])


def test_no_installed_loader_for_the_socket_type_is_reported():
    """A video reference aimed at an IMAGE socket needs a frames loader; with none
    installed, say so rather than silently leaving the input unwired."""
    oi = {k: v for k, v in REF_OI.items() if k not in ("LoadVideo",)}
    oi["FramesNode"] = {"input": {"required": {"frames": ["IMAGE"]}}, "output": ["CONDITIONING"]}
    models = {"slots": [
        {"id": "f", "node_class": "FramesNode", "inputs": {}, "wires": {},
         "input_sources": {"frames": "ref:m3"}},
    ]}
    params = _ref_params({"id": "m3", "kind": "video", "name": "clip.mp4",
                          "filename": "funpack_movie_m3.mp4"})
    graph, report = builder.build(oi, models, params)
    assert "frames" not in graph["slot_f"]["inputs"]
    assert any("no installed node can load a video reference" in m for m in report["unsatisfied"])


def test_h3_audio_decodes_from_the_sampler_latent_not_a_separated_stream():
    """ComfyUI's official H3 templates feed the raw sampler latent to BOTH decodes and let
    each VAE take its own stream. Unbinding first hands the audio VAE a different object
    than the reference graph does. LTX is unchanged — it keeps the separate step."""
    graph, _ = builder.build(H3_OI, H3_MODELS, {"prompt": "a shot"})
    assert graph["audiodec"]["inputs"]["samples"] == ["sampler", 0]
    # `separate` stays: it still supplies the VIDEO latent to the refinement save.
    assert graph["saveref"]["inputs"]["latent"] == ["separate", 0]

    ltx, _ = builder.build(OI, {"slots": []}, PARAMS)
    assert ltx["audiodec"]["inputs"]["samples"] == ["separate", 1]


# ── a link that does not fire says so ────────────────────────────────────────
# Every one of these used to be silent: the node kept its own widget value, so a project
# setting the user had just changed did not reach the graph and nothing said which value won.

def _link_models(members, **link):
    return {
        "slots": [{"id": "a", "node_class": "ImgProc", "inputs": {"length": 50}, "wires": {}}],
        "links": [{"id": "L", "name": "Project · Size", "source": "editor",
                   "editor_key": "num_frames_per_scene", "members": members, **link}],
    }


def test_a_link_that_fires_is_reported_with_its_value():
    graph, report = builder.build(OI, _link_models([{"slotId": "a", "input": "length"}]), PARAMS)
    assert graph["slot_a"]["inputs"]["length"] == 121
    assert any("Project · Size" in w and "121" in w for w in report["wired"])


def test_a_link_pointing_at_a_deleted_node_is_reported():
    """Re-adding a node gives it a new slot id; the link still names the old one."""
    models = _link_models([{"slotId": "gone", "input": "length"}])
    _graph, report = builder.build(OI, models, PARAMS)
    assert any("no longer in the pipeline" in u for u in report["ignored"])
    # Reported, never blocking: the run is still valid, it just isn't what was set.
    assert not any("no longer in the pipeline" in b for b in report["blocking"])


def test_a_link_naming_a_widget_the_node_does_not_have_is_reported():
    """Writing an unknown key sends ComfyUI something it ignores, so the node keeps its
    default and the link looks like it fired."""
    models = _link_models([{"slotId": "a", "input": "widht"}])
    graph, report = builder.build(OI, models, PARAMS)
    assert "widht" not in graph["slot_a"]["inputs"]
    assert graph["slot_a"]["inputs"]["length"] == 50
    assert any("has no widget called" in u for u in report["ignored"])


def test_a_link_with_nothing_to_send_is_reported():
    models = _link_models([{"slotId": "a", "input": "length"}], editor_key="nonesuch")
    graph, report = builder.build(OI, models, PARAMS)
    assert graph["slot_a"]["inputs"]["length"] == 50
    assert any("had no value to send" in u for u in report["ignored"])


def test_a_link_beats_the_value_saved_on_the_slot():
    """The whole point: the project's number wins over whatever the node was left set to."""
    graph, _ = builder.build(OI, _link_models([{"slotId": "a", "input": "length"}]), PARAMS)
    assert graph["slot_a"]["inputs"]["length"] == 121 != 50


def test_bypassing_a_connected_run_of_nodes_is_not_refused_by_its_own_members():
    """Two chained LoRAs bypassed together. The first one's MODEL is read only by the
    second, which is also going — so once both are gone nothing reads it and it needs no
    pass-through. Judging each node against a graph that still held its doomed siblings
    refused this, which meant no group of connected nodes could ever be bypassed at once.
    """
    models = {"slots": [
        {"id": "u", "node_class": "UnetLoader", "inputs": {}, "wires": {}},
        {"id": "l1", "node_class": "LTXICLoRALoaderModelOnly", "bypassed": True,
         "inputs": {}, "input_sources": {"model": "out:u:MODEL"}, "wires": {}},
        {"id": "l2", "node_class": "LTXICLoRALoaderModelOnly", "bypassed": True,
         "inputs": {}, "input_sources": {"model": "out:l1:model"},
         "wires": {"model": "port:FunPackStudio.model"}},
    ]}
    graph, report = builder.build(OI, models, PARAMS)
    assert "slot_l1" not in graph and "slot_l2" not in graph
    assert not any("bypass" in b for b in report["blocking"]), report["blocking"]
    # both collapse away and Studio ends up on the loader itself
    assert graph["studio"]["inputs"]["model"] == ["slot_u", 0]


# ── the default pipeline actually builds ──────────────────────────────────────
# Seeding loaders is only worth anything if the graph they produce is complete. This runs
# the real recipe through the real builder: what a user gets on a fresh project, minus the
# model files they still have to pick.
from movie_editor.backend import pipeline_wiring  # noqa: E402

OI_DEFAULTS = dict(OI, **{
    "FunPackDiffusionModelLoader": {
        "input": {"required": {"model_name": [["m.safetensors"]],
                               "weight_dtype": [["default", "fp8_e4m3fn"], {"default": "default"}]}},
        "output": ["MODEL", "STRING"], "output_name": ["MODEL", "status"]},
    "FunPackCLIPLoader": {
        "input": {"required": {"clip_list": ["STRING", {"default": "[]", "funpack_list": {}}],
                               "type": [["ltxv"], {"default": "ltxv"}]}},
        "output": ["CLIP", "STRING"], "output_name": ["CLIP", "status"]},
    "FunPackVAELoader": {
        "input": {"required": {"vae_name": [["v.safetensors"]],
                               "dtype": [["default", "bf16"], {"default": "default"}]}},
        "output": ["VAE", "STRING"], "output_name": ["VAE", "status"]},
    "LTXVEmptyLatentAudio": {
        "input": {"required": {"frames_number": ["INT", {"default": 97}],
                               "frame_rate": ["FLOAT,INT", {"default": 25.0, "widgetType": "FLOAT"}],
                               "audio_vae": ["VAE", {}]}},
        "output": ["LATENT"], "output_name": ["Latent"]},
    "CLIPTextEncode": {
        "input": {"required": {"text": ["STRING", {"default": ""}], "clip": ["CLIP", {}]}},
        "output": ["CONDITIONING"], "output_name": ["CONDITIONING"]},
    "FunPackLoraLoader": {
        "input": {"required": {"model": ["MODEL", {}],
                               "lora_list": ["STRING", {"default": "[]",
                                                        "funpack_list": {"allow_empty": True}}]},
                  "optional": {"clip": ["CLIP", {}],
                               "per_block": ["BOOLEAN", {"default": False}]}},
        "output": ["MODEL", "CLIP", "FP_LORA_STACK", "STRING"],
        "output_name": ["MODEL", "CLIP", "lora_stack", "status"]},
})


def _seeded_models(family="ltxav"):
    models = {"slots": [], "model_family": family}
    pipeline_wiring.seed_default_pipeline(models, OI_DEFAULTS)
    for slot in models["slots"]:            # the files the user picks; everything else is set
        if slot["node_class"] == "FunPackDiffusionModelLoader":
            slot["inputs"]["model_name"] = "m.safetensors"
        if slot["node_class"] == "FunPackVAELoader":
            slot["inputs"]["vae_name"] = "v.safetensors"
    return models


def test_a_freshly_seeded_pipeline_builds_with_nothing_left_to_wire():
    """t2v: only the i2v anchor is unfed, and that comes from a scene's image, not a loader."""
    _graph, report = builder.build(OI_DEFAULTS, _seeded_models(), PARAMS)
    assert report["blocking"] == []
    assert len(report["unsatisfied"]) == 1
    assert "source_image" in report["unsatisfied"][0]


def test_a_freshly_seeded_pipeline_is_complete_for_i2v_too():
    """With a scene image the builder materialises it as a LoadImage, which is the IMAGE
    producer Studio's anchor was waiting for — so nothing is left over at all."""
    _graph, report = builder.build(OI_DEFAULTS, _seeded_models(), PARAMS,
                                   media={"filename": "scene.png"})
    assert report["blocking"] == []
    assert report["unsatisfied"] == []


def test_the_seeded_loaders_reach_the_core_ports_they_were_wired_to():
    graph, _ = builder.build(OI_DEFAULTS, _seeded_models(), PARAMS)
    by_class = {n["class_type"]: nid for nid, n in graph.items()}
    assert graph["studio"]["inputs"]["model"] == [by_class["FunPackLoraLoader"], 0]
    assert graph["studio"]["inputs"]["clip"] == [by_class["FunPackCLIPLoader"], 0]
    # two VAE loaders: the video one feeds the sampler, the audio one the audio decode
    video_vae = graph["sampler"]["inputs"]["vae"]
    audio_vae = graph["audiodec"]["inputs"]["audio_vae"]
    assert graph[video_vae[0]]["class_type"] == "FunPackVAELoader"
    assert graph[audio_vae[0]]["class_type"] == "FunPackVAELoader"
    assert video_vae[0] != audio_vae[0]


def test_the_seeded_lora_loader_is_a_wire_the_model_passes_through():
    """It is seeded empty on purpose: the hop every LoRA needs is already in the graph, so
    using one is picking a file — not adding a node and rewiring the model path."""
    graph, report = builder.build(OI_DEFAULTS, _seeded_models(), PARAMS)
    by_class = {n["class_type"]: nid for nid, n in graph.items()}
    lora = by_class["FunPackLoraLoader"]
    assert graph[lora]["inputs"]["model"] == [by_class["FunPackDiffusionModelLoader"], 0]
    assert graph[lora]["inputs"]["lora_list"] == "[]"
    assert graph["studio"]["inputs"]["model"] == [lora, 0]
    assert report["blocking"] == []


def test_the_seeded_audio_latent_follows_the_project_not_its_own_widgets():
    graph, _ = builder.build(OI_DEFAULTS, _seeded_models(), PARAMS)
    audio = next(n for n in graph.values() if n["class_type"] == "LTXVEmptyLatentAudio")
    assert audio["inputs"]["frames_number"] == ["frames", 0]
    assert audio["inputs"]["frame_rate"] == ["fps", 0]
    # and it takes the AUDIO vae, not whichever VAE auto-wire happened to reach first
    assert audio["inputs"]["audio_vae"] == graph["audiodec"]["inputs"]["audio_vae"]


def test_a_conditioning_node_never_captures_the_prompt_by_itself():
    """positive_conditioning replaces the typed prompt wholesale — shortcuts, $variables and
    the per-scene split all go with it — so it must be something the user wired on purpose."""
    models = _seeded_models()
    models["slots"].append({
        "id": "enc", "role": "custom", "node_class": "CLIPTextEncode",
        "inputs": {"text": "hello"}, "wires": {}, "input_sources": {}})
    graph, report = builder.build(OI_DEFAULTS, models, PARAMS)
    assert "positive_conditioning" not in graph["studio"]["inputs"]
    assert not any("positive_conditioning" in m for m in report["blocking"])


def test_a_conditioning_wire_the_user_asked_for_is_honoured():
    models = _seeded_models()
    models["slots"].append({
        "id": "enc", "role": "custom", "node_class": "CLIPTextEncode",
        "inputs": {"text": "hello"},
        "wires": {"CONDITIONING": ["port:FunPackStudio.positive_conditioning"]},
        "input_sources": {}})
    graph, report = builder.build(OI_DEFAULTS, models, PARAMS)
    assert isinstance(graph["studio"]["inputs"]["positive_conditioning"], list)
    assert report["blocking"] == []


def test_an_unwired_pass_through_is_not_counted_as_a_source():
    """The seeded LoRA loader hands CLIP straight back. With its clip input unwired it emits
    nothing, and counting it made every other CLIP consumer read as ambiguous."""
    models = _seeded_models()
    models["slots"].append({
        "id": "enc", "role": "custom", "node_class": "CLIPTextEncode",
        "inputs": {"text": "hello"},
        "wires": {"CONDITIONING": ["port:FunPackStudio.positive_conditioning"]},
        "input_sources": {}})
    graph, report = builder.build(OI_DEFAULTS, models, PARAMS)
    by_class = {n["class_type"]: nid for nid, n in graph.items()}
    assert graph[by_class["CLIPTextEncode"]]["inputs"]["clip"] == [by_class["FunPackCLIPLoader"], 0]
    assert report["ambiguous"] == []


# ── bypass as an A/B switch ───────────────────────────────────────────────────
# Wiring two alternatives at one input and bypassing the one you are not using is the
# natural way to switch between them (MiniMax H3's ref-to-video and first-last-to-video
# both produce the sampler's LATENT). Before this, bypassing either one failed: the
# bypassed node still won the input, and then could not pass anything through it.

# Two distinct i2v-shaped classes, so a message can be attributed to ONE of them. Both have
# a LATENT output and no LATENT input — exactly the node that cannot pass its own output
# through, which is what made this case fail.
OI_AB = dict(OI, **{
    "RefToVideo": {"input": {"required": {"vae": ["VAE"], "image": ["IMAGE"]}},
                   "output": ["LATENT"], "output_name": ["latent"]},
    "FirstLastToVideo": {"input": {"required": {"vae": ["VAE"], "image": ["IMAGE"]}},
                         "output": ["LATENT"], "output_name": ["latent"]},
})


def _two_alternatives(bypassed):
    return {"full_control": True, "slots": [
        {"id": "u", "node_class": "UnetLoader", "inputs": {},
         "wires": {"MODEL": "port:FunPackStudio.model"}},
        {"id": "v", "node_class": "VaeLoader", "inputs": {}, "wires": {}},
        {"id": "img", "node_class": "LoadImage", "inputs": {}, "wires": {}},
        {"id": "r2v", "node_class": "RefToVideo", "inputs": {}, "bypassed": bypassed == "r2v",
         "wires": {"latent": "port:FunPackStudio.latent"}},
        {"id": "fl2v", "node_class": "FirstLastToVideo", "inputs": {},
         "bypassed": bypassed == "fl2v",
         "wires": {"latent": "port:FunPackStudio.latent"}},
    ]}


def _bypass_blocks(report):
    return [b for b in report["blocking"] if "bypass" in b]


@pytest.mark.parametrize("off,on", [("r2v", "fl2v"), ("fl2v", "r2v")])
def test_bypassing_one_of_two_alternatives_hands_the_input_to_the_other(off, on):
    graph, report = builder.build(OI_AB, _two_alternatives(off), PARAMS)
    assert f"slot_{off}" not in graph
    assert f"slot_{on}" in graph
    assert graph["studio"]["inputs"]["latent"] == [f"slot_{on}", 0]
    assert _bypass_blocks(report) == []


def test_bypassing_the_only_producer_of_a_required_input_still_blocks():
    """The relaxation is 'somebody else feeds it', not 'stop checking'. With no alternative
    the bypass is still the thing that would silently empty a required input."""
    models = {"full_control": True, "slots": [
        {"id": "v", "node_class": "VaeLoader", "inputs": {}, "wires": {}},
        {"id": "img", "node_class": "LoadImage", "inputs": {}, "wires": {}},
        {"id": "r2v", "node_class": "RefToVideo", "inputs": {}, "bypassed": True,
         "wires": {"latent": "port:LTXVConcatAVLatent.video_latent"}},
    ]}
    _graph, report = builder.build(OI_AB, models, PARAMS)
    assert any("bypass needs exactly one input" in b for b in report["blocking"])


def test_a_bypassed_nodes_own_unwired_inputs_are_not_demanded():
    """Telling someone to wire a node they explicitly switched off is asking them to fix
    something they already decided not to use."""
    models = _two_alternatives("r2v")
    models["slots"] = [s for s in models["slots"] if s["id"] != "img"]   # no IMAGE producer
    _graph, report = builder.build(OI_AB, models, PARAMS)
    assert not any("RefToVideo" in b for b in report["blocking"]), report["blocking"]
    # the ACTIVE alternative is still held to the same standard
    assert any("FirstLastToVideo" in b for b in report["blocking"]), report["blocking"]


def test_a_bypassed_node_feeding_an_optional_input_needs_no_passthrough():
    """Studio's source_image is optional. Losing it means Studio runs without it — which is
    what bypassing the node that fed it means, not a graph to repair."""
    models = {"full_control": True, "slots": [
        {"id": "um", "node_class": "UpscaleModelLoader", "inputs": {}, "wires": {}},
        {"id": "img", "node_class": "LoadImage", "inputs": {}, "wires": {}},
        {"id": "up", "node_class": "ImageUpscaleWithModel", "inputs": {}, "bypassed": True,
         "wires": {"IMAGE": "port:FunPackStudio.source_image"}},
    ]}
    _graph, report = builder.build(OI, models, PARAMS)
    assert _bypass_blocks(report) == []


def test_two_alternatives_at_one_port_are_not_a_double_claim_when_one_is_bypassed():
    """Guided mode allows one source per built-in input. A bypassed slot is not a source."""
    from movie_editor.backend import pipeline_wiring
    models = {k: v for k, v in _two_alternatives("r2v").items() if k != "full_control"}
    assert not any("already wired from" in e
                   for e in pipeline_wiring.validate_models_wiring(models))


def test_two_ACTIVE_sources_on_one_port_are_still_refused():
    from movie_editor.backend import pipeline_wiring
    models = {k: v for k, v in _two_alternatives("none").items() if k != "full_control"}
    assert any("already wired from" in e
               for e in pipeline_wiring.validate_models_wiring(models))
