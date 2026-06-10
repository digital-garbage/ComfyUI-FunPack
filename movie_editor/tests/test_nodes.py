"""Tests for the pluggable node-slot discovery (object_info filtering)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from movie_editor.backend import nodes  # noqa: E402

OI = {
    "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["a.safetensors", "b.ckpt"]]}},
                               "output": ["MODEL", "CLIP", "VAE"], "display_name": "Load Checkpoint"},
    "VAELoader": {"input": {"required": {"vae_name": [["ltxv-vae.safetensors"]]}},
                  "output": ["VAE"], "display_name": "Load VAE"},
    "LoraLoader": {"input": {"required": {"model": ["MODEL"], "clip": ["CLIP"],
                                          "lora_name": [["l.safetensors"]],
                                          "strength_model": ["FLOAT", {"default": 1.0}]}},
                   "output": ["MODEL", "CLIP"], "display_name": "Load LoRA"},
    "EmptyLTXVLatentVideo": {"input": {"required": {"width": ["INT", {"default": 768}],
                                                    "length": ["INT", {"default": 97}]}},
                             "output": ["LATENT"], "display_name": "Empty LTXV Latent"},
    "KSampler": {"input": {"required": {"model": ["MODEL"], "latent_image": ["LATENT"]}},
                 "output": ["LATENT"], "display_name": "KSampler"},
    "LTXVPreprocess": {"input": {"required": {"image": ["IMAGE"], "crf": ["INT", {"default": 35}]}},
                       "output": ["IMAGE"], "display_name": "LTXV Preprocess"},
}


def names(role):
    return [c["class"] for c in nodes.candidates(OI, role)]


def test_source_roles_exclude_patchers():
    assert names("unet") == ["CheckpointLoaderSimple"]          # LoraLoader (consumes MODEL) excluded
    assert names("empty_latent") == ["EmptyLTXVLatentVideo"]    # KSampler (consumes LATENT) excluded
    assert names("clip") == ["CheckpointLoaderSimple"]


def test_vae_role_matches_all_vae_sources():
    assert set(names("video_vae")) == {"CheckpointLoaderSimple", "VAELoader"}
    assert set(names("audio_vae")) == {"CheckpointLoaderSimple", "VAELoader"}


def test_patcher_and_processor_roles():
    assert names("lora") == ["LoraLoader"]                      # MODEL in + MODEL out
    assert names("image_processing") == ["LTXVPreprocess"]      # IMAGE in + IMAGE out


def test_widget_inputs_skip_connections_and_expose_combos():
    lora = next(c for c in nodes.candidates(OI, "lora") if c["class"] == "LoraLoader")
    kinds = {w["name"]: w["kind"] for w in lora["inputs"]}
    assert "model" not in kinds and "clip" not in kinds        # connections skipped
    assert kinds["lora_name"] == "combo"
    assert kinds["strength_model"] == "float"
    ckpt = next(c for c in nodes.candidates(OI, "unet") if c["class"] == "CheckpointLoaderSimple")
    assert ckpt["inputs"][0]["choices"] == ["a.safetensors", "b.ckpt"]


def test_node_outputs_and_connection_inputs():
    lora = next(c for c in nodes.candidates(OI, "lora") if c["class"] == "LoraLoader")
    assert {o["type"] for o in lora["outputs"]} == {"MODEL", "CLIP"}
    ci = {c["name"]: c["type"] for c in lora["connection_inputs"]}
    assert ci == {"model": "MODEL", "clip": "CLIP"}        # widgets (lora_name/strength) excluded


def test_ports_from_input_types_derives_connection_points():
    it = {
        "required": {"rating": (["a", "b"], {}), "studio_settings": ("STRING", {})},
        "optional": {"model": ("MODEL",), "clip": ("CLIP",), "source_image": ("IMAGE",),
                     "latent": ("LATENT",), "positive_prompt": ("STRING", {"forceInput": True})},
    }
    ports = nodes.ports_from_input_types("Studio", "FunPackStudio", it)
    by_input = {p["input"]: p["type"] for p in ports}
    assert by_input == {
        "model": "MODEL",
        "clip": "CLIP",
        "source_image": "IMAGE",
        "latent": "LATENT",
        "positive_prompt": "STRING",
    }
    assert all(p["id"].startswith("FunPackStudio.") for p in ports)  # STRING widgets excluded


def test_all_nodes_and_describe_node():
    listing = nodes.all_nodes(OI)
    assert {n["class"] for n in listing} == set(OI)            # every node addable via "any node"
    spec = nodes.describe_node(OI, "LTXVPreprocess")
    assert spec["class"] == "LTXVPreprocess"
    assert any(o["type"] == "IMAGE" for o in spec["outputs"])
    assert nodes.describe_node(OI, "NoSuchNode") is None


def test_ports_from_object_info_exposes_core_inputs():
    oi = {"LTXVAudioVAEDecode": {"input": {"required": {"samples": ["LATENT"], "audio_vae": ["VAE"]}},
                                 "output": ["AUDIO"]}}
    ports = nodes.ports_from_object_info(oi, "LTXVAudioVAEDecode", "Audio VAE Decode")
    by = {p["input"]: p["type"] for p in ports}
    assert by == {"samples": "LATENT", "audio_vae": "VAE"}
    assert all(p["label"].startswith("Audio VAE Decode · ") for p in ports)


def test_malformed_node_is_skipped_not_fatal():
    # a node with V3-style dict outputs / odd shapes must not crash the whole list
    bad = dict(OI)
    bad["WeirdNode"] = {"input": {"required": {"x": {"nested": "dict"}}},
                        "output": {"a": "MODEL"}, "output_name": {"a": "model"},
                        "display_name": "Weird", "category": "x"}
    bad["AlsoBad"] = {"input": "not-a-dict", "output": None}
    # candidates for a role still returns the good loaders, no exception
    got = {c["class"] for c in nodes.candidates(bad, "unet")}
    assert "CheckpointLoaderSimple" in got
    # all_nodes lists everything (incl. the weird ones) without crashing
    assert {n["class"] for n in nodes.all_nodes(bad)} >= set(OI)


def test_v1_combo_and_force_input():
    """V1/io.ComfyNode nodes encode combos as ("COMBO", {"options": [...]}) and mark
    socket-only fields with forceInput=True. Both must be handled correctly."""
    v1_node = {
        "input": {
            "required": {
                "model": ["MODEL"],
                "lora_name": ["COMBO", {"options": ["a.safetensors", "b.safetensors"], "tooltip": "pick a LoRA"}],
                "strength_model": ["FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0}],
            },
            "optional": {
                "opt_path": ["STRING", {"forceInput": True, "tooltip": "absolute path override"}],
                "blocks": ["SELECTEDDITBLOCKS"],
            },
        },
        "output": ["MODEL"],
        "display_name": "LTX2 LoRA Advanced",
        "category": "test",
    }
    oi = dict(OI)
    oi["LTX2LoraLoaderAdvanced"] = v1_node
    desc = nodes.describe_node(oi, "LTX2LoraLoaderAdvanced")
    widget_map = {w["name"]: w for w in desc["inputs"]}
    # lora_name is a V1-style COMBO — must appear as combo widget with choices
    assert "lora_name" in widget_map
    assert widget_map["lora_name"]["kind"] == "combo"
    assert "a.safetensors" in widget_map["lora_name"]["choices"]
    # strength_model is a normal FLOAT widget
    assert widget_map["strength_model"]["kind"] == "float"
    # opt_path has forceInput=True → must be a socket, NOT a widget
    assert "opt_path" not in widget_map
    # blocks is a custom connection type — also not a widget
    assert "blocks" not in widget_map
    # model is a connection — not a widget
    assert "model" not in widget_map
    # connection_inputs: model and blocks (custom type)
    ci = {c["name"]: c["type"] for c in desc["connection_inputs"]}
    assert ci["model"] == "MODEL"
    assert ci["blocks"] == "SELECTEDDITBLOCKS"


def test_models_store_roundtrip(tmp_path, monkeypatch):
    from movie_editor.backend import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert nodes.load_models() == {"slots": []}
    nodes.save_models({"slots": [{"id": "x", "role": "unet", "node_class": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a.safetensors"}}]})
    assert nodes.load_models()["slots"][0]["node_class"] == "CheckpointLoaderSimple"
