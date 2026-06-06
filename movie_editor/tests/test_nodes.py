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


def test_models_store_roundtrip(tmp_path, monkeypatch):
    from movie_editor.backend import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert nodes.load_models() == {"slots": []}
    nodes.save_models({"slots": [{"id": "x", "role": "unet", "node_class": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a.safetensors"}}]})
    assert nodes.load_models()["slots"][0]["node_class"] == "CheckpointLoaderSimple"
