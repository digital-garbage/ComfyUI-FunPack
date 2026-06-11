"""Tests for guided vs full-control pipeline wiring rules."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from movie_editor.backend import builder, pipeline_wiring  # noqa: E402
from movie_editor.tests.test_builder import OI, PARAMS  # noqa: E402


def test_wiring_locked_by_default():
    assert pipeline_wiring.wiring_locked({"slots": []}) is True
    assert pipeline_wiring.wiring_locked({"slots": [], "full_control": True}) is False
    assert pipeline_wiring.wiring_locked({"slots": [], "disable_core": True}) is False


def test_audio_latent_only_to_concat():
    err = pipeline_wiring.validate_port_wire(
        role="audio_encoder", out_type="LATENT", out_name="LATENT",
        target="port:FunPackStudio.latent", models={"slots": []},
    )
    assert err is not None
    assert "Concat AV Latent" in err

    ok = pipeline_wiring.validate_port_wire(
        role="audio_encoder", out_type="LATENT", out_name="LATENT",
        target="port:LTXVConcatAVLatent.audio_latent", models={"slots": []},
    )
    assert ok is None


def test_image_only_to_studio_source_image():
    err = pipeline_wiring.validate_port_wire(
        role="image_processing", out_type="IMAGE", out_name="IMAGE",
        target="port:LTXVConcatAVLatent.audio_latent", models={"slots": []},
    )
    assert err is not None

    ok = pipeline_wiring.validate_port_wire(
        role="image_processing", out_type="IMAGE", out_name="IMAGE",
        target="port:FunPackStudio.source_image", models={"slots": []},
    )
    assert ok is None


def test_full_control_allows_any_port():
    models = {"slots": [], "full_control": True}
    ok = pipeline_wiring.validate_port_wire(
        role="audio_encoder", out_type="LATENT", out_name="LATENT",
        target="port:FunPackStudio.latent", models=models,
    )
    assert ok is None


def test_duplicate_port_wire_blocked():
    models = {
        "slots": [
            {"id": "a", "role": "unet", "node_class": "UnetLoader", "label": "U1",
             "wires": {"MODEL": "port:FunPackStudio.model"}},
            {"id": "b", "role": "unet", "node_class": "UnetLoader", "label": "U2",
             "wires": {"MODEL": "port:FunPackStudio.model"}},
        ],
    }
    errs = pipeline_wiring.validate_models_wiring(models)
    assert any("already wired" in e for e in errs)


def test_core_overrides_ignored_when_guided():
    models = {
        "full_control": False,
        "core_overrides": {"sampler": {"vae": "out:bad:0"}},
        "slots": [
            {"id": "u", "node_class": "UnetLoader", "inputs": {}, "role": "unet",
             "wires": {"MODEL": "port:FunPackStudio.model"}},
            {"id": "c", "node_class": "ClipLoader", "inputs": {}, "role": "clip",
             "wires": {"CLIP": "port:FunPackStudio.clip"}},
            {"id": "v", "node_class": "VaeLoader", "inputs": {}, "role": "video_vae",
             "wires": {"VAE": "port:FunPackLTXAVSceneChainSampler.vae"}},
            {"id": "av", "node_class": "VaeLoader", "inputs": {}, "role": "audio_vae",
             "wires": {"VAE": "port:LTXVAudioVAEDecode.audio_vae"}},
            {"id": "ae", "node_class": "ImgProc", "inputs": {}, "role": "audio_encoder",
             "wires": {"Latent": "port:LTXVConcatAVLatent.audio_latent"}},
        ],
    }
    graph, _report = builder.build(OI, models, PARAMS)
    assert graph["sampler"]["inputs"]["vae"] == ["slot_v", 0]


def test_core_graph_marks_locked_inputs():
    nodes = builder.core_graph(OI, {"slots": [], "full_control": False})
    studio = next(n for n in nodes if n["id"] == "studio")
    sampler_in = next(i for i in studio["inputs"] if i["name"] == "model")
    assert sampler_in["locked"] is True
    concat = next(n for n in nodes if n["id"] == "concat")
    aud = next(i for i in concat["inputs"] if i["name"] == "audio_latent")
    assert aud["locked"] is True
