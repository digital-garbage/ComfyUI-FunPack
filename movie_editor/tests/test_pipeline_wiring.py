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


def test_lora_model_may_terminate_at_studio():
    ok = pipeline_wiring.validate_port_wire(
        role="lora", out_type="MODEL", out_name="MODEL",
        target="port:FunPackStudio.model", models={"slots": []},
    )
    assert ok is None


def test_unet_may_chain_to_lora_via_node_wire():
    ok = pipeline_wiring.validate_port_wire(
        role="unet", out_type="MODEL", out_name="MODEL",
        target="node:lora1:model", models={"slots": []},
    )
    assert ok is None


def test_custom_model_may_terminate_at_studio():
    ok = pipeline_wiring.validate_port_wire(
        role="custom", out_type="MODEL", out_name="MODEL",
        target="port:FunPackStudio.model", models={"slots": []},
    )
    assert ok is None


def test_latent_chains_to_studio_not_concat_video():
    ok = pipeline_wiring.validate_port_wire(
        role="empty_latent", out_type="LATENT", out_name="LATENT",
        target="port:FunPackStudio.latent", models={"slots": []},
    )
    assert ok is None

    err = pipeline_wiring.validate_port_wire(
        role="empty_latent", out_type="LATENT", out_name="LATENT",
        target="port:LTXVConcatAVLatent.video_latent", models={"slots": []},
    )
    assert err is not None
    assert "Studio · latent" in err

    ok = pipeline_wiring.validate_port_wire(
        role="custom", out_type="LATENT", out_name="LATENT",
        target="port:FunPackStudio.latent", models={"slots": []},
    )
    assert ok is None


def test_image_chains_to_studio_source_image():
    ok = pipeline_wiring.validate_port_wire(
        role="custom", out_type="IMAGE", out_name="IMAGE",
        target="port:FunPackStudio.source_image", models={"slots": []},
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


def test_core_overrides_open_port_apply_when_guided():
    models = {
        "full_control": False,
        "core_overrides": {"studio": {"latent": "out:custom:LATENT"}},
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
            {"id": "custom", "node_class": "EmptyLatent", "inputs": {}, "role": "custom",
             "wires": {"LATENT": "port:FunPackStudio.latent"}},
        ],
    }
    graph, _report = builder.build(OI, models, PARAMS)
    assert graph["studio"]["inputs"]["latent"] == ["slot_custom", 0]


def test_core_overrides_internal_still_ignored_when_guided():
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
    model_in = next(i for i in studio["inputs"] if i["name"] == "model")
    latent_in = next(i for i in studio["inputs"] if i["name"] == "latent")
    pos_in = next(i for i in studio["inputs"] if i["name"] == "positive_prompt")
    assert model_in["locked"] is False
    assert latent_in["locked"] is False
    assert pos_in["locked"] is True


# ── a role with no rules ──────────────────────────────────────────────────────
# "Any node…" gives a slot role `custom`, which no rule table mentions. Guided mode is
# there to keep the core's INTERNAL links fixed, not to decide which node may fill an open
# socket — restricting to role rules meant a custom VAE loader could not be wired into the
# pipeline at all, and an already-saved wire read back as "(not allowed)".

def test_custom_role_may_reach_a_port_of_its_type():
    ok = pipeline_wiring.validate_port_wire(
        role="custom", out_type="VAE", out_name="VAE",
        target="port:FunPackLTXAVSceneChainSampler.vae", models={"slots": []},
    )
    assert ok is None


def test_custom_role_still_cannot_reach_an_internal_link():
    err = pipeline_wiring.validate_port_wire(
        role="custom", out_type="LATENT", out_name="LATENT",
        target="port:LTXVConcatAVLatent.video_latent", models={"slots": []},
    )
    assert err is not None and "Studio · latent" in err


def test_a_role_with_a_rule_keeps_it():
    """The fallback must not widen a role that HAS an opinion about this type."""
    assert pipeline_wiring.allowed_port_ids("audio_encoder", "LATENT", family="ltxav") == \
        ["LTXVConcatAVLatent.audio_latent"]


def test_the_panel_filters_exactly_as_the_builder_validates():
    """models.js allowedDestinations() reads type_fallback_ports for a role with no rule.
    A panel that hides a wire the builder would accept is how "(not allowed)" appeared."""
    for family in ("ltxav", "minimax_h3"):
        payload = pipeline_wiring.wiring_rules_payload(family)
        for out_type, ports in payload["type_fallback_ports"].items():
            assert ports == pipeline_wiring.allowed_port_ids(
                "custom", out_type, family=family)
        # and nothing internal leaks into it
        for ports in payload["type_fallback_ports"].values():
            assert not set(ports) & set(payload["guided_hidden_ports"])


# ── the default pipeline ──────────────────────────────────────────────────────
# Shipping FunPack's own loaders was pointless if a new project still started empty: the
# stated goal is that setting up a model is choosing files and nothing else.
OI_LOADERS = {
    "FunPackDiffusionModelLoader": {"input": {"required": {
        "model_name": [["a.safetensors", "b.safetensors"]],
        "weight_dtype": [["default", "fp8_e4m3fn"], {"default": "default"}]}},
        "output": ["MODEL"]},
    "FunPackCLIPLoader": {"input": {"required": {
        "clip_list": ["STRING", {"default": "[]", "funpack_list": {}}],
        "type": [["ltxv", "minimax"], {"default": "ltxv"}]}},
        "output": ["CLIP"]},
    "FunPackVAELoader": {"input": {"required": {
        "vae_name": [["v.safetensors"]],
        "dtype": [["default", "bf16"], {"default": "default"}]}},
        "output": ["VAE"]},
}


def test_a_fresh_pipeline_is_funpacks_own_loaders_already_wired():
    slots = pipeline_wiring.default_pipeline_slots("ltxav", OI_LOADERS)
    assert [s["role"] for s in slots] == ["unet", "clip", "video_vae", "audio_vae"]
    wired = {s["role"]: list(s["wires"].values())[0][0] for s in slots}
    assert wired["unet"] == "port:FunPackStudio.model"
    assert wired["clip"] == "port:FunPackStudio.clip"
    assert wired["video_vae"] == "port:FunPackLTXAVSceneChainSampler.vae"
    assert wired["audio_vae"] == "port:LTXVAudioVAEDecode.audio_vae"


def test_the_default_pipeline_follows_the_family():
    """H3 decodes audio with core's node, so the audio VAE lands somewhere else entirely."""
    slots = pipeline_wiring.default_pipeline_slots("minimax_h3", OI_LOADERS)
    wired = {s["role"]: list(s["wires"].values())[0][0] for s in slots}
    assert wired["audio_vae"] == "port:VAEDecodeAudio.vae"


def test_seeded_loaders_carry_declared_defaults_but_no_model_file():
    """Pre-selecting the first file would make an unconfigured loader look configured."""
    slots = {s["role"]: s for s in pipeline_wiring.default_pipeline_slots("ltxav", OI_LOADERS)}
    assert slots["unet"]["inputs"] == {"weight_dtype": "default"}
    assert "model_name" not in slots["unet"]["inputs"]
    assert slots["clip"]["inputs"] == {"clip_list": "[]", "type": "ltxv"}


def test_loaders_this_comfyui_does_not_have_are_not_seeded():
    assert pipeline_wiring.default_pipeline_slots("ltxav", {}) == []


def test_seeding_happens_once_and_is_recorded():
    models = {"slots": []}
    pipeline_wiring.seed_default_pipeline(models, OI_LOADERS)
    assert models["defaults_seeded"] is True and len(models["slots"]) == 4
    models["slots"] = []
    pipeline_wiring.seed_default_pipeline(models, OI_LOADERS)
    assert models["slots"] == []          # emptied on purpose stays empty


def test_an_existing_pipeline_is_never_replaced():
    models = {"slots": [{"id": "mine", "role": "unet", "node_class": "UNETLoader"}]}
    pipeline_wiring.seed_default_pipeline(models, OI_LOADERS)
    assert [s["id"] for s in models["slots"]] == ["mine"]
    assert models["defaults_seeded"] is True


def test_an_imported_workflow_is_never_seeded_over():
    models = {"slots": [], "workflow_import": {"name": "mine.json"}}
    pipeline_wiring.seed_default_pipeline(models, OI_LOADERS)
    assert models["slots"] == []


def test_seeding_is_deferred_when_the_node_schema_is_unavailable():
    """Marking it done without a schema would mean this project never gets loaders at all."""
    models = {"slots": []}
    pipeline_wiring.seed_default_pipeline(models, None)
    assert "defaults_seeded" not in models
    pipeline_wiring.seed_default_pipeline(models, OI_LOADERS)
    assert len(models["slots"]) == 4


OI_WITH_PLUMBING = dict(OI_LOADERS, **{
    "LTXVEmptyLatentAudio": {"input": {"required": {
        "frames_number": ["INT", {"default": 97}],
        "frame_rate": ["FLOAT,INT", {"default": 25.0, "widgetType": "FLOAT"}],
        "audio_vae": ["VAE", {}]}},
        "output": ["LATENT"], "output_name": ["Latent"]},
})


def test_the_default_pipeline_is_complete_not_just_the_loaders():
    """A required slot the user still has to find, add and wire by hand is the thing this
    is here to remove — so the family's own plumbing is seeded with it."""
    slots = {s["role"]: s for s in
             pipeline_wiring.default_pipeline_slots("ltxav", OI_WITH_PLUMBING)}
    assert "audio_encoder" in slots
    audio = slots["audio_encoder"]
    # wired by output NAME ("Latent"), while the rules table is keyed by type ("LATENT")
    assert audio["wires"] == {"Latent": ["port:LTXVConcatAVLatent.audio_latent"]}
    assert audio["input_sources"]["audio_vae"] == "out:fp_audio_vae:VAE"
    # length and rate come from the project, not from a second copy inside the node
    assert audio["input_sources"]["frames_number"] == "core:frames:0"
    assert audio["input_sources"]["frame_rate"] == "core:fps:0"


def test_plumbing_for_another_family_is_not_seeded():
    slots = pipeline_wiring.default_pipeline_slots("minimax_h3", OI_WITH_PLUMBING)
    assert "audio_encoder" not in {s["role"] for s in slots}
