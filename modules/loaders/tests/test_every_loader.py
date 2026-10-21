"""The VAE, CLIP, LoRA and checkpoint loaders, every option they offer, against ComfyUI's real loading code.

A VAE is a real (random) SD-shaped one, 335 MB written once to a temp folder, because ComfyUI's VAE class is fixed to
that shape. The text encoder cannot be built small, so its slots, order, device and every family name are checked
through ComfyUI's own `CLIPType`, with the file reading replaced. LoRAs are built for the tiny real H3 in every
key dialect the matcher knows. Random weights prove the path runs; they say nothing about quality.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture(scope="module")
def tiny(comfyui, tmp_path_factory):
    import comfy.model_patcher as mp
    import comfy.supported_models as sm
    torch.manual_seed(0)
    config = sm.MiniMaxH3(dict(
        image_model="minimax_h3", hidden_size=64, num_layers=4, token_refiner_num_layers=1, num_attention_heads=2,
        attention_head_dim=32, ffn_hidden_size=64, text_dim=48, timestep_input_dim=32, time_embed_hidden_size=64,
        time_embed_dim=32, rope_inv_freq_len=4))
    config.set_inference_dtype(torch.float32, None)
    base = config.get_model({}, device=torch.device("cpu"))
    for p in base.diffusion_model.parameters():
        torch.nn.init.normal_(p, std=0.05)
        p.requires_grad_(False)
    return mp.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


@pytest.fixture(scope="module")
def folders(comfyui, tiny, tmp_path_factory):
    import folder_paths
    import safetensors.torch as st
    from comfy.ldm.models.autoencoder import AutoencoderKL
    root = tmp_path_factory.mktemp("models")
    for kind in ("vae", "loras", "checkpoints"):
        (root / kind).mkdir()
        folder_paths.add_model_folder_path(kind, str(root / kind))
    ddconfig = dict(double_z=True, z_channels=4, resolution=256, in_channels=3, out_ch=3, ch=128, ch_mult=[1, 2, 4, 4],
                    num_res_blocks=2, attn_resolutions=[], dropout=0.0)
    vae = AutoencoderKL(ddconfig=ddconfig, embed_dim=4)
    st.save_file({k: v.contiguous() for k, v in vae.state_dict().items()}, str(root / "vae" / "tiny_vae.safetensors"))
    unet = {k: v.contiguous() for k, v in tiny.model.diffusion_model.state_dict().items()}
    st.save_file(unet, str(root / "checkpoints" / "unet_only.safetensors"))
    return root


def test_the_vae_loader_decodes_in_every_dtype_it_offers(folders):
    from modules.loaders.vae.nodes import DTYPES, FunPackVAELoader
    for name in DTYPES:
        vae, status = FunPackVAELoader.execute("tiny_vae.safetensors", name).result
        assert name in status
        assert vae.decode(torch.zeros(1, 4, 4, 4)).shape == (1, 32, 32, 3)


def test_the_checkpoint_loader_says_what_a_file_without_encoder_or_vae_lacks(folders):
    from modules.loaders.checkpoint.nodes import FunPackCheckpointLoader
    model, clip, vae, status = FunPackCheckpointLoader.execute("unet_only.safetensors").result
    assert type(model.model).__name__ == "MiniMaxH3"
    assert clip is None and "(no clip, vae)" in status


def _target(tiny):
    for key, weight in tiny.model.diffusion_model.state_dict().items():
        if key.endswith(".weight") and weight.dim() == 2 and min(weight.shape) >= 8 and "blocks" in key:
            return key[:-len(".weight")], weight.shape
    raise AssertionError("no linear layer found")


DIALECTS = {
    "bare": ("", "lora_A.weight", "lora_B.weight"),
    "diffusion_model": ("diffusion_model.", "lora_A.weight", "lora_B.weight"),
    "transformer": ("transformer.", "lora_A.weight", "lora_B.weight"),
    "peft wrapper": ("base_model.model.", "lora_A.weight", "lora_B.weight"),
    "peft dit wrapper": ("base_model.model.dit.", "lora_A.weight", "lora_B.weight"),
    "dit": ("dit.", "lora_A.weight", "lora_B.weight"),
    "up and down": ("diffusion_model.", "lora_down.weight", "lora_up.weight"),
}


@pytest.mark.parametrize("dialect", DIALECTS)
def test_a_lora_in_every_known_key_dialect_lands_in_the_model(dialect, tiny, folders):
    import safetensors.torch as st
    from modules.loaders.lora.nodes import FunPackLoraLoader
    prefix, down, up = DIALECTS[dialect]
    layer, shape = _target(tiny)
    st.save_file({f"{prefix}{layer}.{down}": torch.randn(4, shape[1]), f"{prefix}{layer}.{up}": torch.randn(shape[0], 4)},
                 str(folders / "loras" / f"{dialect.replace(' ', '_')}.safetensors"))
    patched, _, status = FunPackLoraLoader.execute(tiny, f"{dialect.replace(' ', '_')}.safetensors", 0.7).result
    assert patched is not tiny and patched.patches, status
    assert "matched nothing" not in status


def test_a_lora_for_another_model_is_reported_not_silently_applied(tiny, folders):
    import safetensors.torch as st
    from modules.loaders.lora.nodes import FunPackLoraLoader
    st.save_file({"nothing.here.lora_A.weight": torch.randn(4, 8), "nothing.here.lora_B.weight": torch.randn(8, 4)},
                 str(folders / "loras" / "other.safetensors"))
    same, _, status = FunPackLoraLoader.execute(tiny, "other.safetensors", 1.0).result
    assert same is tiny and "matched nothing" in status


def test_negative_and_large_strengths_apply(tiny, folders):
    from modules.loaders.lora.nodes import FunPackLoraLoader
    for strength in (-1.0, 0.01, 5.0):
        patched, *_ = FunPackLoraLoader.execute(tiny, "diffusion_model.safetensors", strength).result
        assert patched.patches


def test_every_clip_family_name_is_accepted_or_refused_in_plain_words(folders, monkeypatch):
    import comfy.sd
    from modules.loaders.clip import nodes
    seen = []

    def fake_load_clip(ckpt_paths, embedding_directory=None, clip_type=None, model_options={}):
        seen.append((clip_type, len(ckpt_paths), model_options))
        return object()
    monkeypatch.setattr(comfy.sd, "load_clip", fake_load_clip)
    monkeypatch.setattr(nodes.folder_paths, "get_full_path_or_raise", lambda kind, name: f"/x/{name}.safetensors")
    monkeypatch.setattr(nodes.gguf_support, "has_gguf_magic", lambda p: False)
    for family in nodes.clip_types():
        out = nodes.FunPackCLIPLoader.execute("a", family, "default", clip_name2="b", clip_name3="c", clip_name4="d")
        assert f"type={family}" in out.result[1]
    assert len(seen) == len(nodes.clip_types()) and all(n == 4 for _, n, _ in seen)
    nodes.FunPackCLIPLoader.execute("a", "ltxv", "cpu")
    assert seen[-1][2]["load_device"] == torch.device("cpu")
    with pytest.raises(RuntimeError, match="no encoder family"):
        nodes.FunPackCLIPLoader.execute("a", "not_a_family")
    with pytest.raises(RuntimeError, match="could not|Could not"):
        monkeypatch.setattr(comfy.sd, "load_clip", lambda **k: None)
        nodes.FunPackCLIPLoader.execute("a", "ltxv")
