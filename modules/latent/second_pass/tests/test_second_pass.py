"""Second pass nodes: loading H3's upscaler, sharpen/upscale, and pins that still fit."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def _small_resizer():
    from modules.models.minimax_h3.latent_upscaler import H3LatentResizer3D
    torch.manual_seed(0)
    return H3LatentResizer3D(channels=32, in_blocks=1, out_blocks=1).eval().requires_grad_(False)


@pytest.fixture
def upscaler_dir(tmp_path, monkeypatch):
    import folder_paths
    from safetensors.torch import save_file
    save_file(_small_resizer().state_dict(), str(tmp_path / "h3_up.safetensors"))
    save_file({"conv_in.weight": torch.zeros(2, 2)}, str(tmp_path / "other.safetensors"))
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "latent_upscale_models",
                        ([str(tmp_path)], {".safetensors"}))
    folder_paths.filename_list_cache.pop("latent_upscale_models", None)
    return tmp_path


def test_the_loader_builds_h3s_upscaler_from_its_weights(upscaler_dir):
    from modules.latent.second_pass.nodes import FunPackLatentUpscalerLoader
    model = FunPackLatentUpscalerLoader.execute("h3_up.safetensors").result[0]
    assert callable(model.funpack_latent_upscale)


def test_an_unknown_file_is_refused_with_where_to_go(upscaler_dir):
    from modules.latent.second_pass.nodes import FunPackLatentUpscalerLoader
    with pytest.raises(RuntimeError, match="Load Latent Upscale Model"):
        FunPackLatentUpscalerLoader.execute("other.safetensors")


def _nested(h=4, w=4):
    from comfy.nested_tensor import NestedTensor
    torch.manual_seed(1)
    return NestedTensor((torch.randn(1, 24, 2, h, w), torch.randn(1, 32, 2, 3)))


def _pinned(h=4, w=4):
    z = torch.randn(1, 24, 1, h, w)
    return [[torch.zeros(1, 5, 48), {"minimax_keyframes": [{"resolved_frame_index": 0, "latent": z}]}]]


def test_sharpen_keeps_the_size_changes_the_video_and_leaves_audio(comfyui):
    from modules.latent.second_pass.nodes import FunPackLatentResample
    samples = _nested()
    pos = _pinned()
    out, p, _n, status = FunPackLatentResample.execute(
        {"samples": samples, "noise_mask": torch.ones(1)}, _small_resizer(), "sharpen", 2.0,
        positive=pos).result
    video, audio = out["samples"].unbind()
    assert video.shape == samples.unbind()[0].shape
    assert not torch.allclose(video, samples.unbind()[0])
    assert torch.equal(audio, samples.unbind()[1])
    assert p is pos and "noise_mask" in out and "sharpen" in status


def test_upscale_doubles_the_grid_and_resizes_the_pin(comfyui):
    from modules.latent.second_pass.nodes import FunPackLatentResample
    out, pos, _n, status = FunPackLatentResample.execute(
        {"samples": _nested()}, _small_resizer(), "upscale", 2.0, positive=_pinned()).result
    assert tuple(out["samples"].unbind()[0].shape[-2:]) == (8, 8)
    assert tuple(pos[0][1]["minimax_keyframes"][0]["latent"].shape[-2:]) == (8, 8)
    assert "1 anchor pin(s) resized" in status


def test_a_resized_pin_fits_a_real_h3_forward_and_a_stale_one_does_not(tiny_h3):
    """The reason rescaling exists: H3 refuses a pin sized to the old grid."""
    from modules.models.minimax_h3.anchor import rescale_pins
    dm = tiny_h3.patcher.model.diffusion_model
    video = torch.randn(1, 24, 2, 8, 8)
    stale = _pinned(4, 4)
    fresh, n = rescale_pins(stale, 8, 8)
    assert n == 1

    def fwd(cond):
        kf = cond[0][1]["minimax_keyframes"]
        payload = {"keyframes": kf, "cond_video_latents": [k["latent"] for k in kf]}
        with torch.inference_mode():
            return dm([video, tiny_h3.audio], torch.tensor([500.0]), tiny_h3.context,
                      transformer_options={}, minimax_payload=payload)

    with pytest.raises(RuntimeError, match="shape mismatch"):
        fwd(stale)
    assert fwd(fresh)[0].shape == video.shape


def test_conditioning_no_model_module_recognises_passes_through():
    from modules.latent.second_pass.nodes import rescale
    plain = [[torch.zeros(1, 3, 8), {}]]
    assert rescale(plain, 8, 8) == (plain, 0)
