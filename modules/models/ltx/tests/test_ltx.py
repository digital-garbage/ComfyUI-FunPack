"""The LTX module: recognising the model, the joint latent, the decode, and its pipelines."""

import asyncio
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def registered(comfyui):
    import nodes as comfy_nodes

    async def load():
        await comfy_nodes.init_extra_nodes(init_custom_nodes=False)
        await comfy_nodes.load_custom_node(str(Path(__file__).resolve().parents[4]), module_parent="custom_nodes")

    asyncio.run(load())
    return comfy_nodes


def _ltx():
    from modules.models import ltx
    return ltx


def test_the_audio_video_architecture_is_recognised_by_its_key_names_alone(comfyui):
    m = _ltx()
    assert m.detect({"audio_adaln_single.linear.weight"})
    assert m.detect({"model.diffusion_model.audio_adaln_single.linear.weight"})
    assert not m.detect({"adaln_single.linear.weight"})                      # video-only LTX
    assert not m.detect({"diffusion_model.audio_adaln_single.linear.weight"})  # not a prefix ComfyUI reads
    assert not m.detect({"video_patch_proj.weight", "audio_patch_proj.weight"})  # that is H3
    assert "audio_stream" in m.probe_traits({"audio_adaln_single.linear.weight"})
    assert m.probe_traits({"x"}) == []


def test_it_does_not_claim_block_hooks_it_has_no_modules_for(comfyui, monkeypatch):
    m = _ltx()
    monkeypatch.setattr(m, "has_block", lambda model, cls: True)
    assert "dit_block_hooks" not in m.traits(object())


@pytest.mark.parametrize("asked, got", [(1, 1), (9, 9), (10, 17), (121, 121), (122, 129)])
def test_a_length_off_the_8k_plus_1_grid_is_rounded_up(comfyui, asked, got):
    assert _ltx().frames_for(asked) == got


class _AudioVae:
    latent_channels = 8
    first_stage_model = types.SimpleNamespace(
        latent_frequency_bins=16, output_sample_rate=24000,
        num_of_latents_from_frames=lambda frames, rate: int(frames / rate * 25))

    def decode(self, latent):
        import torch
        return torch.zeros(latent.shape[0], 100, 2)


def test_the_latent_is_video_and_audio_sized_from_the_audio_vae(comfyui, monkeypatch):
    m = _ltx()
    monkeypatch.setattr(m, "is_ltx", lambda model: True)
    out = m.empty_latent(object(), width=768, height=512, length=121, batch_size=1,
                         audio_vae=_AudioVae(), frame_rate=25.0)
    video, audio = out["samples"].unbind()
    assert tuple(video.shape) == (1, 128, 16, 16, 24)
    assert tuple(audio.shape) == (1, 8, 121, 16)
    assert out["downscale_ratio_spacial"] == 32 and out["downscale_ratio_temporal"] == 8


def test_no_audio_vae_is_refused_not_made_silent(comfyui, monkeypatch):
    m = _ltx()
    monkeypatch.setattr(m, "is_ltx", lambda model: True)
    with pytest.raises(RuntimeError, match="audio VAE"):
        m.empty_latent(object(), 768, 512, 121, 1)


def test_another_model_is_not_claimed(comfyui, monkeypatch):
    m = _ltx()
    monkeypatch.setattr(m, "is_ltx", lambda model: False)
    assert m.empty_latent(object(), 768, 512, 121, 1) is None
    assert m.decode(object(), model=object()) is None


def test_decode_gives_a_picture_and_a_waveform(comfyui, monkeypatch):
    import torch
    from comfy.nested_tensor import NestedTensor
    m = _ltx()
    monkeypatch.setattr(m, "is_ltx", lambda model: True)

    class Vae:
        def decode(self, latent):
            return torch.zeros(1, 9, 64, 64, 3)

    latent = NestedTensor((torch.zeros(1, 128, 2, 2, 2), torch.zeros(1, 8, 5, 16)))
    images, audio = m.decode(latent, model=object(), vae=Vae(), audio_vae=_AudioVae())
    assert tuple(images.shape) == (9, 64, 64, 3)
    assert audio["sample_rate"] == 24000 and audio["waveform"].shape[0] == 1
    with pytest.raises(RuntimeError, match="audio VAE"):
        m.decode(latent, model=object(), vae=Vae(), audio_vae=None)


def test_the_empty_latent_node_passes_the_audio_vae_to_a_provider_that_asks_and_not_to_one_that_does_not(registered):
    from core import registry
    from modules.latent.empty.nodes import FunPackEmptyLatent
    seen = {}

    def wants(model, width, height, length, batch_size, audio_vae=None, frame_rate=25.0):
        seen["wants"] = (audio_vae, frame_rate)
        return {"samples": 1}

    def plain(model, width, height, length, batch_size):
        seen["plain"] = True
        return None

    class Reg:
        def providers(self, cap):
            return [(types.SimpleNamespace(id="x"), plain), (types.SimpleNamespace(id="y"), wants)]
    import unittest.mock as mock
    with mock.patch.object(registry, "current", lambda rescan=False: Reg()):
        FunPackEmptyLatent.execute(object(), 64, 64, 1, 1, audio_vae="A", frame_rate=30.0)
    assert seen == {"plain": True, "wants": ("A", 30.0)}


@pytest.mark.parametrize("pid", ["ltx23_text_to_video", "ltx23_image_to_video", "ltx23_anchor_guide"])
def test_the_pipelines_build_a_real_graph(registered, pid):
    from core import graph
    from modules.models.ltx.pipeline import presets

    slots = next(p["slots"] for p in presets() if p["id"] == pid)
    ids = [s["id"] for s in slots]
    assert len(ids) == len(set(ids))
    for slot in slots:
        for value in slot["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in ids
    _prompt, problems = graph.build(slots)
    unset = ("needs 'ckpt_name'", "needs 'text_encoder'", "is '', which is not one of")   # a file nobody has picked yet
    structural = [p for p in problems if not any(u in p for u in unset)]
    assert structural == [], structural


def test_the_presets_are_offered_through_the_registry(registered):
    from core import routes
    found = {preset["id"] for _spec, make in routes.modules().providers("pipeline_presets") for preset in make()}
    assert {"ltx23_text_to_video", "ltx23_image_to_video", "ltx23_anchor_guide"} <= found


def test_the_picture_is_found_in_both_shapes_a_sampling_latent_takes(comfyui):
    import torch
    from comfy.nested_tensor import NestedTensor
    m = _ltx()
    video, audio = torch.randn(1, 128, 2, 2, 3), torch.randn(1, 8, 5, 16)
    found, rebuild = m.PROVIDES["video_stream"](NestedTensor((video, audio)))
    assert torch.equal(found, video)
    packed = torch.cat([video.reshape(1, 1, -1), audio.reshape(1, 1, -1)], dim=-1)
    found, rebuild = m.PROVIDES["video_stream"](packed, {"latent_shapes": [video.shape, audio.shape]})
    assert torch.equal(found, video)
    assert torch.equal(rebuild(found * 2)[..., video.numel():], packed[..., video.numel():])
    assert m.PROVIDES["video_stream"](torch.randn(1, 4, 8, 8)) is None


def test_a_tiled_decode_is_asked_the_way_comfy_asks_it(comfyui, monkeypatch):
    import torch
    from comfy.nested_tensor import NestedTensor
    m = _ltx()
    monkeypatch.setattr(m, "is_ltx", lambda model: True)
    seen = {}

    class Vae:
        def temporal_compression_decode(self): return 8
        def spacial_compression_decode(self): return 32
        def decode_tiled(self, latent, **kw):
            seen.update(kw)
            return torch.zeros(1, 9, 64, 64, 3)

    latent = NestedTensor((torch.zeros(1, 128, 2, 2, 2), torch.zeros(1, 8, 5, 16)))
    m.decode(latent, model=object(), vae=Vae(), audio_vae=_AudioVae(), tile_size=512)
    assert seen == {"tile_x": 16, "tile_y": 16, "overlap": 4, "tile_t": 8, "overlap_t": 1}


def test_a_size_that_is_not_a_multiple_of_32_is_said(comfyui, monkeypatch):
    import torch
    from core import log
    m = _ltx()
    monkeypatch.setattr(m, "is_ltx", lambda model: True)
    said = []
    monkeypatch.setattr(log, "once", lambda key, level, source, msg: said.append(msg))
    m.empty_latent(object(), 816, 512, 9, 1, audio_vae=_AudioVae(), frame_rate=25.0)
    assert said and "800x512" in said[0]
