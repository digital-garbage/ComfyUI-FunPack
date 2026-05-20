import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeNestedTensor:
    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors

    @property
    def shape(self):
        return self.tensors[0].shape

    @property
    def device(self):
        return self.tensors[0].device

    @property
    def dtype(self):
        return self.tensors[0].dtype

    @property
    def layout(self):
        return self.tensors[0].layout

    def size(self):
        return self.tensors[0].size()


sample_calls = []


def _zeros_like(value):
    if getattr(value, "is_nested", False):
        return FakeNestedTensor([torch.zeros_like(t) for t in value.unbind()])
    return torch.zeros_like(value)


def _sample_like(value, mask, seed):
    if getattr(value, "is_nested", False):
        masks = mask.unbind() if getattr(mask, "is_nested", False) else [None] * len(value.unbind())
        return FakeNestedTensor([
            _sample_like(tensor, masks[index], seed)
            for index, tensor in enumerate(value.unbind())
        ])
    if mask is None:
        return value + float(seed)
    return value + mask.to(value.device, value.dtype) * float(seed)


def fake_prepare_noise(samples, seed, noise_inds=None):
    return _zeros_like(samples)


def fake_sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image,
                       noise_mask=None, callback=None, disable_pbar=False, seed=None):
    sample_calls.append({
        "seed": seed,
        "positive": positive,
        "negative": negative,
        "cfg": cfg,
        "latent_image": _sample_snapshot(latent_image),
        "noise_mask": _sample_snapshot(noise_mask),
    })
    return _sample_like(latent_image, noise_mask, seed)


def _sample_snapshot(value):
    if value is None:
        return None
    if getattr(value, "is_nested", False):
        return FakeNestedTensor([tensor.detach().clone() for tensor in value.unbind()])
    return value.detach().clone()


comfy_mod = types.ModuleType("comfy")
comfy_kd_mod = types.ModuleType("comfy.k_diffusion")
comfy_kd_sampling_mod = types.ModuleType("comfy.k_diffusion.sampling")
comfy_model_sampling_mod = types.ModuleType("comfy.model_sampling")
comfy_nested_mod = types.ModuleType("comfy.nested_tensor")
comfy_sample_mod = types.ModuleType("comfy.sample")
comfy_samplers_mod = types.ModuleType("comfy.samplers")
comfy_utils_mod = types.ModuleType("comfy.utils")

comfy_nested_mod.NestedTensor = FakeNestedTensor
comfy_sample_mod.prepare_noise = fake_prepare_noise
comfy_sample_mod.sample_custom = fake_sample_custom

comfy_mod.k_diffusion = comfy_kd_mod
comfy_kd_mod.sampling = comfy_kd_sampling_mod
comfy_mod.model_sampling = comfy_model_sampling_mod
comfy_mod.nested_tensor = comfy_nested_mod
comfy_mod.sample = comfy_sample_mod
comfy_mod.samplers = comfy_samplers_mod
comfy_mod.utils = comfy_utils_mod

sys.modules.setdefault("comfy", comfy_mod)
sys.modules.setdefault("comfy.k_diffusion", comfy_kd_mod)
sys.modules.setdefault("comfy.k_diffusion.sampling", comfy_kd_sampling_mod)
sys.modules.setdefault("comfy.model_sampling", comfy_model_sampling_mod)
sys.modules.setdefault("comfy.nested_tensor", comfy_nested_mod)
sys.modules.setdefault("comfy.sample", comfy_sample_mod)
sys.modules.setdefault("comfy.samplers", comfy_samplers_mod)
sys.modules.setdefault("comfy.utils", comfy_utils_mod)

from samplers import FunPackLTXAVSceneChainSampler


class FakeVAE:
    downscale_index_formula = (1, 1, 1)


def scene_cond(index):
    return (
        torch.ones(1, 2, 3) * float(index + 1),
        {"funpack_scene_text": f"scene {index + 1}"},
    )


def test_scene_chain_detects_scene_count_and_increments_seed():
    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()
    latent_template = {"samples": torch.zeros(1, 2, 5, 3, 3)}
    positive = [scene_cond(0), scene_cond(1), scene_cond(2)]
    negative = [(torch.zeros(1, 2, 3), {})]

    latent, status, scene_count, report = node.sample(
        model=object(),
        vae=FakeVAE(),
        positive=positive,
        negative=negative,
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        seed=10,
        latent_template=latent_template,
        num_frames_per_scene=5,
        frame_overlap=2,
        cfg=1.5,
        max_scenes=8,
    )

    assert scene_count == 3
    assert [call["seed"] for call in sample_calls] == [10, 11, 12]
    assert [call["positive"][0][1]["funpack_scene_text"] for call in sample_calls] == ["scene 1", "scene 2", "scene 3"]
    assert latent["samples"].shape[2] == 11
    assert "Scene chain complete" in status
    assert "Scene 3" in report


def test_scene_chain_preserves_nested_av_structure_and_audio_length():
    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()
    video = torch.zeros(1, 2, 5, 3, 3)
    audio = torch.zeros(1, 1, 10, 4)
    latent_template = {"samples": FakeNestedTensor([video, audio])}
    positive = [scene_cond(0), scene_cond(1)]

    latent, _, scene_count, _ = node.sample(
        model=object(),
        vae=FakeVAE(),
        positive=positive,
        negative=[],
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        seed=20,
        latent_template=latent_template,
        num_frames_per_scene=5,
        frame_overlap=2,
        cfg=1.0,
        max_scenes=8,
    )

    video_out, audio_out = latent["samples"].unbind()
    assert scene_count == 2
    assert video_out.shape[2] == 8
    assert audio_out.shape[2] == 16


def test_scene_chain_default_max_is_eight_but_allows_more():
    inputs = FunPackLTXAVSceneChainSampler.INPUT_TYPES()["required"]["max_scenes"][1]
    assert inputs["default"] == 8
    assert "max" not in inputs

    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()
    latent_template = {"samples": torch.zeros(1, 2, 3, 2, 2)}
    positive = [scene_cond(index) for index in range(10)]

    latent, status, scene_count, report = node.sample(
        model=object(),
        vae=FakeVAE(),
        positive=positive,
        negative=[],
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        seed=30,
        latent_template=latent_template,
        num_frames_per_scene=3,
        frame_overlap=0,
        cfg=1.0,
        max_scenes=10,
    )

    assert scene_count == 10
    assert len(sample_calls) == 10
    assert sample_calls[-1]["seed"] == 39
    assert latent["samples"].shape[2] == 30
    assert "10 scene(s)" in status
    assert "Scene 10" in report


def test_scene_chain_can_append_i2v_template_as_hidden_guide():
    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()
    samples = torch.zeros(1, 2, 5, 1, 1)
    samples[:, :, 0] = 7.0
    mask = torch.ones(1, 1, 5, 1, 1)
    mask[:, :, 0] = 0.0
    latent_template = {"samples": samples, "noise_mask": mask}
    positive = [scene_cond(0), scene_cond(1)]

    latent, status, scene_count, _ = node.sample(
        model=object(),
        vae=FakeVAE(),
        positive=positive,
        negative=[],
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        seed=40,
        latent_template=latent_template,
        num_frames_per_scene=5,
        frame_overlap=2,
        cfg=1.0,
        max_scenes=2,
        carry_i2v_guides=True,
    )

    second_call = sample_calls[1]
    assert scene_count == 2
    assert second_call["latent_image"].shape[2] == 6
    assert torch.all(second_call["latent_image"][:, :, 2:5] == 0.0)
    assert torch.all(second_call["latent_image"][:, :, 5] == 7.0)
    assert torch.all(second_call["noise_mask"][:, :, :2] == 0.0)
    assert torch.all(second_call["noise_mask"][:, :, 2:5] == 1.0)
    assert torch.all(second_call["noise_mask"][:, :, 5] == 0.0)
    assert "keyframe_idxs" in second_call["positive"][0][1]
    assert second_call["positive"][0][1]["guide_attention_entries"][0]["pre_filter_count"] == 1
    assert latent["samples"].shape[2] == 8
    assert "i2v guide tokens=1 latent frame(s)" in status


def test_scene_chain_expands_compact_i2v_guide_mask_to_spatial_chunk_mask():
    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()
    samples = torch.zeros(1, 2, 5, 24, 3)
    samples[:, :, 0] = 7.0
    mask = torch.ones(1, 1, 5, 1, 1)
    mask[:, :, 0] = 0.0
    latent_template = {"samples": samples, "noise_mask": mask}
    positive = [scene_cond(0), scene_cond(1)]

    _, status, scene_count, _ = node.sample(
        model=object(),
        vae=FakeVAE(),
        positive=positive,
        negative=[],
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        seed=45,
        latent_template=latent_template,
        num_frames_per_scene=5,
        frame_overlap=2,
        cfg=1.0,
        max_scenes=2,
        carry_i2v_guides=True,
    )

    second_call = sample_calls[1]
    assert scene_count == 2
    assert second_call["noise_mask"].shape == second_call["latent_image"].shape
    assert torch.all(second_call["noise_mask"][:, :, 5] == 0.0)
    assert "i2v guide tokens=1 latent frame(s)" in status


def test_scene_chain_does_not_carry_i2v_guides_by_default():
    inputs = FunPackLTXAVSceneChainSampler.INPUT_TYPES()["required"]["carry_i2v_guides"][1]
    assert inputs["default"] is False

    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()
    samples = torch.zeros(1, 2, 5, 1, 1)
    samples[:, :, 0] = 7.0
    mask = torch.ones(1, 1, 5, 1, 1)
    mask[:, :, 0] = 0.0
    latent_template = {"samples": samples, "noise_mask": mask}
    positive = [scene_cond(0), scene_cond(1)]

    _, status, scene_count, _ = node.sample(
        model=object(),
        vae=FakeVAE(),
        positive=positive,
        negative=[],
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        seed=50,
        latent_template=latent_template,
        num_frames_per_scene=5,
        frame_overlap=2,
        cfg=1.0,
        max_scenes=2,
    )

    second_call = sample_calls[1]
    assert scene_count == 2
    assert torch.all(second_call["latent_image"][:, :, 2] == 0.0)
    assert torch.all(second_call["noise_mask"][:, :, :2] == 0.0)
    assert torch.all(second_call["noise_mask"][:, :, 2:] == 1.0)
    assert "i2v guide" not in status
