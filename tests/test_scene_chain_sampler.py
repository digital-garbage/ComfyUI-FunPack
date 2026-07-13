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

    def decode(self, samples):
        b, _c, t, _h, _w = samples.shape
        return torch.zeros(b, t, 8, 8, 3)


class FakeModel:
    """Minimal stand-in with the one attribute the per-scene wrapper snapshot/restore
    (samplers.py, around _scene_base_wrapper) needs. Plain object() lacks model_options
    entirely, which pre-dates this feature (every sample()-calling test in this file
    currently fails on that AttributeError — a known stale-mock gap, not something this
    change introduces or fixes file-wide)."""
    def __init__(self):
        self.model_options = {}


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

    latent, _images, status, scene_count, report, _boundaries = node.sample(
        model=FakeModel(),
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


def test_scene_chain_accepts_manual_combined_conditioning_without_metadata():
    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()
    latent_template = {"samples": torch.zeros(1, 2, 5, 3, 3)}
    positive = [
        (torch.ones(1, 2, 3), {}),
        (torch.ones(1, 2, 3) * 2.0, {}),
        (torch.ones(1, 2, 3) * 3.0, {}),
    ]

    _, _images, status, scene_count, report, _boundaries = node.sample(
        model=FakeModel(),
        vae=FakeVAE(),
        positive=positive,
        negative=[],
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        seed=70,
        latent_template=latent_template,
        num_frames_per_scene=5,
        frame_overlap=2,
        cfg=1.0,
        max_scenes=8,
    )

    assert scene_count == 3
    assert [call["seed"] for call in sample_calls] == [70, 71, 72]
    assert [call["positive"][0][0].mean().item() for call in sample_calls] == [1.0, 2.0, 3.0]
    assert "3 scene(s)" in status
    # Per-scene report lines now include a "sampling {s}s" timing segment between seed and text.
    assert "Scene 1: seed=70," in report and "text=Scene 1" in report
    assert "Scene 3: seed=72," in report and "text=Scene 3" in report


def test_scene_chain_uses_scene_seed_metadata():
    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()
    latent_template = {"samples": torch.zeros(1, 2, 5, 3, 3)}
    positive = [
        (torch.ones(1, 2, 3), {"funpack_scene_text": "scene a", "funpack_scene_seed": 101}),
        (torch.ones(1, 2, 3), {"funpack_scene_text": "scene b", "funpack_scene_seed": 202}),
        (torch.ones(1, 2, 3), {"funpack_scene_text": "scene c", "funpack_scene_seed": 303}),
    ]

    _, _images, _status, scene_count, report, _boundaries = node.sample(
        model=FakeModel(),
        vae=FakeVAE(),
        positive=positive,
        negative=[],
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        seed=10,
        latent_template=latent_template,
        num_frames_per_scene=5,
        frame_overlap=2,
        cfg=1.0,
        max_scenes=8,
    )

    assert scene_count == 3
    assert [call["seed"] for call in sample_calls] == [101, 202, 303]
    assert "seed=202" in report


def test_scene_chain_use_same_seed_forces_first_seed():
    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()
    latent_template = {"samples": torch.zeros(1, 2, 5, 3, 3)}
    positive = [
        (torch.ones(1, 2, 3), {"funpack_scene_text": "scene a", "funpack_scene_seed": 101}),
        (torch.ones(1, 2, 3), {"funpack_scene_text": "scene b", "funpack_scene_seed": 202}),
        (torch.ones(1, 2, 3), {"funpack_scene_text": "scene c"}),
    ]

    _, _images, _status, scene_count, _report, _boundaries = node.sample(
        model=FakeModel(),
        vae=FakeVAE(),
        positive=positive,
        negative=[],
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        seed=10,
        latent_template=latent_template,
        num_frames_per_scene=5,
        frame_overlap=2,
        cfg=1.0,
        max_scenes=8,
        use_same_seed=True,
    )

    assert scene_count == 3
    assert [call["seed"] for call in sample_calls] == [101, 101, 101]


def test_scene_chain_preserves_nested_av_structure_and_audio_length():
    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()
    video = torch.zeros(1, 2, 5, 3, 3)
    audio = torch.zeros(1, 1, 10, 4)
    latent_template = {"samples": FakeNestedTensor([video, audio])}
    positive = [scene_cond(0), scene_cond(1)]

    latent, _images, _status, scene_count, _report, _boundaries = node.sample(
        model=FakeModel(),
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

    latent, _images, status, scene_count, report, _boundaries = node.sample(
        model=FakeModel(),
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

    latent, _images, status, scene_count, _report, _boundaries = node.sample(
        model=FakeModel(),
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

    # _append_i2v_guides prepends the guide frame (temporal pos 0) so it never collides with
    # the overlap frames that follow: [guide, overlap0, overlap1, mid, mid, mid].
    second_call = sample_calls[1]
    assert scene_count == 2
    assert second_call["latent_image"].shape[2] == 6
    assert torch.all(second_call["latent_image"][:, :, 0] == 7.0)
    assert torch.all(second_call["latent_image"][:, :, 3:6] == 0.0)
    assert torch.all(second_call["noise_mask"][:, :, :3] == 0.0)
    assert torch.all(second_call["noise_mask"][:, :, 3:6] == 1.0)
    # _append_i2v_guides is a plain protected-frame append (tensor + mask only) — unlike
    # mid_scene_guide/joyai memory it does not add keyframe_idxs/guide_attention_entries.
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

    _, _images, status, scene_count, _report, _boundaries = node.sample(
        model=FakeModel(),
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

    # Guide frame is prepended (temporal pos 0) — see test_scene_chain_can_append_i2v_template_as_hidden_guide.
    second_call = sample_calls[1]
    assert scene_count == 2
    assert second_call["noise_mask"].shape == second_call["latent_image"].shape
    assert torch.all(second_call["noise_mask"][:, :, 0] == 0.0)
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

    _, _images, status, scene_count, _report, _boundaries = node.sample(
        model=FakeModel(),
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


def test_overlap_diagnostics_report_latent_blend_zone():
    import json

    node = FunPackLTXAVSceneChainSampler()
    diag = node._build_overlap_diagnostics(
        scene_count=2,
        video_frames=13,
        num_frames_per_scene=97,
        pixel_overlap=16,
        latent_overlap=2,
        time_scale=8,
        transition_duration=16,
        boundaries=[{
            "between": [1, 2],
            "boundary_latent": 12,
            "pixel_frame": 89,
            "effect": "crossfade",
        }],
        scene_runs=[
            {"index": 1, "text": "hero walks", "encode_text": "hero walks", "mechanisms": []},
            {"index": 2, "text": "hero runs", "encode_text": "hero runs", "mechanisms": ["latent_overlap(16px)"]},
        ],
        carry_i2v_guides=True,
        mid_scene_guide=False,
        embed_guidance=True,
        embed_guidance_strength=0.15,
        embed_guidance_source="absolute",
    )
    assert diag["pixel_overlap"] == 16
    blend = diag["boundaries"][0]["contamination_zones"]["latent_blend"]
    assert blend["scene_prev_tail"] == [73, 88]
    assert blend["scene_next_head"] == [89, 104]
    assert diag["scenes"][0]["whole_scene_steering"] is True
    assert any(g["mechanism"] == "embed_guidance" for g in diag["global_steering"])
    # scene_boundaries output is JSON in the full sample() path
    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()
    positive = [scene_cond(0), scene_cond(1)]
    latent_template = {"samples": torch.zeros(1, 2, 5, 1, 1)}
    _, _, status, _, _, boundaries_json = node.sample(
        model=FakeModel(),
        vae=FakeVAE(),
        positive=positive,
        negative=[],
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        seed=60,
        latent_template=latent_template,
        num_frames_per_scene=5,
        frame_overlap=2,
        cfg=1.0,
        max_scenes=2,
        embed_guidance=False,
    )
    parsed = json.loads(boundaries_json)
    assert parsed["scene_count"] == 2
    assert "boundaries" in parsed
    assert "overlap_blend=2px" in status


def test_mixed_solo_applies_guides_on_first_chunk():
    import json

    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()
    guides = json.dumps({
        "stack_enabled": True,
        "accumulate_prior": False,
        "scenes": [[{"enabled": True, "source": "template", "frame_idx": 0, "apply_at": 0, "strength": 0.35}]],
    })
    latent_template = {
        "samples": torch.zeros(1, 2, 5, 3, 3),
        "noise_mask": torch.cat([torch.zeros(1, 1, 2, 1, 1), torch.ones(1, 1, 3, 1, 1)], dim=2),
    }
    positive = [scene_cond(0)]

    latent, _images, status, scene_count, _report, boundaries_json = node.sample(
        model=FakeModel(),
        vae=FakeVAE(),
        positive=positive,
        negative=[],
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        seed=90,
        latent_template=latent_template,
        num_frames_per_scene=5,
        frame_overlap=0,
        cfg=1.0,
        max_scenes=1,
        funpack_scene_guides=guides,
        prompt={"n1": {"inputs": {}}},
        unique_id="test-node",
    )

    assert scene_count == 1
    chunk = sample_calls[0]["latent_image"]
    frames = chunk["samples"].shape[2] if isinstance(chunk, dict) else chunk.shape[2]
    assert frames == 7
    parsed = json.loads(boundaries_json)
    assert "custom_guide_stack" in parsed["scenes"][0]["mechanisms"]


def test_mixed_anchor_skips_frame_overlap():
    import json

    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()
    node._load_image_tensor = lambda _fn: torch.ones(1, 8, 8, 3)
    node._apply_img2video_to_video_latent = lambda _vae, _img, chunk, _strength: node._clone_latent(chunk)

    positive = [scene_cond(0), scene_cond(1)]
    latent_template = {"samples": torch.zeros(1, 2, 5, 3, 3)}
    anchors = json.dumps({"1": {"filename": "anchor.png", "strength": 1.0}})

    latent, _images, status, scene_count, _report, boundaries_json = node.sample(
        model=FakeModel(),
        vae=FakeVAE(),
        positive=positive,
        negative=[],
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        seed=80,
        latent_template=latent_template,
        num_frames_per_scene=5,
        frame_overlap=2,
        cfg=1.0,
        max_scenes=2,
        funpack_scene_anchors=anchors,
        prompt={"n1": {"inputs": {}}},
        unique_id="test-node",
    )

    assert scene_count == 2
    assert latent["samples"].shape[2] == 10
    parsed = json.loads(boundaries_json)
    mechs = parsed["scenes"][1]["mechanisms"]
    assert "mixed_i2v_anchor" in mechs
    assert not any("latent_overlap" in m for m in mechs)


def test_mixed_anchor_carries_overlap_when_enabled():
    """carry_overlap_through_anchor=True: the chunk fed to scene 1's sampler call is seeded
    with scene 0's tail (frame_overlap latent frames) instead of a bare template, even though
    scene 1 has its own i2v anchor. The mocked _apply_img2video_to_video_latent is an identity
    passthrough here, so whatever _build_mixed_anchor_chunk hands it is exactly what gets
    sampled — letting us assert on the carried values and protected mask directly."""
    import json

    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()
    node._load_image_tensor = lambda _fn: torch.ones(1, 8, 8, 3)
    node._apply_img2video_to_video_latent = lambda _vae, _img, chunk, _strength: node._clone_latent(chunk)

    positive = [scene_cond(0), scene_cond(1)]
    latent_template = {"samples": torch.zeros(1, 2, 5, 3, 3)}
    anchors = json.dumps({"1": {"filename": "anchor.png", "strength": 1.0}})

    latent, _images, status, scene_count, _report, boundaries_json = node.sample(
        model=FakeModel(),
        vae=FakeVAE(),
        positive=positive,
        negative=[],
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        seed=80,
        latent_template=latent_template,
        num_frames_per_scene=5,
        frame_overlap=2,
        cfg=1.0,
        max_scenes=2,
        funpack_scene_anchors=anchors,
        carry_overlap_through_anchor=True,
        prompt={"n1": {"inputs": {}}},
        unique_id="test-node",
    )

    assert scene_count == 2
    assert latent["samples"].shape[2] == 10
    parsed = json.loads(boundaries_json)
    mechs = parsed["scenes"][1]["mechanisms"]
    assert "mixed_i2v_anchor" in mechs
    assert "latent_overlap_through_anchor(2px)" in mechs

    # Scene 0 (no anchor) samples with mask=None, so _sample_like adds its seed (80) uniformly:
    # scene 0's full output is all 80s. Scene 1's chunk should carry the last 2 frames of that
    # (80, 80) into its own leading frames, protected (mask=0), template's remaining 3 frames
    # untouched (0) and free to denoise (mask=1).
    scene1_chunk = sample_calls[1]["latent_image"]
    scene1_mask = sample_calls[1]["noise_mask"]
    samples = scene1_chunk["samples"] if isinstance(scene1_chunk, dict) else scene1_chunk
    mask = scene1_mask["samples"] if isinstance(scene1_mask, dict) else scene1_mask
    assert torch.allclose(samples[:, :, :2], torch.full_like(samples[:, :, :2], 80.0))
    assert torch.allclose(samples[:, :, 2:], torch.zeros_like(samples[:, :, 2:]))
    assert torch.allclose(mask[:, :, :2], torch.zeros_like(mask[:, :, :2]))
    assert torch.allclose(mask[:, :, 2:], torch.ones_like(mask[:, :, 2:]))


def test_mixed_anchor_resolves_identity_pin_when_configured():
    """The mixed_i2v_anchor branch skips _apply_configured_guides entirely, so without the
    explicit lookup an identity_pin guide configured for the anchor scene would never resolve
    and Best-FaceID identity_transfer could never fire on an anchor-swap scene."""
    import json

    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()
    node._load_image_tensor = lambda _fn: torch.ones(1, 8, 8, 3)
    node._apply_img2video_to_video_latent = lambda _vae, _img, chunk, _strength: node._clone_latent(chunk)

    positive = [scene_cond(0), scene_cond(1)]
    latent_template = {"samples": torch.zeros(1, 2, 5, 3, 3)}
    anchors = json.dumps({"1": {"filename": "anchor.png", "strength": 1.0}})
    guides = json.dumps({
        "stack_enabled": True,
        "scenes": [
            [],
            [{"enabled": True, "source": "image", "media_ref": "pin", "identity_pin": True, "strength": 0.35}],
        ],
    })
    media_refs = json.dumps({"pin": "pin.png"})

    _latent, _images, _status, scene_count, _report, boundaries_json = node.sample(
        model=FakeModel(),
        vae=FakeVAE(),
        positive=positive,
        negative=[],
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        seed=80,
        latent_template=latent_template,
        num_frames_per_scene=5,
        frame_overlap=2,
        cfg=1.0,
        max_scenes=2,
        funpack_scene_anchors=anchors,
        funpack_scene_guides=guides,
        funpack_scene_media_refs=media_refs,
        identity_transfer_enabled=True,
        prompt={"n1": {"inputs": {}}},
        unique_id="test-node",
    )

    assert scene_count == 2
    parsed = json.loads(boundaries_json)
    mechs = parsed["scenes"][1]["mechanisms"]
    assert "identity_pin_on_anchor_scene" in mechs


def test_plateau_cache_reuses_forward_on_noise_plateau_and_recomputes_below():
    """On the near-noise plateau (sigma >= threshold) the base-model forward is computed once
    and reused; once sigma drops below threshold every step recomputes."""
    node = FunPackLTXAVSceneChainSampler()
    model = FakeModel()
    stats = node._build_plateau_cache_wrapper(model, 0.975)
    wrapper = model.model_options["model_function_wrapper"]

    calls = {"n": 0}
    x = torch.zeros(1, 4, 2, 2, 2)

    def apply_fn(inp, ts, **c):
        calls["n"] += 1
        return torch.full_like(inp, float(calls["n"]))

    def run(sig):
        args = {"input": x, "timestep": torch.tensor([sig]), "cond_or_uncond": [0], "c": {}}
        return wrapper(apply_fn, args)

    # Default 8-step distilled schedule: sigmas 1.0..0.975 are the plateau, 0.909 onward is not.
    sigmas = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]
    outs = [run(s) for s in sigmas]

    # Plateau: 1 real forward, 4 reuses. Below threshold: 3 real forwards. Total 4 forwards vs 8.
    assert stats["computed"] == 1
    assert stats["reused"] == 4
    assert calls["n"] == 4
    for o in outs[:5]:
        assert torch.allclose(o, torch.ones_like(x))  # all reuse the first plateau output
    assert torch.allclose(outs[5], torch.full_like(x, 2.0))  # structure steps recompute fresh
    assert torch.allclose(outs[7], torch.full_like(x, 4.0))


def test_plateau_cache_keys_cond_and_uncond_separately():
    """A CFG>1 split cond/uncond pair must each get its own cache slot, not thrash one."""
    node = FunPackLTXAVSceneChainSampler()
    model = FakeModel()
    stats = node._build_plateau_cache_wrapper(model, 0.975)
    wrapper = model.model_options["model_function_wrapper"]

    calls = {"n": 0}
    x = torch.zeros(1, 4, 2, 2, 2)

    def apply_fn(inp, ts, **c):
        calls["n"] += 1
        return torch.full_like(inp, float(calls["n"]))

    def run(co):
        args = {"input": x, "timestep": torch.tensor([1.0]), "cond_or_uncond": co, "c": {}}
        return wrapper(apply_fn, args)

    run([0]); run([1]); run([0]); run([1])
    assert stats["computed"] == 2  # cond + uncond each computed once
    assert stats["reused"] == 2    # then each reused once
    assert calls["n"] == 2
