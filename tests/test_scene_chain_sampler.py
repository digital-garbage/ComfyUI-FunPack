import sys
import types
from pathlib import Path

import pytest
import torch

import _comfy_stubs

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
        "steps": max(0, int(sigmas.numel()) - 1),
        # What the sampler is telling the UI it's doing, AS this call runs — the point of
        # the label is that it is live, so reading it afterwards would prove nothing.
        "phase": run_phase.current()["label"],
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


class FakeProgressBar:
    """Records what the sampler declared as the run's total step count."""
    last = None

    def __init__(self, total):
        self.total = int(total)
        self.value = 0
        FakeProgressBar.last = self

    def update_absolute(self, value, total=None, preview=None):
        self.value = int(value)


# These fakes ARE the test: a full sample() run is driven through them. They attach to the
# shared comfy stubs conftest built rather than registering modules of their own — the old
# `sys.modules.setdefault(...)` lost the race whenever another test module imported first,
# and every test here then died on a missing attribute in a full-suite run while passing
# alone. Only this file and test_minimax_h3_sampler.py touch the sampling entry points, and
# that one installs inert lambdas, so owning them for the session is safe.
_comfy_stubs.install_module("comfy.utils", ProgressBar=FakeProgressBar)
_comfy_stubs.install_module("comfy.nested_tensor", NestedTensor=FakeNestedTensor)
_comfy_stubs.install_module("comfy.sample",
                        prepare_noise=fake_prepare_noise, sample_custom=fake_sample_custom)


@pytest.fixture(autouse=True)
def _own_comfy_stubs(monkeypatch):
    """Re-pin this file's fakes around every test.

    Module-level installation is not enough: several other test modules assign
    `comfy.nested_tensor.NestedTensor = object` on the same shared module at COLLECTION
    time, so whichever is collected last wins and a run() here builds `object()` latents.
    Re-pinning per test makes these assertions about the sampler instead of about
    collection order."""
    monkeypatch.setattr(sys.modules["comfy.nested_tensor"], "NestedTensor",
                        FakeNestedTensor, raising=False)
    monkeypatch.setattr(sys.modules["comfy.utils"], "ProgressBar",
                        FakeProgressBar, raising=False)
    monkeypatch.setattr(sys.modules["comfy.sample"], "prepare_noise",
                        fake_prepare_noise, raising=False)
    monkeypatch.setattr(sys.modules["comfy.sample"], "sample_custom",
                        fake_sample_custom, raising=False)


import run_phase  # noqa: E402
from samplers import FunPackLTXAVSceneChainSampler  # noqa: E402


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


def test_install_v2a_scale_rolls_back_hooks_already_attached_when_a_later_one_fails():
    """register_forward_hook is called once per transformer block; if one raises partway
    through the loop, the hooks already attached to EARLIER blocks must not be stranded --
    the local `handles` list holding the only references able to remove them would
    otherwise never reach the caller (the loop itself raises out of _install_v2a_scale).
    Each live hook multiplies that block's video_to_audio_attn output by `scale`
    indefinitely, on a model this scene never actually owns exclusively."""
    node = FunPackLTXAVSceneChainSampler()
    removed = []

    class _Sub:
        def __init__(self, index, boom=False):
            self.index = index
            self.boom = boom

        def register_forward_hook(self, hook):
            if self.boom:
                raise RuntimeError(f"block {self.index} refused the hook")
            handle = types.SimpleNamespace(remove=lambda: removed.append(self.index))
            return handle

    class _Block:
        def __init__(self, sub):
            self.video_to_audio_attn = sub

    blocks = [_Block(_Sub(0)), _Block(_Sub(1)), _Block(_Sub(2, boom=True)), _Block(_Sub(3))]
    model = types.SimpleNamespace(model=types.SimpleNamespace(
        diffusion_model=types.SimpleNamespace(transformer_blocks=blocks)))

    with pytest.raises(RuntimeError, match="block 2 refused the hook"):
        node._install_v2a_scale(model, 1.5)

    # Blocks 0 and 1's hooks were attached before block 2 raised -- both must have been
    # rolled back, not left live with no reference to remove them by. Block 3 is never
    # reached because the loop already raised.
    assert removed == [0, 1]


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






# ── second pass: the progress bar and the phase readout ─────────────────────

@pytest.fixture
def live_stubs():
    """Force this file's comfy fakes onto whatever module objects are actually live.

    The module-level `sys.modules.setdefault` above only wins when this file is imported
    FIRST — under the full suite another test module has usually registered bare comfy
    stubs already, so the fakes here are silently discarded and every sample()-calling test
    in this file fails on the missing attribute. Snapshot/restore rather than leaving them
    behind, so the next module's expectations are not the ones broken instead."""
    targets = [
        (sys.modules["comfy.sample"], "prepare_noise", fake_prepare_noise),
        (sys.modules["comfy.sample"], "sample_custom", fake_sample_custom),
        (sys.modules["comfy.utils"], "ProgressBar", FakeProgressBar),
        (sys.modules["comfy.nested_tensor"], "NestedTensor", FakeNestedTensor),
        (sys.modules["comfy"], "sample", sys.modules["comfy.sample"]),
        (sys.modules["comfy"], "utils", sys.modules["comfy.utils"]),
        (sys.modules["comfy"], "nested_tensor", sys.modules["comfy.nested_tensor"]),
    ]
    saved = [(mod, name, getattr(mod, name, _MISSING)) for mod, name, _ in targets]
    for mod, name, value in targets:
        setattr(mod, name, value)
    yield
    for mod, name, value in saved:
        if value is _MISSING:
            delattr(mod, name)
        else:
            setattr(mod, name, value)


_MISSING = object()


def _second_pass_run(scene_count=2, main=torch.tensor([1.0, 0.5, 0.0]),
                     second=torch.tensor([0.4, 0.2, 0.0])):
    sample_calls.clear()
    run_phase.clear()
    node = FunPackLTXAVSceneChainSampler()
    return node.sample(
        model=FakeModel(),
        vae=FakeVAE(),
        positive=[scene_cond(i) for i in range(scene_count)],
        negative=[(torch.zeros(1, 2, 3), {})],
        sampler=object(),
        sigmas=main,
        seed=10,
        latent_template={"samples": torch.zeros(1, 2, 5, 3, 3)},
        num_frames_per_scene=5,
        frame_overlap=0,
        cfg=1.5,
        max_scenes=8,
        second_pass=True,
        second_pass_sigmas=second,
    )


def test_second_pass_progress_total_counts_both_passes(live_stubs):
    """A second pass samples every scene twice. If its steps aren't in the total, the bar
    overflows during scene 1 and then jumps BACKWARDS at scene 2, whose offset is a
    multiple of the (too small) per-scene stride."""
    _second_pass_run(scene_count=2)
    # 2 main steps + 2 second-pass steps, per scene, over 2 scenes.
    assert [c["steps"] for c in sample_calls] == [2, 2, 2, 2]
    assert FakeProgressBar.last.total == 2 * (2 + 2)


def test_progress_total_ignores_an_unusable_second_pass_schedule(live_stubs):
    """The loop skips a malformed schedule with a note, so the total must not reserve steps
    for a pass that never runs — the bar would stall short of the end."""
    _second_pass_run(scene_count=2, second=torch.tensor([0.4, 0.2]))  # doesn't reach 0
    assert [c["steps"] for c in sample_calls] == [2, 2]
    assert FakeProgressBar.last.total == 2 * 2


def test_second_pass_says_which_pass_is_running(live_stubs):
    """The run report names the second pass only after the fact, which is no help while
    you're waiting on one. The live label has to say it AS it happens."""
    _second_pass_run(scene_count=2)
    assert [c["phase"] for c in sample_calls] == [
        "scene 1/2 · pass 1 of 2", "scene 1/2 · pass 2 of 2",
        "scene 2/2 · pass 1 of 2", "scene 2/2 · pass 2 of 2",
    ]
    # ...and nothing is sampling once the node returns.
    assert run_phase.current()["label"] == ""


def test_phase_label_drops_the_scene_number_on_a_single_scene_run(live_stubs):
    _second_pass_run(scene_count=1)
    assert [c["phase"] for c in sample_calls] == ["pass 1 of 2", "pass 2 of 2"]


def test_explore_first_step_candidates_never_reach_the_scenes_own_guidance_wrapper(monkeypatch):
    """explore_first_step's candidate calls must be scored on the model exactly as it was
    BEFORE this scene's own guidance stack goes on, never on a wrapper installed for the
    real run (context_windows / embed_guidance / score_slider / dynashift /
    output_guidance / trajectory_guidance / the trajectory-probe recorder / temporal
    styles / v2a / identity_overlap). A first version of this feature called
    _select_best_seed right before _sample_chunk with no reset, so candidates were
    silently scored on post-guidance predictions, and a live trajectory-probe recorder
    would have banked a discarded candidate's step-1 prediction as the scene's real one
    (its dedup-by-sigma keeps only the first call it sees at a given sigma).

    A prior version of this test only checked `model.model_options[...] is not sentinel`
    at the top of each sample_custom call -- but _select_best_seed ALWAYS installs its own
    fresh `_observe_wrapper` closure there (never the raw sentinel object), so that
    assertion passed identically whether the underlying old_wrapper it chains to was None
    or the sentinel. This version instead makes the sentinel itself count its own
    invocations, so it can tell whether a candidate's call chain actually reached it."""
    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()

    sentinel_calls = []

    def _sentinel(apply_fn, args):
        sentinel_calls.append(1)
        return apply_fn(args["input"], args["timestep"], **args.get("c", {}))

    def fake_install_context_windows(self, model, length, overlap, schedule, fuse,
                                     freenoise, retain_first):
        model.model_options["model_function_wrapper"] = _sentinel
        return (lambda: model.model_options.pop("model_function_wrapper", None), 10**9, None)

    monkeypatch.setattr(FunPackLTXAVSceneChainSampler, "_install_context_windows",
                        fake_install_context_windows, raising=True)

    class _ReadyValueFn:
        def is_ready(self):
            return True

        def compress(self, x):
            return x

        def forward(self, x):
            return x.mean()

    monkeypatch.setattr(FunPackLTXAVSceneChainSampler, "_load_output_value_function",
                        lambda self, key: _ReadyValueFn(), raising=True)

    sentinel_calls_at = []

    def recording_sample_custom(model, noise, cfg, sampler, sigmas, positive, negative,
                                latent_image, noise_mask=None, callback=None,
                                disable_pbar=False, seed=None):
        wrapper = model.model_options.get("model_function_wrapper")
        if wrapper is not None:
            # Mirrors what a real comfy.sample.sample_custom does: invoke whatever
            # wrapper is currently installed, so a chain down to `_sentinel` (if any)
            # actually fires instead of sitting unexercised.
            wrapper(lambda x, t, **c: latent_image,
                    {"input": noise, "timestep": torch.tensor([1.0]), "c": {}})
        sentinel_calls_at.append(len(sentinel_calls))
        return fake_sample_custom(model, noise, cfg, sampler, sigmas, positive, negative,
                                  latent_image, noise_mask=noise_mask, callback=callback,
                                  disable_pbar=disable_pbar, seed=seed)

    monkeypatch.setattr(sys.modules["comfy.sample"], "sample_custom",
                        recording_sample_custom, raising=False)

    latent_template = {"samples": torch.zeros(1, 2, 5, 3, 3)}
    positive = [scene_cond(0)]
    negative = [(torch.zeros(1, 2, 3), {})]

    node.sample(
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
        context_windows=True,
        refinement_key_input="testkey",
        explore_first_step=True,
        explore_first_step_candidates=2,
    )

    # 2 throwaway candidate calls from _select_best_seed, then 1 real _sample_chunk call
    # for this single scene.
    assert len(sentinel_calls_at) == 3
    # Neither candidate call's chain ever reached the temporal-wrapper sentinel -- if the
    # ordering bug reappeared (candidates scored on the fully-stacked wrapper with no
    # reset), sentinel_calls would already be nonzero by the second snapshot.
    assert sentinel_calls_at[0] == 0
    assert sentinel_calls_at[1] == 0
    # The real run's call DOES reach the sentinel -- proving the wrapper chain really is
    # installed for the committed run, so the two zeros above are meaningful and not an
    # artifact of the sentinel never firing at all in this test.
    assert sentinel_calls_at[2] == 1


def _explore_first_step_ready_value_fn_patch(monkeypatch):
    class _ReadyValueFn:
        def is_ready(self):
            return True

        def compress(self, x):
            return x

        def forward(self, x):
            return x.mean()

    monkeypatch.setattr(FunPackLTXAVSceneChainSampler, "_load_output_value_function",
                        lambda self, key: _ReadyValueFn(), raising=True)


def test_explore_first_step_candidates_are_isolated_from_context_windows(monkeypatch):
    """A round-3 review found that resetting model_options["model_function_wrapper"]
    (the fix proven by the test above) only isolates candidates from the mechanisms that
    live in THAT chain -- context_windows patches model_options["context_handler"]
    directly, which that reset never touches. It must be torn down before candidate
    scoring and reinstalled before the real run, exactly like the scene's own
    final-teardown finally block does. See the two tests below for v2a_grad_scale and
    identity_overlap, the other two non-wrapper-chain mechanisms round 3 flagged --
    each needs a mutually exclusive gate (custom guides vs. JoyAI memory) to fire for
    real, so they're kept as separate scenes rather than forced into one."""
    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()

    install_calls = []
    remove_calls = []

    def fake_install_context_windows(self, model, length, overlap, schedule, fuse,
                                     freenoise, retain_first):
        install_calls.append("context_windows")
        model.model_options["context_handler"] = "sentinel-handler"

        def _remove():
            remove_calls.append("context_windows")
            model.model_options.pop("context_handler", None)

        return _remove, 10**9, None

    monkeypatch.setattr(FunPackLTXAVSceneChainSampler, "_install_context_windows",
                        fake_install_context_windows, raising=True)
    _explore_first_step_ready_value_fn_patch(monkeypatch)

    context_handler_during_candidates = []

    def recording_sample_custom(model, noise, cfg, sampler, sigmas, positive, negative,
                                latent_image, noise_mask=None, callback=None,
                                disable_pbar=False, seed=None):
        context_handler_during_candidates.append(model.model_options.get("context_handler"))
        return fake_sample_custom(model, noise, cfg, sampler, sigmas, positive, negative,
                                  latent_image, noise_mask=noise_mask, callback=callback,
                                  disable_pbar=disable_pbar, seed=seed)

    monkeypatch.setattr(sys.modules["comfy.sample"], "sample_custom",
                        recording_sample_custom, raising=False)

    latent_template = {"samples": torch.zeros(1, 2, 5, 3, 3)}
    positive = [scene_cond(0)]
    negative = [(torch.zeros(1, 2, 3), {})]

    node.sample(
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
        context_windows=True,
        refinement_key_input="testkey",
        explore_first_step=True,
        explore_first_step_candidates=2,
    )

    # 2 candidate calls, then 1 real _sample_chunk call.
    assert len(context_handler_during_candidates) == 3
    # Neither candidate call saw the context_handler -- it was stripped before scoring.
    assert context_handler_during_candidates[0] is None
    assert context_handler_during_candidates[1] is None
    # The real run's call DOES see it -- proving it was reinstalled before the committed
    # run, not just torn down and left off.
    assert context_handler_during_candidates[2] == "sentinel-handler"
    # Installed once for the scene, torn down for candidate scoring, reinstalled for the
    # real run, then torn down again at scene teardown: install x2, remove x2.
    assert install_calls == ["context_windows", "context_windows"]
    assert remove_calls == ["context_windows", "context_windows"]


def test_explore_first_step_reinstall_failure_replaces_not_duplicates_the_run_report_entry(monkeypatch):
    """A round-6 review found that a failed reinstall appended a SKIPPED note to
    run_mechanisms WITHOUT removing the SUCCESS-shaped entry context_windows' original
    install already added earlier in the same scene -- the run report would then tell the
    user the mechanism both ran and didn't for the same scene. The fix strips any earlier
    entry for that mechanism before appending the SKIPPED one; this proves exactly one
    context_windows entry survives, and that it's the SKIPPED one."""
    import json
    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()

    install_attempt = []

    def fake_install_context_windows(self, model, length, overlap, schedule, fuse,
                                     freenoise, retain_first):
        install_attempt.append(1)
        if len(install_attempt) == 1:
            model.model_options["context_handler"] = "sentinel-handler"

            def _remove():
                model.model_options.pop("context_handler", None)

            return _remove, 10**9, None
        # The reinstall (second call, made by explore_first_step after candidate
        # scoring) fails -- this is the state round 6 found was mis-reported.
        raise RuntimeError("reinstall boom")

    monkeypatch.setattr(FunPackLTXAVSceneChainSampler, "_install_context_windows",
                        fake_install_context_windows, raising=True)
    _explore_first_step_ready_value_fn_patch(monkeypatch)

    latent_template = {"samples": torch.zeros(1, 2, 5, 3, 3)}
    positive = [scene_cond(0)]
    negative = [(torch.zeros(1, 2, 3), {})]

    *_rest, boundaries_json = node.sample(
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
        context_windows=True,
        refinement_key_input="testkey",
        explore_first_step=True,
        explore_first_step_candidates=2,
    )

    mechs = json.loads(boundaries_json)["scenes"][0]["mechanisms"]
    ctx_entries = [m for m in mechs if m.startswith("context_windows")]
    # Exactly one context_windows entry survives -- the success-shaped one from the
    # original install must have been replaced, not left alongside the failure note.
    assert len(ctx_entries) == 1
    assert "SKIPPED" in ctx_entries[0]
    assert "reinstall boom" in ctx_entries[0]


def test_explore_first_step_candidates_are_isolated_from_v2a_grad_scale(monkeypatch):
    """v2a_grad_scale installs raw torch forward hooks directly on submodules -- not the
    model_function_wrapper chain the earlier test's reset covers. Real _install_v2a_scale
    needs a real model.model.diffusion_model.transformer_blocks, which FakeModel doesn't
    have, so it's faked the same way context_windows is faked above. joyai_audio_memory's
    own gate lives ONLY in the continuation-scene branch (there's nothing to remember
    before scene 1 finishes), so this needs two scenes -- scene 1 never touches v2a at
    all, scene 2 does."""
    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()
    model = FakeModel()  # _remove_v2a_scale takes no `model` argument in the real
                         # signature, so the removal fake below must close over this
                         # instance directly, matching the one passed to node.sample().

    install_calls = []
    remove_calls = []

    def fake_install_v2a_scale(self, model, scale):
        install_calls.append("v2a")
        model.model_options["_v2a_sentinel"] = True
        return ["v2a-handle"]

    def fake_remove_v2a_scale(self, handles):
        if handles:
            remove_calls.append("v2a")
            model.model_options.pop("_v2a_sentinel", None)

    monkeypatch.setattr(FunPackLTXAVSceneChainSampler, "_install_v2a_scale",
                        fake_install_v2a_scale, raising=True)
    monkeypatch.setattr(FunPackLTXAVSceneChainSampler, "_remove_v2a_scale",
                        fake_remove_v2a_scale, raising=True)
    # audio_tail is computed from a real JoyAI memory bank this test has no reason to
    # build -- force it just enough to make the real call site's `if joyai_audio_memory
    # and audio_tail > 0:` gate fire.
    monkeypatch.setattr(FunPackLTXAVSceneChainSampler, "_append_joyai_audio_memory",
                        lambda self, chunk, audio_frames: (chunk, 5), raising=True)
    _explore_first_step_ready_value_fn_patch(monkeypatch)

    v2a_sentinel_during_candidates = []

    def recording_sample_custom(model, noise, cfg, sampler, sigmas, positive, negative,
                                latent_image, noise_mask=None, callback=None,
                                disable_pbar=False, seed=None):
        v2a_sentinel_during_candidates.append(model.model_options.get("_v2a_sentinel"))
        return fake_sample_custom(model, noise, cfg, sampler, sigmas, positive, negative,
                                  latent_image, noise_mask=noise_mask, callback=callback,
                                  disable_pbar=disable_pbar, seed=seed)

    monkeypatch.setattr(sys.modules["comfy.sample"], "sample_custom",
                        recording_sample_custom, raising=False)

    latent_template = {"samples": torch.zeros(1, 2, 5, 3, 3)}
    positive = [scene_cond(0), scene_cond(1)]
    negative = [(torch.zeros(1, 2, 3), {})]

    node.sample(
        model=model,
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
        joyai_memory=True,
        joyai_memory_size=4,
        joyai_audio_memory=True,
        v2a_grad_scale=1.5,
        refinement_key_input="testkey",
        explore_first_step=True,
        explore_first_step_candidates=2,
    )

    # Scene 1 (first scene) has nothing to remember yet -- v2a never installs, so its 3
    # calls (2 candidates + 1 real) all see no sentinel. Scene 2 is a continuation scene,
    # where the gate fires: its 3 calls are the ones that matter for this test.
    assert len(v2a_sentinel_during_candidates) == 6
    assert v2a_sentinel_during_candidates[:3] == [None, None, None]
    scene2 = v2a_sentinel_during_candidates[3:]
    assert scene2[0] is None
    assert scene2[1] is None
    assert scene2[2] is True
    assert install_calls == ["v2a", "v2a"]
    assert remove_calls == ["v2a", "v2a"]


def test_v2a_install_failure_at_the_original_call_site_degrades_the_scene_not_the_render(monkeypatch):
    """A round-6 review found that hardening _install_v2a_scale's hook-attachment loop
    (so a mid-loop failure rolls back and re-raises, rather than silently returning
    whatever it managed) turned the ORIGINAL (non-explore) call site at the top of the
    scene into a new, previously-impossible crash path: before that hardening, this loop
    never raised at all, so this call site never needed a guard. Sibling mechanisms
    (context_windows, identity_overlap) already report a failed install as a falsy
    result, never a raise -- v2a's original install must follow the same declared-limit
    contract: log it, note it in the run report, and let the scene run without it,
    rather than aborting the whole render over one scene's hook failure."""
    import json
    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()

    def fake_install_v2a_scale(self, model, scale):
        raise RuntimeError("hook registration boom")

    monkeypatch.setattr(FunPackLTXAVSceneChainSampler, "_install_v2a_scale",
                        fake_install_v2a_scale, raising=True)
    monkeypatch.setattr(FunPackLTXAVSceneChainSampler, "_append_joyai_audio_memory",
                        lambda self, chunk, audio_frames: (chunk, 5), raising=True)

    latent_template = {"samples": torch.zeros(1, 2, 5, 3, 3)}
    positive = [scene_cond(0), scene_cond(1)]
    negative = [(torch.zeros(1, 2, 3), {})]

    # Must not raise -- the whole point of the fix.
    *_rest, boundaries_json = node.sample(
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
        joyai_memory=True,
        joyai_memory_size=4,
        joyai_audio_memory=True,
        v2a_grad_scale=1.5,
    )

    mechs = json.loads(boundaries_json)["scenes"][1]["mechanisms"]
    v2a_entries = [m for m in mechs if m.startswith("v2a_grad_scale")]
    assert len(v2a_entries) == 1
    assert "SKIPPED" in v2a_entries[0]
    assert "hook registration boom" in v2a_entries[0]


def test_explore_first_step_candidates_are_isolated_from_identity_overlap(monkeypatch):
    """identity_overlap monkeypatches diffusion_model methods directly -- not the
    model_function_wrapper chain. Real _install_identity_overlap needs a real
    model.model.diffusion_model to patch, which FakeModel doesn't have, so it's faked the
    same way context_windows/v2a are faked above. identity_ref_filename is a LOCAL
    variable inside sample(), only ever set (for the first scene) by
    _apply_configured_guides when a custom guide stack is present -- faked here along
    with the funpack_scene_guides JSON parse that feeds it, rather than hand-building
    real guide JSON."""
    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()
    model = FakeModel()  # _strip_identity_overlap takes no `model` argument in the real
                         # signature, so the removal fake below must close over this
                         # instance directly, matching the one passed to node.sample().

    install_calls = []
    remove_calls = []

    def fake_install_identity_overlap(self, model, ref_latent, seg_value):
        install_calls.append("identity_overlap")
        model.model_options["_identity_overlap_sentinel"] = True
        return "identity-overlap-handle"

    def fake_strip_identity_overlap(self, handle):
        if handle:
            remove_calls.append("identity_overlap")
            model.model_options.pop("_identity_overlap_sentinel", None)

    monkeypatch.setattr(FunPackLTXAVSceneChainSampler, "_install_identity_overlap",
                        fake_install_identity_overlap, raising=True)
    monkeypatch.setattr(FunPackLTXAVSceneChainSampler, "_strip_identity_overlap",
                        fake_strip_identity_overlap, raising=True)
    # _resolve_identity_overlap normally encodes a real image via VAE; fake it to return
    # a truthy ref_latent so the real call site's `if _id_ref_latent is not None:` gate
    # fires. pos_tokens=None skips the (separately tested) ArcFace token-append branch.
    monkeypatch.setattr(FunPackLTXAVSceneChainSampler, "_resolve_identity_overlap",
                        lambda self, *a, **k: (torch.zeros(1, 2, 3), 2.0, None, None),
                        raising=True)
    monkeypatch.setattr(FunPackLTXAVSceneChainSampler, "_parse_scene_guides",
                        lambda self, raw: {"scenes": [{"identity_pin": "fake.png"}]},
                        raising=True)
    monkeypatch.setattr(
        FunPackLTXAVSceneChainSampler, "_apply_configured_guides",
        lambda self, chunk, scene_index, custom_guides, latent_template, scene_outputs,
               scene_media_by_ref, scene_positive, scene_negative, vae,
               identity_transfer_enabled=False: (
            chunk, scene_positive, scene_negative, 0, 0, "fake.png"),
        raising=True)
    _explore_first_step_ready_value_fn_patch(monkeypatch)

    identity_sentinel_during_candidates = []

    def recording_sample_custom(model, noise, cfg, sampler, sigmas, positive, negative,
                                latent_image, noise_mask=None, callback=None,
                                disable_pbar=False, seed=None):
        identity_sentinel_during_candidates.append(
            model.model_options.get("_identity_overlap_sentinel"))
        return fake_sample_custom(model, noise, cfg, sampler, sigmas, positive, negative,
                                  latent_image, noise_mask=noise_mask, callback=callback,
                                  disable_pbar=disable_pbar, seed=seed)

    monkeypatch.setattr(sys.modules["comfy.sample"], "sample_custom",
                        recording_sample_custom, raising=False)

    latent_template = {"samples": torch.zeros(1, 2, 5, 3, 3)}
    positive = [scene_cond(0)]
    negative = [(torch.zeros(1, 2, 3), {})]

    node.sample(
        model=model,
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
        identity_transfer_enabled=True,
        funpack_scene_guides="ignored-because-_parse_scene_guides-is-faked",
        refinement_key_input="testkey",
        explore_first_step=True,
        explore_first_step_candidates=2,
    )

    assert len(identity_sentinel_during_candidates) == 3
    assert identity_sentinel_during_candidates[0] is None
    assert identity_sentinel_during_candidates[1] is None
    assert identity_sentinel_during_candidates[2] is True
    assert install_calls == ["identity_overlap", "identity_overlap"]
    assert remove_calls == ["identity_overlap", "identity_overlap"]


def test_output_value_snapshot_is_saved_on_h3(monkeypatch):
    """Regression (2026-09-22): the snapshot feeding the output-space value function used
    to be saved `if not self._is_h3` -- a leftover from when output_guidance (forced off
    on H3) was its only consumer. explore_first_step reads the SAME value function and is
    NOT H3-excluded, so skipping the snapshot there permanently starved it: the console
    printed "value function not ready yet (needs 10+ rated generations)" forever, no
    matter how many generations were rated, because MIN_SAMPLES could never be reached
    with zero snapshots ever written. This proves the snapshot call now fires on H3 too."""
    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()

    import minimax_h3
    monkeypatch.setattr(minimax_h3, "is_h3_model", lambda model: True, raising=True)

    snapshot_calls = []
    monkeypatch.setattr(FunPackLTXAVSceneChainSampler, "_save_output_value_snapshot",
                        lambda self, key, snap, mask: snapshot_calls.append(key), raising=True)

    latent_template = {"samples": torch.zeros(1, 2, 5, 3, 3)}
    positive = [scene_cond(0)]
    negative = [(torch.zeros(1, 2, 3), {})]

    node.sample(
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
        refinement_key_input="testkey",
    )

    assert snapshot_calls == ["testkey"]


def test_select_best_seed_reports_when_no_candidate_scores(monkeypatch, capsys):
    """Regression (2026-09-22): explore_first_step's candidate loop used to `continue`
    silently both when a candidate produced no usable prediction and when scoring itself
    raised, then return the unchanged seed with NO console output at all if every
    candidate failed this way -- indistinguishable from "it ran and picked the original
    seed" to a user just watching generation timing. fake_sample_custom (below) never
    invokes the model_function_wrapper, so every candidate here naturally produces no
    capture -- exactly the silent path this test pins."""
    monkeypatch.setattr(sys.modules["comfy.sample"], "prepare_noise",
                        fake_prepare_noise, raising=False)
    monkeypatch.setattr(sys.modules["comfy.sample"], "sample_custom",
                        fake_sample_custom, raising=False)
    sample_calls.clear()
    node = FunPackLTXAVSceneChainSampler()

    class _ReadyValueFn:
        def is_ready(self):
            return True

    latent = {"samples": torch.zeros(1, 2, 5, 3, 3)}
    positive = [scene_cond(0)]
    negative = [(torch.zeros(1, 2, 3), {})]

    result = node._select_best_seed(
        model=FakeModel(), sampler=object(), sigmas=torch.tensor([1.0, 0.0]), seed=10,
        cfg=1.5, positive=positive, negative=negative, latent=latent, n_candidates=2,
        value_fn=_ReadyValueFn(),
    )

    assert result == 10, "seed must be unchanged when nothing scored"
    out = capsys.readouterr().out
    assert out.count("produced no usable prediction") == 2, "one line per failed candidate"
    assert "0/2 candidate(s) produced a usable score" in out
    assert "seed unchanged" in out


def test_select_best_seed_reports_a_scoring_exception_per_candidate(monkeypatch, capsys):
    """The OTHER silent path this same regression covers: a candidate that DOES produce a
    capture but whose value_fn.forward/.compress call raises must also say so, not just
    the couldn't-extract-a-prediction case above."""
    class _RecordingWrapperModel:
        def __init__(self):
            self.model_options = {}

    def sample_custom_that_fires_the_wrapper(model, noise, cfg, sampler, sigmas, positive,
                                             negative, latent_image, noise_mask=None,
                                             callback=None, disable_pbar=False, seed=None):
        wrapper = model.model_options.get("model_function_wrapper")
        wrapper(lambda x, t, **c: latent_image,
                {"input": noise, "timestep": torch.tensor([1.0]), "c": {}})
        return latent_image

    monkeypatch.setattr(sys.modules["comfy.sample"], "prepare_noise",
                        fake_prepare_noise, raising=False)
    monkeypatch.setattr(sys.modules["comfy.sample"], "sample_custom",
                        sample_custom_that_fires_the_wrapper, raising=False)
    node = FunPackLTXAVSceneChainSampler()

    class _BoomValueFn:
        def is_ready(self):
            return True

        def compress(self, x):
            raise RuntimeError("boom")

    latent = {"samples": torch.zeros(1, 2, 5, 3, 3)}
    positive = [scene_cond(0)]
    negative = [(torch.zeros(1, 2, 3), {})]

    result = node._select_best_seed(
        model=_RecordingWrapperModel(), sampler=object(), sigmas=torch.tensor([1.0, 0.0]),
        seed=10, cfg=1.5, positive=positive, negative=negative, latent=latent,
        n_candidates=2, value_fn=_BoomValueFn(),
    )

    assert result == 10
    out = capsys.readouterr().out
    assert out.count("failed to score") == 2
    assert "0/2 candidate(s) produced a usable score" in out


def test_output_value_fn_sample_count_reads_an_under_threshold_file(monkeypatch, tmp_path):
    """Regression (2026-09-22): 'value function not ready yet' used to give no indication
    of how many rated generations it actually has banked -- the user called this out as
    "literally blind" on the count. _load_output_value_function collapses an under-
    threshold value function to None (correctly -- every other caller trusts that as
    "not safe to steer/select with"), so the count has to come from a SEPARATE read."""
    import conditioning
    from value_function import LatentValueFunction

    monkeypatch.setattr(conditioning, "refinement_state_path",
                        lambda key, mode, prefix="refine", extension="json":
                            str(tmp_path / f"{key}.{mode}.{extension}"), raising=True)

    node = FunPackLTXAVSceneChainSampler()
    assert node._output_value_fn_sample_count("testkey") is None, \
        "no file at all -- never trained, not merely under threshold"

    vf = LatentValueFunction(hidden_dim=LatentValueFunction.DEFAULT_HIDDEN_DIM)
    for i in range(4):
        vf.train_on(torch.zeros(LatentValueFunction.DEFAULT_HIDDEN_DIM), float(i))
    vf.save(str(tmp_path / "testkey.value_fn_x0.pt"))

    assert vf.is_ready() is False, "fixture must actually be under MIN_SAMPLES (10)"
    assert node._output_value_fn_sample_count("testkey") == 4
    assert node._load_output_value_function("testkey") is None, \
        "the gated loader must still treat this as not-ready"
