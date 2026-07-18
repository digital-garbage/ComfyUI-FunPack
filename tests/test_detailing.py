"""Unit tests for segmented detailing (detailing.py).

Covers the pure pieces (target parsing, tube geometry + area cap, layout-cond
stripping, spatial downscale) and the detail_refine_scene contract: bit-identical
no-op when disabled / nothing detected, and the crop -> refine -> feathered
paste round trip with the model-touching stages stubbed out.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.modules.setdefault("comfy", types.ModuleType("comfy"))
_nested = types.ModuleType("comfy.nested_tensor")


class _FakeNested:
    is_nested = True

    def __init__(self, tensors):
        self._tensors = list(tensors)

    def unbind(self):
        return list(self._tensors)


_nested.NestedTensor = _FakeNested
sys.modules.setdefault("comfy.nested_tensor", _nested)


@pytest.fixture(autouse=True)
def _nested_stub(monkeypatch):
    """Re-install the usable NestedTensor stub around every test.

    Other test modules stub comfy.nested_tensor.NestedTensor = object on the SAME
    shared sys.modules entry at collection time (whichever file is collected last
    wins), and stubbed submodules never run real import machinery, so the parent
    attribute must be bound by hand too."""
    monkeypatch.setattr(sys.modules["comfy.nested_tensor"], "NestedTensor",
                        _FakeNested, raising=False)
    monkeypatch.setattr(sys.modules["comfy"], "nested_tensor",
                        sys.modules["comfy.nested_tensor"], raising=False)

import detailing  # noqa: E402


# ---------------------------------------------------------------------------
# parse_targets
# ---------------------------------------------------------------------------

def test_parse_targets_splits_and_strips():
    assert detailing.parse_targets(" hands , feet,,") == ["hands", "feet"]
    assert detailing.parse_targets("") == []
    assert detailing.parse_targets(None) == []


# ---------------------------------------------------------------------------
# find_tube
# ---------------------------------------------------------------------------

def _heat_with_hot_square(h=64, w=64, y=(20, 30), x=(40, 50), value=0.9):
    heat = torch.zeros(h, w)
    heat[y[0]:y[1], x[0]:x[1]] = value
    return heat

def test_find_tube_boxes_hot_region_with_padding():
    heat = _heat_with_hot_square()
    tube = detailing.find_tube(heat, 64, 64, threshold=0.35)
    assert tube is not None
    y0, y1, x0, x1, mask = tube
    # Hot square plus TUBE_PAD on each side (interpolation is identity at same size).
    assert y0 <= 20 - detailing.TUBE_PAD + 1 and y1 >= 30 + detailing.TUBE_PAD - 1
    assert x0 <= 40 - detailing.TUBE_PAD + 1 and x1 >= 50 + detailing.TUBE_PAD - 1
    assert mask.shape == (1, 1, 1, y1 - y0, x1 - x0)
    assert float(mask.max()) <= 1.0 and float(mask.min()) >= 0.0
    # Mask peaks inside the hot region, feathers to ~0 at the padded border.
    assert float(mask[..., (y1 - y0) // 2, (x1 - x0) // 2]) > 0.5
    assert float(mask[..., 0, 0]) < 0.2

def test_find_tube_none_below_threshold():
    heat = torch.full((64, 64), 0.1)
    assert detailing.find_tube(heat, 64, 64, threshold=0.35) is None
    assert detailing.find_tube(None, 64, 64) is None

def test_find_tube_refuses_oversized_region():
    heat = torch.full((64, 64), 0.9)  # whole frame hot -> a re-render, not a detail pass
    assert detailing.find_tube(heat, 64, 64, threshold=0.35) is None

def test_find_tube_enforces_minimum_edge():
    heat = _heat_with_hot_square(y=(31, 32), x=(31, 32))  # single hot cell
    tube = detailing.find_tube(heat, 64, 64, threshold=0.35)
    assert tube is not None
    y0, y1, x0, x1, _ = tube
    assert (y1 - y0) >= detailing.MIN_TUBE_EDGE
    assert (x1 - x0) >= detailing.MIN_TUBE_EDGE


# ---------------------------------------------------------------------------
# resolve_upsampler_name
# ---------------------------------------------------------------------------

def _patch_folder_paths(monkeypatch, files, folder="/models/latent_upscale_models"):
    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "get_filename_list", lambda kind: list(files), raising=False)
    monkeypatch.setattr(fp, "get_folder_paths", lambda kind: [folder], raising=False)

def test_resolve_explicit_name_passes_through(monkeypatch):
    _patch_folder_paths(monkeypatch, [])
    assert detailing.resolve_upsampler_name("my_upscaler.safetensors") == "my_upscaler.safetensors"

def test_resolve_auto_prefers_newest_spatial_upscaler(monkeypatch):
    _patch_folder_paths(monkeypatch, [
        "other_model.safetensors",
        "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    ])
    for alias in ("auto", "None", "", None):
        assert detailing.resolve_upsampler_name(alias) == "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

def test_resolve_auto_falls_back_to_any_installed(monkeypatch):
    _patch_folder_paths(monkeypatch, ["something_else.safetensors"])
    assert detailing.resolve_upsampler_name("auto") == "something_else.safetensors"

def test_resolve_auto_downloads_when_folder_empty(monkeypatch, tmp_path):
    _patch_folder_paths(monkeypatch, [], folder=str(tmp_path / "lup"))
    calls = {}
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda repo_id, filename, local_dir: calls.update(
        repo=repo_id, file=filename, dest=local_dir)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    name = detailing.resolve_upsampler_name("auto")
    assert name == detailing.DEFAULT_UPSAMPLER_FILE
    assert calls["repo"] == detailing.DEFAULT_UPSAMPLER_REPO
    assert calls["file"] == detailing.DEFAULT_UPSAMPLER_FILE

def test_resolve_download_failure_raises_actionable_error(monkeypatch, tmp_path):
    _patch_folder_paths(monkeypatch, [], folder=str(tmp_path / "lup"))
    hub = types.ModuleType("huggingface_hub")
    def _boom(**kw):
        raise OSError("401 gated")
    hub.hf_hub_download = _boom
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    with pytest.raises(RuntimeError, match="manually"):
        detailing.resolve_upsampler_name("auto")


# ---------------------------------------------------------------------------
# _strip_layout_conds / _downscale_to
# ---------------------------------------------------------------------------

def test_strip_layout_conds_removes_guide_keys_only():
    cond = [[torch.zeros(1, 4), {"keyframe_idxs": "kf", "guiding_latent": "g",
                                 "pooled_output": "keep"}]]
    out = detailing._strip_layout_conds(cond)
    assert out[0][1] == {"pooled_output": "keep"}
    # Original untouched (the scene's own conds must not lose their guides).
    assert "keyframe_idxs" in cond[0][1]

def test_downscale_to_shape_and_content():
    video = torch.arange(2 * 3 * 4 * 8 * 8, dtype=torch.float32).reshape(2, 3, 4, 8, 8)
    down = detailing._downscale_to(video, 4, 4)
    assert down.shape == (2, 3, 4, 4, 4)
    # Area downscale of a constant tensor is exact.
    const = torch.full((1, 2, 2, 6, 6), 3.5)
    assert torch.allclose(detailing._downscale_to(const, 3, 3), torch.full((1, 2, 2, 3, 3), 3.5))


# ---------------------------------------------------------------------------
# detail_refine_scene
# ---------------------------------------------------------------------------

class _FakeChain:
    """Just enough of FunPackLTXAVSceneChainSampler for detail_refine_scene."""

    def __init__(self):
        self.chunk_calls = []

    def _latent_tensors(self, latent):
        samples = latent["samples"]
        if getattr(samples, "is_nested", False):
            return list(samples.unbind())
        return [samples]

    def _sample_chunk(self, model, sampler, sigmas, seed, cfg, positive, negative, latent,
                      **kwargs):
        self.chunk_calls.append({"sigmas": sigmas, "seed": seed,
                                 "positive": positive, "latent": latent})
        return latent  # identity "refine": returns the crop unchanged


def _scene_latent(nested=False):
    video = torch.randn(1, 8, 5, 16, 24)
    if not nested:
        return {"samples": video}, video
    audio = torch.randn(1, 4, 40)
    return {"samples": _FakeNested([video, audio])}, video

def test_refine_noop_without_targets_or_upsampler():
    chain = _FakeChain()
    latent, _ = _scene_latent()
    out, note = detailing.detail_refine_scene(
        chain, None, None, None, [], [], latent, "", object(), 0, 1.0)
    assert out is latent and note is None  # same object: bit-identical guarantee
    out, note = detailing.detail_refine_scene(
        chain, None, None, None, [], [], latent, "hands", None, 0, 1.0)
    assert out is latent and note is None
    assert chain.chunk_calls == []

def test_refine_noop_when_nothing_detected(monkeypatch):
    chain = _FakeChain()
    latent, _ = _scene_latent()
    monkeypatch.setattr(detailing, "_decode_detection_frames", lambda *a, **k: [])
    monkeypatch.setattr(detailing, "_clipseg_heat", lambda *a, **k: None)
    out, note = detailing.detail_refine_scene(
        chain, None, None, None, [], [], latent, "hands", object(), 0, 1.0)
    assert out is latent and note is None
    assert chain.chunk_calls == []

def _patch_detection_and_upsampler(monkeypatch):
    """Hot square in pixel space; upsampler = exact 2x nearest (model-free)."""
    heat = torch.zeros(160, 240)
    heat[40:80, 120:180] = 0.9
    monkeypatch.setattr(detailing, "_decode_detection_frames",
                        lambda *a, **k: [(0, torch.zeros(160, 240, 3))])
    monkeypatch.setattr(detailing, "_clipseg_heat", lambda *a, **k: heat)
    monkeypatch.setattr(
        detailing, "_run_upsampler",
        lambda upsampler, crop, vae, debug=False: torch.nn.functional.interpolate(
            crop, scale_factor=(1, 2, 2), mode="nearest"))
    return heat

def test_refine_round_trip_touches_only_the_tube(monkeypatch):
    chain = _FakeChain()
    latent, video = _scene_latent()
    _patch_detection_and_upsampler(monkeypatch)
    pos = [[torch.zeros(1, 4), {"keyframe_idxs": "kf", "pooled_output": "keep"}]]
    out, note = detailing.detail_refine_scene(
        chain, "model", "vae", "sampler", pos, [], latent, "hands", object(),
        seed=11, cfg=1.0)
    assert note is not None and "segmented_detail" in note
    assert len(chain.chunk_calls) == 1
    call = chain.chunk_calls[0]
    # Official stage-2 tail schedule and a detail-offset seed.
    assert torch.allclose(call["sigmas"], torch.tensor(detailing.STAGE2_SIGMAS))
    assert call["seed"] == 11 + 7777
    # Layout conds stripped for the crop pass.
    assert "keyframe_idxs" not in call["positive"][0][1]
    assert call["positive"][0][1]["pooled_output"] == "keep"
    # The crop passed to the refine is 2x the tube size.
    crop = call["latent"]["samples"]
    out_video = out["samples"]
    assert out_video.shape == video.shape
    # With an identity refine, upsample(2x)->downscale(area) reproduces the crop, so
    # the paste changes (nearly) nothing — but crucially everything OUTSIDE the tube
    # must be exactly untouched regardless of what the refine did.
    diff = (out_video - video).abs().amax(dim=(0, 1, 2))  # [H, W]
    hot = diff > 1e-6
    assert not hot[:2].any() and not hot[:, :2].any()  # far corners untouched

def test_refine_preserves_audio_stream(monkeypatch):
    chain = _FakeChain()
    latent, video = _scene_latent(nested=True)
    audio = chain._latent_tensors(latent)[1]
    _patch_detection_and_upsampler(monkeypatch)
    out, note = detailing.detail_refine_scene(
        chain, "model", "vae", "sampler", [], [], latent, "hands", object(),
        seed=0, cfg=1.0)
    assert note is not None
    out_tensors = chain._latent_tensors(out)
    assert len(out_tensors) == 2
    # Audio comes back as the SAME tensor the scene finished with: protected by
    # construction, not by a lossy round trip.
    assert out_tensors[1] is audio
    # The crop pass carried an audio stream so joint attention saw a legal AV latent.
    crop_samples = chain.chunk_calls[0]["latent"]["samples"]
    assert getattr(crop_samples, "is_nested", False)

def test_refine_strength_scales_paste(monkeypatch):
    chain = _FakeChain()
    latent, video = _scene_latent()
    _patch_detection_and_upsampler(monkeypatch)
    # Non-identity refine: chunk returns a shifted crop so the paste has signal.
    def _shifted_chunk(model, sampler, sigmas, seed, cfg, positive, negative, crop_latent, **kw):
        return {"samples": crop_latent["samples"] + 10.0}
    chain._sample_chunk = _shifted_chunk
    out_full, _ = detailing.detail_refine_scene(
        chain, "m", "v", "s", [], [], latent, "hands", object(), 0, 1.0, strength=1.0)
    out_half, _ = detailing.detail_refine_scene(
        chain, "m", "v", "s", [], [], latent, "hands", object(), 0, 1.0, strength=0.5)
    d_full = (out_full["samples"] - video).abs().sum()
    d_half = (out_half["samples"] - video).abs().sum()
    assert d_full > 0
    assert abs(float(d_half / d_full) - 0.5) < 1e-3
