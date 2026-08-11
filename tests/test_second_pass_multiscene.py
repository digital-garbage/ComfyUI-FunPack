"""Multi-scene chains survive a resolution-changing second_pass_op.

`second_pass_op="upscale_2x"` hands back a scene at twice the latent size, but every later
scene is still built from `latent_template` at the original size. Everything that crosses a
scene boundary — the carried overlap, the anchor's continuation, the soft join, the JoyAI
memory frame, per-scene guide sources — is spliced into a chunk on the template's grid, so
before this fix the second scene died on a shape mismatch and multi-scene + second pass was
simply unavailable.

`_match_template_resolution` is the single choke point. What matters:

1. It brings a resized scene back to the reference grid, and leaves audio alone.
2. It is an exact no-op (same object) when the grids already agree, so every run that does
   not resize anything stays bit-identical.
3. It never raises — a failed resample must not take the whole chain down.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _name in (
    "comfy", "comfy.k_diffusion", "comfy.k_diffusion.sampling",
    "comfy.model_sampling", "comfy.nested_tensor", "comfy.sample",
    "comfy.samplers", "comfy.utils",
):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["comfy.nested_tensor"].NestedTensor = object

import samplers  # noqa: E402


class _FakeNested(list):
    is_nested = True

    def unbind(self):
        return list(self)


@pytest.fixture
def nested_stub():
    comfy_mod = sys.modules["comfy"]
    nested_mod = sys.modules["comfy.nested_tensor"]
    saved = (nested_mod.NestedTensor, getattr(comfy_mod, "nested_tensor", None))
    nested_mod.NestedTensor = _FakeNested
    comfy_mod.nested_tensor = nested_mod
    yield
    nested_mod.NestedTensor = saved[0]
    if saved[1] is None:
        delattr(comfy_mod, "nested_tensor")
    else:
        comfy_mod.nested_tensor = saved[1]


def _node():
    return samplers.FunPackLTXAVSceneChainSampler()


def _latent(h=16, w=24, frames=5):
    return {"samples": torch.randn(1, 128, frames, h, w)}


def _av_latent(h=16, w=24, frames=5, audio=40):
    return {"samples": _FakeNested([torch.randn(1, 128, frames, h, w),
                                    torch.randn(1, 8, audio)])}


# ── the resample ────────────────────────────────────────────────────────────

def test_an_upscaled_scene_comes_back_to_the_template_grid():
    out = _node()._match_template_resolution(_latent(32, 48), _latent(16, 24))
    assert out["samples"].shape[-2:] == (16, 24)


def test_frame_count_and_channels_are_preserved():
    """Only the spatial axes move — a resample that changed the duration would desync the
    carried frames it exists to make usable."""
    out = _node()._match_template_resolution(_latent(32, 48, frames=7), _latent(16, 24, frames=5))
    assert out["samples"].shape[:3] == (1, 128, 7)


def test_matching_grids_are_an_exact_no_op():
    """Identity, not equality: every run without a resizing op must be bit-identical, and
    the cheapest proof of that is that nothing was rebuilt."""
    latent = _latent(16, 24)
    assert _node()._match_template_resolution(latent, _latent(16, 24)) is latent


def test_audio_is_never_touched(nested_stub):
    src = _av_latent(32, 48, audio=40)
    before = src["samples"][1].clone()
    out = _node()._match_template_resolution(src, _av_latent(16, 24, audio=40))
    video, audio = out["samples"].unbind()
    assert video.shape[-2:] == (16, 24)
    assert torch.equal(audio, before), "audio has no spatial axes to resample"


def test_a_stale_mask_is_dropped():
    """A noise_mask describes the OLD grid; the caller builds a fresh one for the chunk it
    splices into, and keeping this one would apply it at the wrong scale."""
    src = _latent(32, 48)
    src["noise_mask"] = torch.zeros(1, 1, 5, 32, 48)
    out = _node()._match_template_resolution(src, _latent(16, 24))
    assert "noise_mask" not in out


def test_the_source_latent_is_never_mutated():
    src = _latent(32, 48)
    before = src["samples"].clone()
    _node()._match_template_resolution(src, _latent(16, 24))
    assert torch.equal(src["samples"], before)


def test_a_smaller_scene_is_grown_to_the_template():
    """Not reachable from upscale_2x, which only grows — but a scene SMALLER than the
    template would corrupt the splice just as surely, so it is handled rather than passed
    through at the wrong shape."""
    out = _node()._match_template_resolution(_latent(8, 12), _latent(16, 24))
    assert out["samples"].shape[-2:] == (16, 24)


def test_constant_input_survives_the_resample():
    """The carried frames are spliced next to real generated ones, so the resample has to be
    energy-preserving in the DC sense — a flat region must not come back brighter or darker."""
    flat = {"samples": torch.full((1, 128, 3, 32, 48), 0.7)}
    out = _node()._match_template_resolution(flat, _latent(16, 24))
    assert torch.allclose(out["samples"], torch.full_like(out["samples"], 0.7), atol=1e-4)


# ── it must never take the chain down ───────────────────────────────────────

def test_an_unreadable_latent_returns_unchanged():
    junk = {"samples": "not a tensor"}
    assert _node()._match_template_resolution(junk, _latent(16, 24)) is junk


def test_an_unreadable_template_returns_unchanged():
    latent = _latent(32, 48)
    assert _node()._match_template_resolution(latent, {"samples": None}) is latent


def test_a_failing_resample_is_reported_and_survived(monkeypatch, capsys):
    """A resample that blows up must cost continuity, not the whole run — and must say so,
    because silently carrying the wrong grid is how this became invisible in the first place."""
    import detailing
    monkeypatch.setattr(detailing, "_downscale_to",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    latent = _latent(32, 48)
    assert _node()._match_template_resolution(latent, _latent(16, 24)) is latent
    assert "continuity" in capsys.readouterr().out
