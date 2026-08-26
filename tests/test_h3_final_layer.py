"""The final layer: the one place a video-only edit stays video-only.

All 50 DiT blocks share a single attention pass, so anything written into the video rows is
read by the audio rows in the next block — that is what made post-block injection corrupt
the soundtrack. The final layer runs after the last attention: video and audio are
modulated on separate rows and leave through separate heads, so an edit to the video branch
here has no path to the audio at all. These tests hold that guarantee, not the arithmetic.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import minimax_h3 as h3

HIDDEN = 4
ROWS = 2


class FakeAdaln(torch.nn.Module):
    """modalities=1: two chunks (shift, scale), one row per timestep."""

    def __init__(self):
        super().__init__()
        self.modalities = 1
        self.shift = torch.tensor([[0.1] * HIDDEN, [0.2] * HIDDEN])
        self.scale = torch.tensor([[0.5] * HIDDEN, [0.3] * HIDDEN])

    def forward(self, _t_emb):
        return self.shift, self.scale


class FakeFinalLayer(torch.nn.Module):
    """Mirrors comfy.ldm.minimax.model.FinalLayer.forward."""

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Identity()
        self.adaln_proj = FakeAdaln()
        self.video_out = torch.nn.Linear(HIDDEN, 3, dtype=torch.float32)
        self.audio_out = torch.nn.Linear(HIDDEN, 2, dtype=torch.float32)

    def forward(self, x, t_emb, video_seg, audio_seg):
        shift, scale = self.adaln_proj(t_emb)
        va, vb, vrow = video_seg
        aa, ab, arow = audio_seg
        hv = (self.norm(x[va:vb]) * (1.0 + scale[vrow]) + shift[vrow]).to(torch.float32)
        ha = (self.norm(x[aa:ab]) * (1.0 + scale[arow]) + shift[arow]).to(torch.float32)
        return self.video_out(hv), self.audio_out(ha)


def run(module, scale):
    x = torch.arange(6 * HIDDEN, dtype=torch.float32).view(6, HIDDEN) / 10.0
    args = (x, None, (0, 3, 0), (3, 6, 1))
    base_v, base_a = module(*args)
    got_v, got_a = h3.FinalLayerVideoScale(module, scale)(*args)
    return (base_v, base_a), (got_v, got_a)


def test_the_audio_is_bit_for_bit_what_it_would_have_been():
    """The whole reason this handle exists. Not 'close' — identical."""
    (_, base_a), (_, got_a) = run(FakeFinalLayer(), 1.6)
    assert torch.equal(base_a, got_a)


def test_the_video_actually_moves():
    (base_v, _), (got_v, _) = run(FakeFinalLayer(), 1.6)
    assert not torch.allclose(base_v, got_v)


def test_a_scale_of_one_is_the_untouched_forward():
    (base_v, base_a), (got_v, got_a) = run(FakeFinalLayer(), 1.0)
    assert torch.equal(base_v, got_v) and torch.equal(base_a, got_a)


def test_the_scale_multiplies_the_modulation_not_the_output():
    """It changes how strongly the final layer READS the accumulated rows. Scaling the
    output instead would be a step-size change, which is a different thing entirely."""
    module = FakeFinalLayer()
    x = torch.arange(6 * HIDDEN, dtype=torch.float32).view(6, HIDDEN) / 10.0
    got_v, _ = h3.FinalLayerVideoScale(module, 2.0)(x, None, (0, 3, 0), (3, 6, 1))
    shift, scale = module.adaln_proj(None)
    want = module.video_out(x[0:3] * (1.0 + scale[0] * 2.0) + shift[0])
    assert torch.allclose(got_v, want)


def test_the_bias_is_left_alone():
    """Scale is a contrast dial; shift would push every channel off its trained centre."""
    module = FakeFinalLayer()
    x = torch.zeros(6, HIDDEN)
    got_v, _ = h3.FinalLayerVideoScale(module, 3.0)(x, None, (0, 3, 0), (3, 6, 1))
    base_v, _ = module(x, None, (0, 3, 0), (3, 6, 1))
    assert torch.allclose(got_v, base_v)   # x is 0, so only the shift survives — unchanged


def test_a_final_layer_of_a_shape_we_do_not_know_declines():
    """Mirroring upstream arithmetic means upstream can move underneath it. Declining is the
    only safe answer — computing something else would be a silent behaviour change."""
    class Different(torch.nn.Module):
        def forward(self, x, t_emb, video_seg, audio_seg):
            return x[:1], x[1:2]
    module = Different()
    x = torch.ones(4, HIDDEN)
    got = h3.FinalLayerVideoScale(module, 1.5)(x, None, (0, 3, 0), (3, 4, 1))
    assert torch.equal(got[0], module(x, None, (0, 3, 0), (3, 4, 1))[0])


# ── installing it ───────────────────────────────────────────────────────────

class FakePatcher:
    def __init__(self, final=None):
        self.objects = {"diffusion_model.final_layer": final if final is not None
                        else FakeFinalLayer()}
        self.patched = {}

    def get_model_object(self, name):
        if name in self.patched:
            return self.patched[name]
        return self.objects[name]

    def clone(self):
        other = FakePatcher.__new__(FakePatcher)
        other.objects = self.objects
        other.patched = dict(self.patched)
        return other

    def add_object_patch(self, name, obj):
        self.patched[name] = obj


def test_one_is_a_no_op_and_does_not_clone():
    model = FakePatcher()
    out, note = h3.apply_final_video_scale(model, 1.0)
    assert out is model and note is None and model.patched == {}


def test_it_patches_the_forward_not_the_module():
    """Replacing a module renames every weight under it in the state dict; ComfyUI records
    weight keys off the live tree, so an unwrap by another patcher would walk a path that is
    no longer there."""
    out, note = h3.apply_final_video_scale(FakePatcher(), 1.4)
    assert list(out.patched) == ["diffusion_model.final_layer.forward"]
    assert isinstance(out.patched["diffusion_model.final_layer.forward"],
                      h3.FinalLayerVideoScale)
    assert "audio stream cannot see it" in note


def test_a_model_with_no_final_layer_declines_with_a_reason():
    class Nothing:
        def get_model_object(self, name):
            raise KeyError(name)
    out, note = h3.apply_final_video_scale(Nothing(), 1.4)
    assert isinstance(out, Nothing)
    assert "no final layer" in note


def test_an_unfamiliar_final_layer_declines_before_it_is_installed():
    class Different(torch.nn.Module):
        pass
    model = FakePatcher(final=Different())
    out, note = h3.apply_final_video_scale(model, 1.4)
    assert out is model and model.patched == {}
    assert "not the shape" in note
