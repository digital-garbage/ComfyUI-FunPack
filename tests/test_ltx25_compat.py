"""Unit tests for the two LTX-2.5 compatibility fixes.

LTX 2.5 reuses LTXAVModel/LTXBaseModel and is gated by config flags, so nearly all of
FunPack binds to it unchanged. Two places assumed 2.3 specifics and would have failed
SILENTLY — which is the only reason they need tests:

1. `conditioning.register_ltxav_split_from_model` — the video/audio text-context split was
   a hardcoded width table. From 2.5 on comfy reads those widths off the checkpoint, so an
   unseen pair made `ltxav_video_channels` return the full width and every steering edit
   moved the audio context too. Symptom: degraded audio, no error.

2. `_vae_with_decode_noise` — 2.5's diffusion decoder takes no decode timestep, but it is an
   nn.Module, so stamping the attributes onto it succeeds and is then ignored. Symptom: a
   knob that reads as live and does nothing.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import conditioning  # noqa: E402
import samplers  # noqa: E402


def _model(video_dim, audio_dim):
    """A stand-in for the ModelPatcher -> BaseModel -> LTXAVModel chain the sampler walks."""
    dm = types.SimpleNamespace(cross_attention_dim=video_dim, audio_cross_attention_dim=audio_dim)
    return types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=dm))


@pytest.fixture(autouse=True)
def _clean_split_state():
    """The learned split and the log-once set are module-global and shared across tests."""
    learned = dict(conditioning._FUNPACK_AV_SPLIT_LEARNED)
    logged = set(conditioning._FUNPACK_AV_LOGGED)
    conditioning._FUNPACK_AV_SPLIT_LEARNED.clear()
    conditioning._FUNPACK_AV_LOGGED.clear()
    yield
    conditioning._FUNPACK_AV_SPLIT_LEARNED.clear()
    conditioning._FUNPACK_AV_SPLIT_LEARNED.update(learned)
    conditioning._FUNPACK_AV_LOGGED.clear()
    conditioning._FUNPACK_AV_LOGGED.update(logged)


# --- 1. model-derived AV channel split ------------------------------------------------------

def test_registers_split_from_model_widths():
    assert conditioning.register_ltxav_split_from_model(_model(4096, 2048)) == 4096
    assert conditioning.ltxav_video_channels(6144) == 4096


def test_unseen_widths_now_protect_audio():
    """The whole point: a split the static table gets WRONG must be overridden. A checkpoint
    at 5120+2560 also totals 7680, where the table would blindly answer 3840 and hand half the
    audio context to the video steering."""
    assert conditioning.ltxav_video_channels(7680) == 3840  # table's LTXv2 entry
    conditioning._FUNPACK_AV_LOGGED.clear()
    conditioning.register_ltxav_split_from_model(_model(5120, 2560))
    assert conditioning.ltxav_video_channels(7680) == 5120  # model wins over the table


def test_learned_split_protects_audio_end_to_end():
    conditioning.register_ltxav_split_from_model(_model(5120, 2560))
    original = torch.randn(1, 3, 7680)
    steered = original + 1.0
    out = conditioning.protect_audio_channels(steered, original)
    assert torch.equal(out[..., :5120], steered[..., :5120]), "video half must keep the steering"
    assert torch.equal(out[..., 5120:], original[..., 5120:]), "audio half must be restored"


def test_single_stream_model_registers_nothing():
    """Video-only LTXV has no audio width; callers must keep no-opping on the full tensor."""
    assert conditioning.register_ltxav_split_from_model(_model(4096, 0)) is None
    assert conditioning._FUNPACK_AV_SPLIT_LEARNED == {}
    assert conditioning.ltxav_video_channels(4096) == 4096


def test_missing_model_attributes_are_not_fatal():
    assert conditioning.register_ltxav_split_from_model(types.SimpleNamespace()) is None
    assert conditioning.register_ltxav_split_from_model(None) is None


def test_registration_is_idempotent():
    first = conditioning.register_ltxav_split_from_model(_model(4096, 2048))
    second = conditioning.register_ltxav_split_from_model(_model(4096, 2048))
    assert first == second == 4096
    assert conditioning._FUNPACK_AV_SPLIT_LEARNED == {6144: 4096}


# --- 2. decode-noise capability check -------------------------------------------------------

class _ConvDecoderVAE:
    """Stands in for comfy's VideoVAE, which sets both attributes in __init__ and reads them."""
    def __init__(self):
        self.first_stage_model = types.SimpleNamespace(decode_timestep=0.05, decode_noise_scale=0.025)


class _DiffusionDecoderVAE:
    """Stands in for CausalDiffusionVAE: no decode_timestep, and assignment still succeeds."""
    def __init__(self):
        self.first_stage_model = types.SimpleNamespace()


def _chain():
    return samplers.FunPackLTXAVSceneChainSampler()


def test_conv_decoder_still_receives_decode_noise():
    vae = _ConvDecoderVAE()
    out = _chain()._vae_with_decode_noise(vae, 0.1, 0.3, 42)
    assert out is not vae, "must never mutate the shared input VAE"
    assert out.first_stage_model.decode_timestep == 0.1
    assert out.first_stage_model.decode_noise_scale == 0.3
    assert out.seed == 42


def test_diffusion_decoder_is_left_alone_and_reported(capsys):
    vae = _DiffusionDecoderVAE()
    out = _chain()._vae_with_decode_noise(vae, 0.1, 0.3, 42)
    assert out is vae, "an unsupported decoder must be returned untouched, not fake-stamped"
    assert not hasattr(vae.first_stage_model, "decode_timestep"), "must not stamp an inert attribute"
    message = capsys.readouterr().out
    assert "IGNORED" in message, "a knob going inert has to say so"
    assert "conv VAE" in message, "and name the way to get it back"


def test_conv_decoder_stays_silent(capsys):
    _chain()._vae_with_decode_noise(_ConvDecoderVAE(), 0.1, 0.3, 42)
    assert "IGNORED" not in capsys.readouterr().out
