"""MiniMax H3's latent upscaler, loaded without its custom node pack.

The architecture is dictated by the checkpoint, so the tests that matter are the ones that
prove a state dict round-trips: build a small net, save its keys, rebuild from them alone,
and load strictly. If the module names or the block interleaving drift, that fails.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h3_latent_upscaler as h3up  # noqa: E402


def _small(**kw):
    cfg = dict(in_channels=24, in_blocks=2, out_blocks=2, channels=32,
               temporal_every=2, temporal_kernel=5)
    cfg.update(kw)
    return h3up.H3LatentResizer3D(**cfg)


def test_the_architecture_is_read_back_off_the_weights():
    """No config travels with these files, so every dimension has to be recoverable."""
    sd = _small(in_blocks=3, out_blocks=2, channels=64, temporal_kernel=3).state_dict()
    cfg = h3up.detect_config(sd)
    assert cfg == {"in_channels": 24, "in_blocks": 3, "out_blocks": 2, "channels": 64,
                   "temporal_every": 2, "temporal_kernel": 3}


def test_a_checkpoint_without_temporal_convs_is_detected_as_such():
    sd = _small(temporal_every=0).state_dict()
    assert h3up.detect_config(sd)["temporal_every"] == 0


def test_a_state_dict_rebuilds_into_the_same_net_strictly():
    """strict=True is the point: a silently-renamed module would load as random weights."""
    original = _small(in_blocks=3, channels=64)
    rebuilt = h3up.from_state_dict(original.state_dict())
    assert isinstance(rebuilt, h3up.H3LatentResizer3D)
    for key, tensor in original.state_dict().items():
        assert torch.equal(rebuilt.state_dict()[key], tensor)


def test_the_wrapper_prefix_some_checkpoints_use_is_stripped():
    sd = {"upscaler." + k: v for k, v in _small().state_dict().items()}
    assert h3up.is_h3_latent_upscaler(sd)
    assert h3up.from_state_dict(sd) is not None


def test_it_is_told_apart_from_the_architectures_comfyui_already_knows():
    assert not h3up.is_h3_latent_upscaler({"post_upsample_res_blocks.0.conv2.bias": None,
                                           "conv_in.weight": None, "in_blocks.0.x": None})
    assert not h3up.is_h3_latent_upscaler({"blocks.0.block.0.conv.weight": None})
    assert h3up.is_h3_latent_upscaler(_small().state_dict())


def test_upscaling_doubles_the_spatial_axes_and_leaves_time_alone():
    """H3's video latent is on a 5k+2 time grid — resampling time would leave a frame count
    the VAE has no defined decode for."""
    model = _small()
    latent = torch.randn(1, 24, 3, 8, 12)
    out = model.funpack_latent_upscale(latent, scale=2.0)
    assert out.shape == (1, 24, 3, 16, 24)


def test_a_latent_of_the_wrong_width_is_refused_by_name():
    """LTX latents are 128-channel; this model's statistics are 24."""
    model = _small()
    with pytest.raises(ValueError) as exc:
        model.funpack_latent_upscale(torch.randn(1, 128, 2, 8, 8), scale=2.0)
    assert "24-channel" in str(exc.value) and "128-channel" in str(exc.value)


def test_the_published_statistics_cover_every_channel():
    assert len(h3up.LATENTS_MEAN) == len(h3up.LATENTS_STD) == 24


def test_the_upscaler_reports_the_latent_width_it_takes():
    """detailing refuses a mismatched upsampler by naming both numbers; that check reads
    the width off this model."""
    import detailing
    assert detailing.upsampler_in_channels(_small()) == 24


def test_an_attention_checkpoint_says_what_to_do_instead_of_loading_wrong():
    sd = _small().state_dict()
    sd["in_blocks.0.q.weight"] = torch.zeros(1)
    with pytest.raises(ValueError) as exc:
        h3up.from_state_dict(sd)
    assert "custom node" in str(exc.value)
