"""H3 tiled decode and the X2 Detail VAE, against a stand-in VAE.

No weights: the real decoder is a 32B-era ViT3D. What matters here is what the code
DOES around it -- which flags it sets and restores, when it refuses, and the pixel layout.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


class _Proj:
    def __init__(self, rows):
        self.weight = torch.zeros(rows, 4)


class _Decoder:
    def __init__(self, rows, patch_t=1, patch=2):
        self.proj_out, self.patch_size_t, self.patch_size, self.out_channels = _Proj(rows), patch_t, patch, 3


class _Inner:
    def __init__(self, rows):
        self.decoder = _Decoder(rows)
        self.tiling, self.tile_size, self.tile_overlap_min = False, 0, 0
        self.pixel_mean, self.pixel_std = torch.arange(3.).view(1, 3, 1, 1, 1), torch.ones(1, 3, 1, 1, 1)
        self.during = None
        self.fail = False
        self.tried = []

    def decode_output_shape(self, shape):
        b, _, t, h, w = shape
        return (b, self.decoder.out_channels, t, h * 2, w * 2)

    def decode(self, x, output_buffer):
        self.during = (self.tiling, self.tile_size, self.tile_overlap_min,
                       self.decoder.out_channels, self.pixel_mean.shape[1])
        self.tried.append(self.tile_size if self.tiling else 0)
        if self.fail:
            raise RuntimeError(self.fail if isinstance(self.fail, str) else "CUDA out of memory")
        output_buffer.copy_(torch.arange(output_buffer.numel(), dtype=output_buffer.dtype).view_as(output_buffer))


class _Patcher:
    pass


class _Vae:
    """What the decode touches. rows = proj_out rows: 12 (patch 2, 3 colours) is stock, 48 is X2."""
    device = torch.device("cpu")
    vae_dtype = torch.float32
    disable_offload = False

    def __init__(self, rows=12):
        self.first_stage_model = _Inner(rows)
        self.patcher = _Patcher()
        self.plain_calls = 0

    def throw_exception_if_invalid(self):
        pass

    def memory_used_decode(self, shape, dtype):
        return 0

    def vae_output_dtype(self):
        return torch.float32

    def process_output(self, x):
        return x

    def decode(self, latent):
        self.plain_calls += 1
        return torch.zeros(1, 2, 4, 4, 3)


@pytest.fixture
def patched(monkeypatch):
    import comfy.model_management as mm
    import contextlib
    monkeypatch.setattr(mm, "cuda_device_context", lambda d: contextlib.nullcontext())
    monkeypatch.setattr(mm, "load_models_gpu", lambda *a, **k: None)
    monkeypatch.setattr(mm, "intermediate_device", lambda: torch.device("cpu"))


def _mod():
    from modules.models.minimax_h3 import vae_decode
    return vae_decode


def test_x2_ratio_reads_the_decoders_output_rows():
    m = _mod()
    assert m.x2_ratio(_Vae(12)) == 1          # 12 // (1*4) // 3 = 1
    assert m.x2_ratio(_Vae(48)) == 2          # 4 -> sqrt 2
    assert m.x2_ratio(_Vae(27)) == 1          # not a square number of phases
    assert m.x2_ratio(object()) == 1          # unreadable: stock


def test_tiles_are_at_least_256_a_multiple_of_16_with_a_quarter_overlap():
    m = _mod()
    assert m.tiles(64) == (256, 64)
    assert m.tiles(500) == (496, 112)
    assert m.tiles(1024) == (1024, 256)


def test_pixel_unshuffle_layout_is_r_then_g_then_b_phases():
    m = _mod()
    packed = torch.zeros(1, 12, 1, 1, 1)       # 3 colours x 2x2 phases
    for color in range(3):
        for phase in range(4):
            packed[0, color * 4 + phase, 0, 0, 0] = color * 10 + phase
    out = m.unpack(packed, 2)
    assert out.shape == (1, 3, 1, 2, 2)
    for color in range(3):
        assert out[0, color, 0].flatten().tolist() == [color * 10 + p for p in range(4)]


def test_decode_fast_sets_the_tile_flags_for_the_call_and_restores_them(patched):
    m = _mod()
    vae = _Vae(12)
    video = torch.zeros(1, 4, 2, 8, 8)
    out = m.decode_fast(vae, video, 300)
    inner = vae.first_stage_model
    assert inner.during == (True, 288, 64, 3, 3)           # tile 288 = 300 floored to 16, overlap a quarter
    assert (inner.tiling, inner.tile_size, inner.tile_overlap_min, inner.decoder.out_channels) == (False, 0, 0, 3)
    assert out.shape == (1, 2, 16, 16, 3)                  # B, T, H, W, C


def test_flags_are_restored_even_when_the_decode_fails(patched):
    m = _mod()
    vae = _Vae(48)
    vae.first_stage_model.fail = True
    with pytest.raises(RuntimeError):
        m.decode_fast(vae, torch.zeros(1, 4, 2, 8, 8), 256)
    inner = vae.first_stage_model
    assert (inner.tiling, inner.decoder.out_channels, inner.pixel_mean.shape[1]) == (False, 3, 3)


def test_an_x2_vae_decodes_packed_then_shuffles_to_double_resolution(patched):
    m = _mod()
    vae = _Vae(48)
    out = m.decode_fast(vae, torch.zeros(1, 4, 2, 8, 8), 256)
    inner = vae.first_stage_model
    assert inner.during == (True, 256, 64, 12, 12)        # 12 packed channels, stats repeated per phase
    assert out.shape == (1, 2, 32, 32, 3)                 # decode_output_shape doubled, then x2 again by the shuffle


def _provider_decode(vae, tile):
    from modules.models import minimax_h3 as h3
    return h3._video_images(vae, torch.zeros(1, 4, 2, 8, 8), tile)


def test_tile_size_zero_on_a_stock_vae_decodes_each_frame_in_one_piece(patched):
    vae = _Vae(12)
    _provider_decode(vae, 0)
    assert vae.plain_calls == 0 and vae.first_stage_model.during[0] is False      # core's own 256px tiling is off


def test_a_stock_vae_with_tiles_goes_through_the_tiled_decoder(patched):
    vae = _Vae(12)
    _provider_decode(vae, 512)
    assert vae.plain_calls == 0 and vae.first_stage_model.during[:3] == (True, 512, 128)


def test_out_of_memory_in_one_piece_steps_down_to_512_then_256_then_stock(patched):
    from core import log
    log._reset()
    vae = _Vae(12)
    vae.first_stage_model.fail = True
    out = _provider_decode(vae, 0)
    assert vae.first_stage_model.tried == [0, 512, 256] and vae.plain_calls == 1 and out.shape == (1, 2, 4, 4, 3)
    assert any("out of memory decoding in one piece" in r["message"] for r in log.history())


def test_any_other_failure_goes_straight_to_the_stock_decode_and_says_so(patched):
    from core import log
    log._reset()
    vae = _Vae(12)
    vae.first_stage_model.fail = "shape mismatch"
    _provider_decode(vae, 512)
    assert vae.first_stage_model.tried == [512] and vae.plain_calls == 1
    assert any("stock way" in r["message"] for r in log.history())


def test_an_x2_vae_never_falls_back_it_names_itself(patched):
    vae = _Vae(48)
    vae.first_stage_model.fail = True
    with pytest.raises(RuntimeError, match="X2 Detail VAE"):
        _provider_decode(vae, 0)
    assert vae.plain_calls == 0


def test_an_x2_vae_is_tiled_even_when_tiles_are_off(patched):
    vae = _Vae(48)
    _provider_decode(vae, 0)
    assert vae.first_stage_model.during[:2] == (True, 256) and vae.plain_calls == 0
