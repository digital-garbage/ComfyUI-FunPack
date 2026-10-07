"""H3 video decode: spatial tiles, and the X2 Detail VAE's packed output.

Ported from v4 (itself from TripleHeadedMonkey/ComfyUI-MiniMaxH3_LatentUpscaler's
"VAE Decode (fast)"). Two things the stock decode cannot do on H3:

* Honour a tile size. `decode_tiled` ignores the tile arguments on H3, and ComfyUI's
  out-of-memory fallback is a generic 3D tiler that ignores H3's blend rules. Here the
  VAE's OWN tile flags are set for the one call and restored.
* Read an X2 Detail VAE, whose decoder emits packed 12-channel output that has to be
  pixel-shuffled to twice the resolution. Stock decode cannot read it at all.

Unvalidated on a GPU, same as v4's own status.
"""

import math

import torch
import torch.nn.functional as F

TILE_MIN = 256          # smaller tiles show a block grid
SPATIAL_RATIO = 16      # the VAE's spatial compression; the tiler assumes it


def x2_ratio(vae) -> int:
    """PixelShuffle factor of an H3 video VAE whose decoder emits packed RGB:
    1 for the stock VAE, 2 for the X2 Detail VAE. 1 when it cannot be read."""
    try:
        d = vae.first_stage_model.decoder
        rows = int(d.proj_out.weight.shape[0])
        packed = rows // (int(d.patch_size_t) * int(d.patch_size) ** 2) // 3
        r = math.isqrt(packed)
        return r if r >= 1 and r * r == packed and rows % 3 == 0 else 1
    except Exception:                                       # noqa: BLE001
        return 1


def unpack(out, up):
    """[B, 3*up*up, T, H, W] packed phases -> [B, 3, T, H*up, W*up]. Layout is R phases,
    then G, then B (what `repeat_interleave` on the pixel statistics matches)."""
    b, c, t, h, w = out.shape
    out = F.pixel_shuffle(out.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), up)
    return out.reshape(b, t, c // (up * up), h * up, w * up).permute(0, 2, 1, 3, 4)


def tiles(tile_px):
    """(tile, overlap) in pixels: at least TILE_MIN, a multiple of 16, overlap a
    quarter of the tile (bigger tiles with a fixed 64px overlap show a block grid)."""
    tile = max(TILE_MIN, (int(tile_px) // SPATIAL_RATIO) * SPATIAL_RATIO)
    return tile, max(16, (tile * 64 // 256 // 16) * 16)


def decode_fast(vae, video, tile_px):
    """Decode an H3 video latent in `tile_px`-pixel spatial tiles straight through H3's
    decoder; `tile_px` 0 = each frame in one piece (core's own default is 256px tiles with
    64px overlaps: ~1.8x the pixels in many small pieces, the slow part). Returns IMAGES
    (B, T, H, W, C) on the intermediate device. The VAE's flags are restored even when the
    decode raises."""
    import comfy.model_management as mm
    inner = vae.first_stage_model
    up = x2_ratio(vae)
    tile, overlap = tiles(tile_px) if tile_px and tile_px > 0 else (inner.tile_size, inner.tile_overlap_min)
    saved = (inner.tiling, inner.tile_size, inner.tile_overlap_min,
             inner.decoder.out_channels, inner.pixel_mean, inner.pixel_std)
    vae.throw_exception_if_invalid()
    try:
        inner.tiling, inner.tile_size, inner.tile_overlap_min = bool(tile_px and tile_px > 0), tile, overlap
        if up > 1:
            inner.decoder.out_channels = 3 * up * up
        with mm.cuda_device_context(vae.device):
            mm.load_models_gpu([vae.patcher],
                               memory_required=vae.memory_used_decode(video.shape, vae.vae_dtype),
                               force_full_load=vae.disable_offload)
            if up > 1:
                # After the load: a (dynamic) load can restore the checkpoint's 3-channel stats.
                inner.pixel_mean = inner.pixel_mean.repeat_interleave(up * up, dim=1)
                inner.pixel_std = inner.pixel_std.repeat_interleave(up * up, dim=1)
            out = torch.empty(inner.decode_output_shape(video.shape),
                              device=mm.intermediate_device(), dtype=vae.vae_output_dtype())
            inner.decode(video.to(device=vae.device, dtype=vae.vae_dtype), output_buffer=out)
            vae.process_output(out)
    finally:
        (inner.tiling, inner.tile_size, inner.tile_overlap_min,
         inner.decoder.out_channels, inner.pixel_mean, inner.pixel_std) = saved
    if up > 1:
        out = unpack(out, up)
    return out.movedim(1, -1)
