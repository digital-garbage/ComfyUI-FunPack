"""Decoding a latent in tiles the way ComfyUI's own VAE Decode (Tiled) does, for any VAE."""


def tiled_decode(vae, latent, tile_size):
    """The ordinary decode, tiled when asked: a quarter-tile overlap, 64-frame temporal tiles with an
    8-frame overlap (a VAE with no time axis gets none)."""
    if not tile_size or tile_size <= 0:
        return vae.decode(latent)
    overlap = tile_size // 4
    t_size, t_overlap = 64, 8
    t_comp = vae.temporal_compression_decode()
    if t_comp is not None:
        t_size, t_overlap = max(2, t_size // t_comp), max(1, min(t_size // t_comp // 2, t_overlap // t_comp))
    else:
        t_size = t_overlap = None
    c = vae.spacial_compression_decode()
    return vae.decode_tiled(latent, tile_x=tile_size // c, tile_y=tile_size // c, overlap=overlap // c,
                            tile_t=t_size, overlap_t=t_overlap)
