"""The picture inside a sampling-time latent, whatever the model packs around it."""


def video_of(x):
    """-> (video [B, C, T, H, W], rebuild(new_video) -> x), or None.

    A model module that packs streams answers `video_stream`; a plain 5-D
    latent is its own picture. Anything else is not guessed at.
    """
    from . import registry
    split = registry.current().ask("video_stream", x)
    if split is not None:
        return split
    if getattr(x, "dim", None) and not getattr(x, "is_nested", False) and x.dim() == 5:
        return x, lambda video: video
    return None
