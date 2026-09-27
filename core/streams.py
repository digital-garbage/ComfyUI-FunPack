"""The picture inside a sampling-time latent, whatever the model packs around it."""


def video_of(x, named=None):
    """-> (video [B, C, T, H, W], rebuild(new_video) -> x), or None.

    A model module that packs streams answers `video_stream`; a plain 5-D
    latent is its own picture. Anything else is not guessed at. `named` is the
    model call's arguments (see model_args) -- a packed result is only
    unpackable with the `latent_shapes` that travel with it.
    """
    from . import registry
    split = registry.current().ask("video_stream", x, named or {})
    if split is not None:
        return split
    if getattr(x, "dim", None) and not getattr(x, "is_nested", False) and x.dim() == 5:
        return x, lambda video: video
    return None


MODEL_ARGS = ("c_concat", "c_crossattn", "control", "transformer_options")


def model_args(args, kwargs) -> dict:
    """An APPLY_MODEL wrapper's call, by name.

    ComfyUI hands c_concat, c_crossattn, control and transformer_options on
    POSITIONALLY (model_base.apply_model), and the model's extra conds by name.
    Reading `kwargs.get("transformer_options")` alone finds nothing on a real
    run -- a step gate built on it never opens, silently. A wrapper that calls
    on with changed values may pass everything by name; this reads both shapes.
    """
    named = dict(zip(MODEL_ARGS, args))
    named.update(kwargs)
    return named
