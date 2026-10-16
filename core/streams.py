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


def av_video_stream(x, named=None):
    """The picture of an audio+video model whose latent is [video, audio], in this order, in both the
    shapes that reach a feature:

    * the model call's result (APPLY_MODEL) is PACKED: comfy's `_apply_model` flattens the parts into
      one [B, 1, N] tensor with `pack_latents`, and the shapes to undo that ride along as
      `latent_shapes`;
    * elsewhere (callbacks, the finished latent) it is a two-part NestedTensor.

    -> (video [B, C, T, H, W], rebuild(new_video) -> x), or None when `x` is neither.
    """
    import math
    if getattr(x, "is_nested", False):
        if len(x.tensors) != 2 or x.tensors[0].dim() != 5:
            return None
        from comfy.nested_tensor import NestedTensor
        rest = x.tensors[1:]
        return x.tensors[0], lambda video: NestedTensor([video, *rest])
    shapes = (named or {}).get("latent_shapes")
    if not hasattr(x, "dim") or x.dim() != 3 or not shapes or len(shapes) != 2 \
            or len(shapes[0]) != 5:
        return None
    n = math.prod(shapes[0][1:])
    if x.shape[-1] != n + math.prod(shapes[1][1:]):
        return None
    b = x.shape[0]
    video = x[..., :n].reshape(b, *shapes[0][1:])
    rest = x[..., n:]

    def rebuild(new_video):
        import torch
        return torch.cat([new_video.reshape(b, 1, n).to(rest.dtype), rest], dim=-1)

    return video, rebuild


MODEL_ARGS = ("c_concat", "c_crossattn", "control", "transformer_options")


def with_options(args, kwargs, transformer_options):
    """The same call with other transformer_options: positional where comfy put it (4th),
    by name otherwise. -> (args, kwargs)."""
    if len(args) > MODEL_ARGS.index("transformer_options"):
        args = list(args)
        args[MODEL_ARGS.index("transformer_options")] = transformer_options
        return tuple(args), kwargs
    return args, {**kwargs, "transformer_options": transformer_options}


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
