"""Where H3's video lives in a sampling-time latent.

Two shapes reach a feature, and both are real:

* the model call's result (APPLY_MODEL) is PACKED: comfy's `_apply_model`
  flattens [video, audio] into one [B, 1, N] tensor with `pack_latents`, and the
  shapes to undo that ride along as `latent_shapes`;
* elsewhere (callbacks, the finished latent) it is a two-part NestedTensor.
"""

import math


def video_stream(x, named=None):
    """-> (video [B, C, T, H, W], rebuild(new_video) -> x), or None when `x` is
    neither of H3's two shapes."""
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
