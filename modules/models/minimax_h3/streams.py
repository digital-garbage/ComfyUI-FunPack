"""Where H3's video lives in a sampling-time latent: the first of [video, audio]."""


def video_stream(x):
    """-> (video [B, C, T, H, W], rebuild(new_video) -> x), or None when `x` is
    not H3's two-branch latent."""
    if not getattr(x, "is_nested", False) or len(x.tensors) != 2 or x.tensors[0].dim() != 5:
        return None
    from comfy.nested_tensor import NestedTensor
    rest = x.tensors[1:]
    return x.tensors[0], lambda video: NestedTensor([video, *rest])
