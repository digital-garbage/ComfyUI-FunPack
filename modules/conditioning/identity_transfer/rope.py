"""Source-phase RoPE tag for the appended reference tokens (port of ltx_core rope.apply_segment_phase)."""

import torch


def _segment_phase(length, seg_value, theta, device, dtype):
    """(cos, sin) of phase = seg_value * theta^(-d/length) over the frequency axis."""
    d = torch.arange(length, device=device, dtype=torch.float32)
    phase = seg_value * (theta ** (-d / float(length)))
    return phase.cos().to(dtype), phase.sin().to(dtype)


def rotate_overlap_freqs(pe, ref_len, seg_value, theta=10000.0):
    """Rotate the LAST ref_len tokens' RoPE freqs by phase = seg_value * theta^(-d/L)
    (port of ltx_core rope.apply_segment_phase).

    This phase tag is what separates the appended reference tokens from the target's real
    frame-0 tokens — they deliberately share the frame-0 coordinate grid, so without it the
    model reads the reference as literal frame 0 and renders the clip's opening as the
    reference image. Never skip it silently; raise instead (see samplers.py's prepare_pe).

    ComfyUI ships the positional embeddings in two layouts, told apart at runtime:

      * ``(cos, sin, split_mode)`` — tokens on axis -2, frequencies on -1. ComfyUI before
        7c59a07 (2026-07-24).
      * ``(rotation_matrix, split_mode)`` — ``(B, T, heads, D, 2, 2)`` holding
        ``[[cos, -sin], [sin, cos]]``, from ComfyUI 7c59a07 "Use comfy kitchen rope
        functions in ltx models" onward.

    Both encode R(angle); the phase composes onto it as R(angle + phase).
    """
    if ref_len <= 0 or seg_value == 0.0:
        return pe
    if not isinstance(pe, (list, tuple)) or len(pe) < 2:
        raise TypeError(
            f"unexpected LTX positional-embedding payload {type(pe).__name__} — "
            f"cannot apply the identity source-phase tag")
    head, second = pe[0], pe[1]
    rest = tuple(pe[2:])

    if torch.is_tensor(second):
        # Legacy cos/sin pair: (..., T, L) for both the split and interleaved rope types.
        cos, sin = head, second
        pc, ps = _segment_phase(cos.shape[-1], seg_value, theta, cos.device, cos.dtype)
        idx = [slice(None)] * cos.dim()
        idx[-2] = slice(cos.shape[-2] - ref_len, cos.shape[-2])
        idx = tuple(idx)
        c0, s0 = cos[idx], sin[idx]
        cos = cos.clone()
        sin = sin.clone()
        cos[idx] = c0 * pc - s0 * ps
        sin[idx] = s0 * pc + c0 * ps
        return (cos, sin, *rest)

    # Rotation-matrix form. The 2x2 block is [[c, -s], [s, c]], so c/s read off column 0.
    mat = head
    if not torch.is_tensor(mat) or mat.dim() < 4 or tuple(mat.shape[-2:]) != (2, 2):
        raise TypeError(
            f"unexpected LTX rope matrix shape {tuple(mat.shape) if torch.is_tensor(mat) else type(mat).__name__} — "
            f"expected (B, T, heads, D, 2, 2); cannot apply the identity source-phase tag")
    if mat.shape[1] < ref_len:
        raise ValueError(
            f"LTX rope matrix has {mat.shape[1]} tokens on axis 1, fewer than the "
            f"{ref_len} reference tokens to tag")
    pc, ps = _segment_phase(mat.shape[-3], seg_value, theta, mat.device, mat.dtype)
    idx = [slice(None)] * mat.dim()
    idx[1] = slice(mat.shape[1] - ref_len, mat.shape[1])  # tokens live on axis 1 here
    idx = tuple(idx)
    blk = mat[idx]
    c0, s0 = blk[..., 0, 0], blk[..., 1, 0]
    c1 = c0 * pc - s0 * ps
    s1 = s0 * pc + c0 * ps
    mat = mat.clone()
    mat[idx] = torch.stack((c1, -s1, s1, c1), dim=-1).reshape(blk.shape)
    return (mat, second, *rest)
