"""Best-FaceID identity transfer — native port of ComfyUI-BFSNodes' LTX Identity Transfer
(alisson-anjos/ComfyUI-BFSNodes, LTXIdentityOverlapConditioning): the reference face is
injected as SEPARATE tokens that share the target's frame-0 RoPE grid (overlap) and are
tagged with a per-source RoPE phase (source_phase), exactly as ltx-trainer's flexible
strategy did at train/validation time. The ref tokens are clean (timestep 0), attend to the
target in self-attention, and are sliced off the output (never rendered) — unlike FunPack's
guide-attention path, they never blend into a real frame. Plus optional ArcFace IdentityProjector
tokens appended to the text context (as trained).

Requires `insightface` (+ buffalo_l, auto-downloaded on first use, CPU) for the ArcFace half;
soft dependency — the overlap tokens alone (identity_projector="None") work without it.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

log = logging.getLogger("FunPackIdentityTransfer")

_USE_GPU = os.environ.get("FUNPACK_ARCFACE_GPU", "0") == "1"
_FACE_APP = None


def _get_face_app():
    global _FACE_APP
    if _FACE_APP is None:
        from insightface.app import FaceAnalysis
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if _USE_GPU else ["CPUExecutionProvider"])
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0 if _USE_GPU else -1, det_size=(640, 640))
        _FACE_APP = app
    return _FACE_APP


def arcface_embed(image_bhwc, mode="auto_adjust"):
    """Return the ArcFace embedding [512], or None if disabled / no face found.
    mode: 'as_is' (detect on the image only), 'auto_adjust' (retry with border-pad
    zoom-out + upscale when detection fails), 'disable' (skip ArcFace entirely)."""
    if mode == "disable":
        return None
    import cv2
    app = _get_face_app()
    img = np.ascontiguousarray(
        (np.clip(image_bhwc[0].detach().cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8)[:, :, ::-1]
    )
    attempts = [img]
    if mode == "auto_adjust":
        h, w = img.shape[:2]
        pad = int(0.4 * max(h, w))
        attempts.append(cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE))
        attempts.append(cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))
        attempts.append(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    for a in attempts:
        faces = app.get(a)
        if faces:
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            return torch.from_numpy(f.normed_embedding.astype(np.float32))
    return None


class IdentityProjector(nn.Module):
    """ArcFace embedding [in_dim] -> num_tokens context tokens [num_tokens, context_dim]."""

    def __init__(self, in_dim=512, context_dim=4096, num_tokens=4, proj=None):
        super().__init__()
        self.in_dim = in_dim
        self.context_dim = context_dim
        self.num_tokens = num_tokens
        self.proj = proj if proj is not None else nn.Sequential(
            nn.Linear(in_dim, 1024), nn.GELU(), nn.Linear(1024, num_tokens * context_dim))
        self.norm = nn.LayerNorm(context_dim)

    def forward(self, e):
        return self.norm(self.proj(e).reshape(-1, self.num_tokens, self.context_dim))


def load_identity_projector(path, device):
    """Load an IdentityProjector of ANY depth by rebuilding proj.N Linears from the
    state_dict (old = 2 linears proj.0/proj.2; enhanced = 3 linears proj.0/proj.2/proj.4)."""
    sd = load_file(path)
    context_dim = sd["norm.weight"].shape[0]
    idxs = sorted({int(k.split(".")[1]) for k in sd if k.startswith("proj.") and k.endswith(".weight")})
    layers = []
    for j, i in enumerate(idxs):
        w = sd[f"proj.{i}.weight"]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        if j < len(idxs) - 1:
            layers.append(nn.GELU())
    proj = nn.Sequential(*layers)
    in_dim = sd["proj.0.weight"].shape[1]
    num_tokens = sd[f"proj.{idxs[-1]}.weight"].shape[0] // context_dim
    p = IdentityProjector(in_dim=in_dim, context_dim=context_dim, num_tokens=num_tokens, proj=proj)
    p.load_state_dict(sd)
    return p.to(device=device, dtype=torch.float32).eval()


def append_context_tokens(conditioning, tokens):
    """Append `tokens` [1 or B, N, D'] onto every entry's text context, padding/truncating
    the last dim to match and extending any attention_mask with all-ones for the new tokens."""
    out = []
    for ce, cd in conditioning:
        t = tokens.to(device=ce.device, dtype=ce.dtype)
        if t.shape[0] != ce.shape[0]:
            t = t.expand(ce.shape[0], -1, -1)
        if t.shape[-1] < ce.shape[-1]:
            t = F.pad(t, (0, ce.shape[-1] - t.shape[-1]))
        elif t.shape[-1] > ce.shape[-1]:
            t = t[..., :ce.shape[-1]]
        nd = cd.copy()
        am = nd.get("attention_mask")
        if am is not None:
            nd["attention_mask"] = torch.cat(
                [am, torch.ones((*am.shape[:-1], t.shape[1]), device=am.device, dtype=am.dtype)], dim=-1)
        out.append([torch.cat([ce, t], dim=1), nd])
    return out


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
