"""The ArcFace half of identity transfer: a face embedding, the trained projector that turns it
into text-context tokens, and appending those tokens to a conditioning.

`insightface` is a soft dependency (buffalo_l is downloaded on first use, on CPU). Without it the
overlap tokens alone still work; the node says so rather than carrying on as if the projector ran.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

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


