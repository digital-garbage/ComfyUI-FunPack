"""The math: take a negative prompt's direction out of the positive's words. Pure torch, no ComfyUI.

H3 samples at CFG 1, where the negative branch is never run, so a negative prompt does nothing. This
gives it a job: pool it to one unit direction and remove that component from each positive word.

Projection, not subtraction: `h - a*(h.n)n` touches only the part of a word that lies along the
negative (a word at right angles is untouched, and a=1 leaves exactly none of it). Whether a plain
sentence has a clean linear presence in the encoder is unproven: nouns ("a hat") should work better
than adjectives ("ugly").
"""

import torch

MODES = ("project", "subtract")
MAX_GAIN = 2.0     # how far a word may be scaled back up: without a cap a word that WAS the unwanted
                   # concept returns as amplified rounding error


def text_rows(meta, length):
    """Indices of the text rows (H3 tags each row's kind, 1 = text), or None for 'every row'."""
    tags = (meta or {}).get("minimax_token_tags")
    if tags is None:
        return None
    flat = tags.reshape(-1)
    if int(flat.numel()) != int(length):
        return None
    idx = (flat == 1).nonzero(as_tuple=False).reshape(-1)
    return idx if idx.numel() else None


def direction(tensor, meta=None):
    """The unit vector the negative occupies (its text rows pooled), or None."""
    if tensor is None or not hasattr(tensor, "dim"):
        return None
    t = tensor.detach().float()
    t = t[0] if t.dim() == 3 else t
    if t.dim() != 2 or t.shape[0] == 0:
        return None
    idx = text_rows(meta, t.shape[0])
    if idx is not None:
        t = t.index_select(0, idx.to(t.device))
    v = t.mean(dim=0)
    n = float(v.norm())
    return v / n if n and n == n else None


def erase(tensor, unit, strength, meta=None, mode="project", renorm=True):
    """`tensor` with `unit` taken out of its text rows; the same tensor when nothing can be done."""
    if tensor is None or unit is None or not strength:
        return tensor
    out = tensor.detach().float().clone()
    squeeze = out.dim() == 3
    work = out[0] if squeeze else out
    if work.dim() != 2 or work.shape[-1] != int(unit.shape[-1]):
        return tensor
    u = unit.to(device=work.device, dtype=work.dtype)
    idx = text_rows(meta, work.shape[0])
    rows = work if idx is None else work.index_select(0, idx.to(work.device))
    before = rows.norm(dim=-1, keepdim=True)
    if mode == "subtract":
        rows = rows - float(strength) * before * u          # a fraction of the row's own size
    else:
        rows = rows - float(strength) * (rows @ u).unsqueeze(-1) * u
    if renorm:
        rows = rows * (before / rows.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=MAX_GAIN)
    if not bool(torch.isfinite(rows).all()):
        return tensor                                       # never hand a non-finite conditioning on
    work = rows if idx is None else work.index_copy(0, idx.to(work.device), rows)
    return (work.unsqueeze(0) if squeeze else work).to(dtype=tensor.dtype, device=tensor.device)


def apply(positive, negative, strength, mode="project", renorm=True):
    """(conditioning, what happened) for a CONDITIONING list. All entries change or none do."""
    if not strength or not positive:
        return positive, "off"
    if not negative:
        return positive, "there is no negative conditioning to take out"
    try:
        unit = direction(negative[0][0], negative[0][1] if len(negative[0]) > 1 else {})
    except Exception as exc:                                # noqa: BLE001
        return positive, f"could not read the negative conditioning ({exc})"
    if unit is None:
        return positive, "the negative prompt is empty or has no usable direction, so nothing was taken out"
    out, changed = [], 0
    for entry in positive:
        meta = entry[1] if len(entry) > 1 else {}
        new = erase(entry[0], unit, strength, meta, mode if mode in MODES else "project", renorm)
        changed += new is not entry[0]
        out.append([new, *entry[1:]])
    if not changed:
        return positive, "the conditioning could not be modified (size mismatch with the negative?)"
    return out, f"{mode} {float(strength):.2f} on {changed} conditioning entr{'y' if changed == 1 else 'ies'}"
