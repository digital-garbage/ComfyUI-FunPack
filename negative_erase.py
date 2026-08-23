"""Use the negative prompt at CFG 1, by erasing its direction from the positive conditioning.

MiniMax H3 runs at CFG 1.0 always, which means the negative branch is never evaluated and the
negative prompt you wrote does nothing at all. This gives it a job: encode it, pool it to one
direction, and take that direction out of the positive conditioning before the DiT sees it.

**Projection, not subtraction.** `h - a*n` moves every token by the same fixed vector whether or
not it had anything to do with the negative concept, and its magnitude is arbitrary. `h - a*(h.n)n`
removes only the COMPONENT of each token that lies along the negative — a token orthogonal to it
is untouched, and at a=1 the result is exactly orthogonal instead of somewhere past it. Plain
subtraction is kept as a mode because it is what CFG's arithmetic looks like and the comparison is
worth having, but it is not the default.

This is the same class of operation as embed_guidance (nudge the conditioning along a unit
direction), which is validated. What is unvalidated is whether a natural-language negative has a
clean linear presence in the encoder's hidden states. Expect concrete nouns ("a hat", "red") to
behave better than quality adjectives ("ugly", "low quality") — the latter are vague in text space
and probably vague in embedding space too.

Pure tensor math, no ComfyUI imports, so it is testable without a model.
"""

MODES = ("project", "subtract")

# How far a token may be scaled back up after the erase. See the note in `erase`: without a
# cap, a token that WAS the unwanted concept comes back as amplified rounding error.
RENORM_MAX_GAIN = 2.0


def _torch():
    import torch
    return torch


def _text_positions(meta, length):
    """Indices this operation may touch: text positions only when the layout says so.

    H3 packs [text | vision | audio] into one conditioning sequence and tags each position
    with its modality (1 = text). Erasing a text direction out of the VISION span would
    corrupt the reference image's conditioning, which is not what a negative PROMPT was asked
    to do. No tags (LTX and friends) means the whole sequence is text.
    """
    torch = _torch()
    tags = (meta or {}).get("minimax_token_tags")
    if tags is None:
        return None
    try:
        flat = tags.reshape(-1)
        if int(flat.numel()) != int(length):
            return None          # tags and conditioning disagree — touch nothing selectively
        idx = (flat == 1).nonzero(as_tuple=False).reshape(-1)
        return idx if int(idx.numel()) else None
    except Exception:
        return None


def direction(neg_tensor, neg_meta=None):
    """The unit direction the negative prompt occupies, or None.

    Pooled over the negative's own text positions: the negative is a different length from the
    positive, so there is no position-for-position correspondence to subtract. One direction
    applied to every positive token is the honest reading of "this concept, less of it".
    """
    torch = _torch()
    if neg_tensor is None or not hasattr(neg_tensor, "dim"):
        return None
    t = neg_tensor.detach().float()
    if t.dim() == 3:
        t = t[0]
    if t.dim() != 2 or t.shape[0] == 0:
        return None
    idx = _text_positions(neg_meta, t.shape[0])
    if idx is not None:
        t = t.index_select(0, idx.to(t.device))
    v = t.mean(dim=0)
    n = float(v.norm())
    if not n or n != n:          # zero or NaN: no direction to speak of
        return None
    return v / n


def erase(pos_tensor, unit, strength, meta=None, mode="project", renorm=True):
    """Take `unit` out of every text position of `pos_tensor`. Returns a new tensor.

    `renorm` puts each touched token back on its original norm afterwards, changing its
    direction without changing its magnitude. The DiT is sensitive to conditioning scale, and
    projection always shrinks — without this, a large strength reads partly as "quieter
    prompt" rather than purely as "less of that concept".
    """
    torch = _torch()
    if pos_tensor is None or unit is None or not strength:
        return pos_tensor
    mode = mode if mode in MODES else "project"
    out = pos_tensor.detach().float().clone()
    squeeze = out.dim() == 3
    work = out[0] if squeeze else out
    if work.dim() != 2 or work.shape[-1] != int(unit.shape[-1]):
        return pos_tensor
    u = unit.to(device=work.device, dtype=work.dtype)

    idx = _text_positions(meta, work.shape[0])
    rows = work if idx is None else work.index_select(0, idx.to(work.device))
    before = rows.norm(dim=-1, keepdim=True)

    if mode == "project":
        rows = rows - float(strength) * (rows @ u).unsqueeze(-1) * u
    else:
        # Scale-free subtraction: the step is a fraction of the row's OWN magnitude, so the
        # same strength means the same thing on any encoder rather than depending on how
        # large this model's hidden states happen to be.
        rows = rows - float(strength) * before * u

    if renorm:
        # The gain is CAPPED, and that cap is the whole safety of this operation. A token
        # pointing almost exactly along the negative is left with nothing but rounding error
        # after projection; scaling that back to its original norm multiplies noise by
        # thousands and puts a garbage vector at full prompt strength into the conditioning.
        # Past the cap the honest answer is that the token really was mostly the thing you
        # asked to remove, so it stays quiet instead of being refilled with residue.
        after = rows.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        rows = rows * (before / after).clamp(max=RENORM_MAX_GAIN)

    # Never hand a non-finite conditioning onward. It does not fail here — it fails deep in
    # the model, and worse, a run that captures conditioning for the refinement key would
    # bank the bad vector and keep steering toward it long after this setting was turned off.
    if not bool(torch.isfinite(rows).all()):
        return pos_tensor

    if idx is None:
        work = rows
    else:
        work = work.index_copy(0, idx.to(work.device), rows)
    result = work.unsqueeze(0) if squeeze else work
    return result.to(dtype=pos_tensor.dtype, device=pos_tensor.device)


def apply(positive, negative, strength, mode="project", renorm=True):
    """Erase the negative's direction from every entry of a CONDITIONING list.

    Returns (conditioning, note). The note says what happened — including when nothing did,
    because a negative prompt that turns out to be empty must not look like a setting that
    silently failed.
    """
    if not strength or not isinstance(positive, list) or not positive:
        return positive, ""
    if not isinstance(negative, list) or not negative:
        return positive, "negative_erase: no negative conditioning to take out"
    try:
        neg_t = negative[0][0]
        neg_m = negative[0][1] if len(negative[0]) > 1 else {}
        unit = direction(neg_t, neg_m)
    except Exception as exc:  # noqa: BLE001
        return positive, f"negative_erase: could not read the negative conditioning ({exc})"
    if unit is None:
        return positive, "negative_erase: the negative conditioning has no usable direction"

    out, touched = [], 0
    for entry in positive:
        try:
            tensor, meta = entry[0], (entry[1] if len(entry) > 1 else {})
            new = erase(tensor, unit, strength, meta=meta, mode=mode, renorm=renorm)
            if new is not tensor:
                touched += 1
            out.append((new, meta) + tuple(entry[2:]))
        except Exception:  # noqa: BLE001
            out.append(entry)
    if not touched:
        return positive, "negative_erase: no conditioning entry could be modified"
    return out, (f"negative_erase: {mode} {float(strength):.2f} on {touched} "
                 f"conditioning entr{'y' if touched == 1 else 'ies'}"
                 f"{'' if renorm else ', norm not restored'}")
