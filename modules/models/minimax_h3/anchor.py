"""H3's i2v anchor: a keyframe PIN in the conditioning payload, not a latent frame.

A pinned keyframe travels as `minimax_payload["keyframes"]`, and its latent as
the matching entry of `cond_video_latents` (pins lead that list, references
follow -- model_base.MiniMaxH3.extra_conds). The opening frame is the pin whose
`resolved_frame_index` is 0. Mid-clip pins are left alone: "loosen the starting
image" is about the opening frame.

Offered as `anchor_pin` so a modifier can swap the anchor for a transformed
copy without learning H3's payload layout.
"""


def anchor_pin(kwargs, transform):
    """apply_model kwargs with the opening pin's latent replaced by
    `transform(latent)`, or None when this call has no opening pin."""
    payload = kwargs.get("minimax_payload")
    if not isinstance(payload, dict):
        return None
    pins = [kf for kf in payload.get("keyframes") or () if kf.get("latent") is not None]
    latents = list(payload.get("cond_video_latents") or ())
    opening = [i for i, kf in enumerate(pins) if int(kf.get("resolved_frame_index", -1)) == 0]
    if not opening or len(latents) < len(pins):
        return None
    for i in opening:
        swapped = transform(latents[i])
        if swapped is None:
            return None
        latents[i] = swapped
    return {**kwargs, "minimax_payload": {**payload, "cond_video_latents": latents}}


def rescale_pins(conditioning, height, width):
    """Conditioning with every keyframe pin's latent resized to (height, width),
    and how many were resized. None when nothing here is H3's.

    A pin is packed as condition ROWS, so its token count belongs to the grid it
    was encoded on: after an upscale between passes the model refuses it outright
    (v4 hit "value tensor of shape [168, 96] cannot be broadcast to [672, 96]").
    Dropping it would un-pin the anchor for pass 2, which is exactly what a
    second pass must not do, so it is resized. Bicubic, not the upscaler: a pin
    has to keep its structure, not gain invented detail.
    """
    import torch.nn.functional as F

    out, changed, ours = [], 0, False
    for entry in conditioning or ():
        meta = entry[1] if isinstance(entry, (list, tuple)) and len(entry) == 2 else None
        if not (isinstance(meta, dict) and meta.get("minimax_keyframes")):
            out.append(entry)
            continue
        ours = True
        pins = []
        for pin in meta["minimax_keyframes"]:
            latent = pin.get("latent")
            if getattr(latent, "ndim", 0) == 5 and tuple(latent.shape[-2:]) != (height, width):
                b, c, t = latent.shape[:3]
                frames = latent.movedim(2, 1).reshape(b * t, c, *latent.shape[-2:])
                resized = F.interpolate(frames.float(), size=(height, width), mode="bicubic",
                                        align_corners=False).to(latent.dtype)
                latent = resized.reshape(b, t, c, height, width).movedim(1, 2)
                changed += 1
            pins.append({**pin, "latent": latent})
        out.append([entry[0], {**meta, "minimax_keyframes": pins}])
    return (out, changed) if ours else None
