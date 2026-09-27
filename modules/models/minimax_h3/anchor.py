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
