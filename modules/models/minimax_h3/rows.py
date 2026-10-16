"""Which rows of H3's joint sequence are the TARGET video and audio.

Upstream packs `[text | cond/ref | audio | video]` and emits one mod_segments
entry per stream, so target video is the LAST entry and target audio the one
before it. Matching by modality tag alone is wrong: tag 0 also marks the i2v
anchor pin, guide keyframes, reference media and the vision tokens inside the
text span, none of which a "video rows" feature may touch (fa04d6c audit).

None -- never an all-False mask -- when the layout cannot be proven (the token
refiner's text-only calls, an upstream layout change), so callers no-op
instead of guessing.
"""

import torch

_TAG = {"video": (0, -1), "audio": (2, -2)}


def target_rows(mod_segments, seq_len, device, stream):
    if stream not in _TAG:
        return None
    tag, at = _TAG[stream]
    segs = list(mod_segments or ())
    if len(segs) < 2:
        return None
    a, b, row = segs[at]
    try:
        ok = bool(((row % 3) == tag).all()) if torch.is_tensor(row) else int(row) % 3 == tag
    except Exception:                            # noqa: BLE001
        return None
    if not ok or not (0 <= a < b <= seq_len):
        return None
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[a:b] = True
    return mask


def text_rows(model_kwargs, length):
    """[length] bool mask of the WORD rows in the text conditioning the model is
    called with, or None when it can't be told. H3's text span carries picture
    tokens too (reference images, tag 0); words are tag 1."""
    payload = (model_kwargs or {}).get("minimax_payload")
    tags = payload.get("text_token_tags") if isinstance(payload, dict) else None
    if tags is None or not hasattr(tags, "view") or tags.numel() != length:
        return None
    return tags.view(-1) == 1
