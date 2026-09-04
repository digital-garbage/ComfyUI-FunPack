"""Does an intervention add DETAIL, or just noise, or just a different picture?

Every mechanism here that claims to sharpen -- second_pass_op "sharpen", quality_sharpness,
segmented detailing, and now block repeat -- is currently judged by eye. This scores the
claim instead, by comparing a run against the one before it on three numbers:

* **detail**   -- high-frequency energy (Laplacian). Did fine structure increase?
* **structure**-- how much of the low-frequency picture survived. Same shot, or a new one?
* **edge-aligned** -- did the EXTRA high frequency land on edges that already existed, or
  spread evenly? Noise and ringing are high-frequency too, so `detail` alone cannot tell
  sharpening from grain; this is the number that can.

Read together they separate the three outcomes: detail up + edge-aligned + structure kept =
genuinely refined. Detail up + not edge-aligned = grain. Structure down = a different
generation, and "it looks different" is not "it looks better".

MEASURED IN LATENT SPACE, deliberately: the video latent is already in hand at the call site
(the same tensor DynaShift banks), so there is no VAE decode, no pixel buffer and no new
plumbing. That is a real approximation -- the VAE is nonlinear, so latent high-frequency
energy is not pixel high-frequency energy. It holds for the comparison this is FOR: an A/B
on the same seed where both sides pass through the identical decoder, so the DIRECTION of a
change is trustworthy even where its magnitude is not. Do not quote these as absolute image
metrics; they are only ever meaningful as a ratio between two runs.

Pairing is CONSECUTIVE runs, which is the workflow anyway: generate with the toggle off,
generate again with it on, read the comparison. One previous latent is kept per key; a run
whose latent shape differs from the stored one is not compared (different resolution or
length is not an A/B).
"""
from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

MAX_ROWS = 20  # comparisons kept per key; oldest roll off

# The conditioning fingerprint is floats, so it needs a tolerance rather than equality.
# Below this it is encoder wobble; above it the conditioning actually moved -- which happens
# for reasons other than typing a new prompt (h3_phrase_emphasis is rating-driven and lives
# on the conditioning), so the readout reports the SIZE of the move, not just that there was
# one.
COND_KEY = "conditioning"
COND_TOL = 1e-3


def _cond_shift(a, b):
    """-> relative shift between two conditioning fingerprints. 1.0 means structurally
    different (a different number of entries, or a different token count), which is a real
    prompt/scene change rather than drift."""
    if not isinstance(a, list) or not isinstance(b, list) or not a or not b:
        return 0.0 if a == b else 1.0
    if len(a) != len(b):
        return 1.0
    worst = 0.0
    for ea, eb in zip(a, b):
        if not isinstance(ea, list) or not isinstance(eb, list) or len(ea) != len(eb):
            return 1.0
        if ea[0] != eb[0]:          # token count -- a different prompt, not drift
            return 1.0
        for va, vb in zip(ea[1:], eb[1:]):
            scale = max(abs(va), abs(vb), 1e-6)
            worst = max(worst, abs(va - vb) / scale)
    return worst

_LAPLACIAN = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
_BLUR = torch.ones(1, 1, 5, 5) / 25.0


def _log():
    try:
        from . import funpack_log as fl
    except ImportError:
        import funpack_log as fl
    return fl


def collection_enabled():
    """Rides the block-influence switch -- both are research measurement, opted into from
    the same place, and a second toggle for the same decision is a worse UI, not a safer one."""
    try:
        from . import block_influence as bi
    except ImportError:
        import block_influence as bi
    return bi.collection_enabled()


def state_path(refinement_key):
    try:
        from .conditioning import refinement_state_path
    except ImportError:
        from conditioning import refinement_state_path
    return refinement_state_path(refinement_key, "detail_probe", prefix="refine_v2",
                                 extension="pt")


def _load(refinement_key):
    path = state_path(refinement_key)
    if not os.path.exists(path):
        return {"rows": [], "previous": None}
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001
        return {"rows": [], "previous": None}
    if not isinstance(data, dict):
        return {"rows": [], "previous": None}
    data.setdefault("rows", [])
    data.setdefault("previous", None)
    return data


def _save(refinement_key, data):
    path = state_path(refinement_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, path)


def _planes(latent):
    """Any [..., H, W] video latent -> [N, 1, H, W] float32 planes. Leading dims are pooled
    rather than interpreted: this needs spatial structure, not a channel/time convention, and
    guessing which axis is time is how the LTXAV/H3 axis disagreement bites."""
    x = latent.detach().float()
    if x.dim() < 2 or x.shape[-1] < 3 or x.shape[-2] < 3:
        return None
    return x.reshape(-1, 1, x.shape[-2], x.shape[-1])


def _hf(planes):
    """Per-plane high-frequency magnitude map."""
    k = _LAPLACIAN.view(1, 1, 3, 3).to(planes)
    return F.conv2d(planes, k, padding=1).abs()


def _lf(planes):
    k = _BLUR.to(planes)
    return F.conv2d(planes, k, padding=2)


def _corr(a, b):
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    d = a.norm() * b.norm()
    return float((a @ b / d).item()) if float(d) > 1e-12 else 0.0


def compare(before, after):
    """-> {"detail": ratio, "structure": 0..1, "edge_aligned": -1..1} or None.

    detail       = mean HF energy after / before. >1 means finer structure appeared.
    structure    = 1 - ||LF(after) - LF(before)|| / ||LF(before)||, clamped to 0. Near 1 means
                   the same picture underneath; a low value means the content itself moved.
    edge_aligned = correlation between the HF INCREASE and the HF that was already there.
                   High means the extra detail landed on existing edges (sharpening); near 0
                   means it spread evenly over flat regions too (grain)."""
    pa, pb = _planes(before), _planes(after)
    if pa is None or pb is None or pa.shape != pb.shape:
        return None
    hf_a, hf_b = _hf(pa), _hf(pb)
    base = float(hf_a.mean())
    if not (base > 1e-12):
        return None
    lf_a, lf_b = _lf(pa), _lf(pb)
    denom = float(lf_a.norm())
    structure = 1.0 - float((lf_b - lf_a).norm()) / denom if denom > 1e-12 else 0.0
    return {
        "detail": float(hf_b.mean()) / base,
        "structure": max(0.0, min(1.0, structure)),
        "edge_aligned": _corr(hf_b - hf_a, hf_a),
    }


def record(refinement_key, video_latent, label="", seed=None, settings=None):
    """Score this run against the previous one on the same key, then become the new previous.
    -> the comparison dict (with "label_before"/"label_after"), or None when there was nothing
    to compare against yet.

    `seed` is recorded so the readout can tell the two reasons `structure` drops apart. A
    LOW structure score means the low-frequency picture moved, and that is either "these are
    two different generations, so this is not an A/B at all" (different seed -- R2V pins the
    subject, not the sample) or "same seed, and the change moved the shot rather than its
    detail", which is a real result. Nothing in the three numbers can separate those; the
    seed can, so it is stored rather than guessed at.

    `settings` is this run's scalar widget values. The row keeps the DIFF against the
    previous run's, so it can say what the A/B actually was rather than trusting a label to
    have named the right variable. An empty diff is not a failure: two identical runs measure
    the instrument's own noise floor, which is what makes every other row interpretable.
    (`seed` stays an explicit argument as well -- it is the one field the readout branches on,
    so it should not depend on the settings capture having worked.)"""
    if not refinement_key or not isinstance(video_latent, torch.Tensor):
        return None
    try:
        lat = video_latent.detach()
        if lat.dim() >= 4 and lat.shape[0] == 1:
            lat = lat.squeeze(0)
        lat = lat.to(torch.float16).cpu()
        data = _load(refinement_key)
        prev = data.get("previous")
        row = None
        if isinstance(prev, dict) and isinstance(prev.get("latent"), torch.Tensor):
            row = compare(prev["latent"], lat)
            if row is not None:
                row["label_before"] = str(prev.get("label") or "")
                row["label_after"] = str(label or "")
                prev_set = prev.get("settings") or {}
                cur_set = settings or {}
                keys = (set(prev_set) | set(cur_set)) - {COND_KEY}
                row["changed"] = {k: [prev_set.get(k), cur_set.get(k)] for k in sorted(keys)
                                  if prev_set.get(k) != cur_set.get(k)}
                shift = _cond_shift(prev_set.get(COND_KEY), cur_set.get(COND_KEY))
                row["cond_shift"] = shift
                if shift > COND_TOL:
                    row["changed"][COND_KEY] = [None, round(shift, 6)]
                row["seed_before"] = prev.get("seed")
                row["seed_after"] = seed
                row["same_seed"] = (seed is not None and prev.get("seed") is not None
                                    and int(seed) == int(prev["seed"]))
                row["stamp"] = time.strftime("%Y-%m-%d_%H-%M-%S")
                data["rows"] = (data["rows"] + [row])[-MAX_ROWS:]
        data["previous"] = {"latent": lat, "label": str(label or ""),
                            "seed": None if seed is None else int(seed),
                            "settings": dict(settings or {})}
        _save(refinement_key, data)
        return row
    except Exception as e:  # noqa: BLE001
        _log().failed("H3 detail probe", "record", e, "this run is not scored")
        return None


def rows(refinement_key):
    return _load(refinement_key).get("rows") or []


def clear_all(refinement_key):
    if not refinement_key:
        return
    try:
        os.remove(state_path(refinement_key))
    except FileNotFoundError:
        pass
