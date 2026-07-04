"""DynaShift negative latent memory.

Per-refinement-key store of generations the user rated as containing something
UNWANTED (an extra person, wrong appearance, ...). The Scene Chain sampler saves the
final video latent of every keyed run as a PENDING entry; the rating that scores that
run either promotes it into the bank (negative ratings) or discards it. At sampling
time the bank acts as a latent-space negative prompt at CFG=1: see
samplers._build_dynashift_wrapper for the steering math.

This is the latent-space sibling of the Refiner's conditioning-space `bad_dir`:
bad_dir repels the INPUT (what we ask for), the negative bank repels the OUTPUT
(what actually appeared). Files ride the refinement-key sidecar convention
(<key>.negative_latents.pt / <key>.pending_negative.pt), so delete_refinement_key's
stem glob removes them atomically with the key - no orphaned-sidecar drift.
"""

import os
import time

import torch

# Bank size: enough to cover the recurring failure modes of one key without turning
# the store into a museum. Oldest entries roll off; keys are disposable anyway.
MAX_NEGATIVES = 8

# V2 rating profile keys whose semantics are "something appeared that I did not want".
# Missing-* ratings are absences, not intrusions - steering away from an absence would
# push toward arbitrary content, so they never feed the bank.
NEGATIVE_RATING_KEYS = ("awful", "wrong_appearance")


def _state_path(refinement_key, mode):
    try:
        from .conditioning import refinement_state_path
    except ImportError:
        from conditioning import refinement_state_path
    return refinement_state_path(refinement_key, mode, prefix="refine_v2", extension="pt")


def _atomic_save(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # a concurrent read never sees a partial write


def save_pending(refinement_key, video_latent, conditioning=None):
    """Persist this run's final video latent (+ mean-pooled conditioning) as the
    pending candidate the NEXT rating will judge. fp16 CPU, batch squeezed; one
    pending per key (a new run overwrites - each rating scores its own run)."""
    if not refinement_key or not isinstance(video_latent, torch.Tensor):
        return False
    try:
        lat = video_latent.detach()
        if lat.dim() >= 4 and lat.shape[0] == 1:
            lat = lat.squeeze(0)
        cond_vec = None
        if isinstance(conditioning, torch.Tensor):
            c = conditioning.detach().float()
            if c.dim() == 3:
                c = c.squeeze(0)
            if c.dim() == 2:
                c = c.mean(dim=0)
            cond_vec = c.to(torch.float16).cpu()
        _atomic_save({
            "latent": lat.to(torch.float16).cpu(),
            "cond": cond_vec,
            "stamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        }, _state_path(refinement_key, "pending_negative"))
        return True
    except Exception as e:
        print(f"[FunPack DynaShift] pending negative save failed: {e}")
        return False


def consume_pending(refinement_key, promote):
    """Pair the pending latent with the rating that scores its run: promote it into
    the negative bank (ring buffer) or discard it. Always removes the pending file so
    a stale candidate can never be promoted by a later, unrelated rating. Returns the
    bank size after promotion, or None when nothing was promoted."""
    if not refinement_key:
        return None
    pending_path = _state_path(refinement_key, "pending_negative")
    if not os.path.exists(pending_path):
        return None
    count = None
    try:
        if promote:
            pending = torch.load(pending_path, map_location="cpu", weights_only=False)
            if isinstance(pending, dict) and isinstance(pending.get("latent"), torch.Tensor):
                store_path = _state_path(refinement_key, "negative_latents")
                entries = load_negatives(refinement_key)
                entries.append(pending)
                entries = entries[-MAX_NEGATIVES:]
                _atomic_save({"version": 1, "entries": entries}, store_path)
                count = len(entries)
    except Exception as e:
        print(f"[FunPack DynaShift] negative bank update failed: {e}")
    finally:
        try:
            os.remove(pending_path)
        except OSError:
            pass
    return count


def load_negatives(refinement_key):
    """Bank entries ([{latent: fp16 [C,T,H,W], cond: fp16 [D]|None, stamp}], oldest
    first), or [] when the key has no bank yet."""
    if not refinement_key:
        return []
    path = _state_path(refinement_key, "negative_latents")
    if not os.path.exists(path):
        return []
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        entries = data.get("entries") if isinstance(data, dict) else None
        return [e for e in (entries or [])
                if isinstance(e, dict) and isinstance(e.get("latent"), torch.Tensor)]
    except Exception as e:
        print(f"[FunPack DynaShift] negative bank load failed: {e}")
        return []
