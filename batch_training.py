"""Batch Training server routes + learning intake.

Serves the Studio batch-rating panel and, on submit, trains the value function on the
batch's (frozen) conditioning with each entry's rating — same cond, N rewards = the
variance-reduction RLHF signal. Batches live in ComfyUI's temp dir (wiped on restart).
"""

import os
import re
import json
import shutil

import torch

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover - only available inside ComfyUI
    web = None
    PromptServer = None


def _safe_key(key):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(key or "default").strip() or "default")


def _batches_root():
    try:
        import folder_paths
        base = folder_paths.get_temp_directory()
    except Exception:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refinements", "tmp")
    return os.path.join(base, "funpack_batches")


def _key_dir(key):
    return os.path.join(_batches_root(), _safe_key(key))


def _latest_batch(key):
    """Most recent batch stamp dir for a key (stamps sort lexically), or None."""
    kd = _key_dir(key)
    if not os.path.isdir(kd):
        return None
    stamps = sorted(
        (d for d in os.listdir(kd) if os.path.isdir(os.path.join(kd, d))),
        reverse=True,
    )
    for stamp in stamps:
        d = os.path.join(kd, stamp)
        if os.path.exists(os.path.join(d, "batch.json")):
            return stamp, d
    return None


def _load_manifest(d):
    with open(os.path.join(d, "batch.json"), "r") as f:
        return json.load(f)


def _rating_labels():
    try:
        try:
            from .conditioning import V2_RATING_LABELS
        except ImportError:
            from conditioning import V2_RATING_LABELS
        return list(V2_RATING_LABELS)
    except Exception:
        return ["Perfect", "Loved it", "Nailed it", "Missing details", "Missing action",
                "Missing quality", "Awful", "-Just forget it-"]


def _profile_for(label):
    try:
        try:
            from .conditioning import normalize_refiner_v2_rating
        except ImportError:
            from conditioning import normalize_refiner_v2_rating
        return normalize_refiner_v2_rating(label)
    except Exception:
        return None


def _train_batch(key, manifest, batch_dir, ratings):
    """Train the value function on (cond, reward) for each rated entry — each entry carries its
    own conditioning (identical across a seed-only batch, distinct per shortcut variant)."""
    try:
        try:
            from .value_function import OnlineValueFunction
            from .conditioning import refinement_state_path, tensor_to_serializable, FunPackVideoRefinerV2
        except ImportError:
            from value_function import OnlineValueFunction
            from conditioning import refinement_state_path, tensor_to_serializable, FunPackVideoRefinerV2
    except Exception as e:
        return 0, f"learning modules unavailable: {e}"

    vf_path = refinement_state_path(key, "value_fn", prefix="refine_v2", extension="pt")
    trained = 0
    rated_items = []   # for Phase 3b axis/direction learning
    with torch.inference_mode(False):
        vf = None
        for item in manifest.get("items", []):
            label = ratings.get(str(item.get("id")))
            if not label:
                continue
            profile = _profile_for(label)
            if not profile or profile.get("skip_learning"):
                continue  # '-Just forget it-' and unknowns: skip
            cond_file = item.get("cond")
            if not cond_file:
                continue
            cond_path = os.path.join(batch_dir, cond_file)
            if not os.path.exists(cond_path):
                continue
            cond = torch.load(cond_path, map_location="cpu")
            if not isinstance(cond, torch.Tensor):
                continue
            if vf is None:
                vf = OnlineValueFunction.load_or_create(vf_path, hidden_dim=int(cond.shape[-1]))
                if vf is None:
                    return 0, "could not create value function"
            reward = float(profile.get("reward", 0.0))
            vf.train_on(cond.clone(), reward)            # 3a: value function
            try:
                rated_items.append({"rating": label, "conditioning": tensor_to_serializable(cond)})
            except Exception:
                pass
            trained += 1
        if trained and vf is not None:
            vf.save(vf_path)
    if trained == 0:
        return 0, "no rated entries with saved conditioning"
    # 3b: feed the per-axis direction memory + persistent repair bias (axes consistently rated
    # missing/wrong across the frozen batch = high-confidence signal for the next generation).
    axis_note = None
    try:
        axis_note = FunPackVideoRefinerV2()._v2_learn_from_batch(key, rated_items)
        print(f"[FunPack batch] {axis_note}")
    except Exception as e:
        axis_note = f"axis learning skipped: {e}"
    return trained, axis_note


if PromptServer is not None and web is not None:

    @PromptServer.instance.routes.get("/funpack/batch/list")
    async def funpack_batch_list(request):
        key = request.rel_url.query.get("key", "")
        if not str(key).strip():
            return web.json_response({"found": False, "reason": "no key", "labels": _rating_labels()})
        latest = _latest_batch(key)
        if latest is None:
            return web.json_response({"found": False, "labels": _rating_labels()})
        stamp, d = latest
        try:
            manifest = _load_manifest(d)
        except Exception as e:
            return web.json_response({"found": False, "reason": f"manifest read failed: {e}",
                                      "labels": _rating_labels()})
        subfolder = manifest.get("subfolder") or f"funpack_batches/{_safe_key(key)}/{stamp}"
        items = [{
            "id": it.get("id"),
            "index": it.get("index"),
            "seed": it.get("seed"),
            "preview": it.get("preview"),
            "variant": it.get("variant"),
            "prompt": it.get("prompt"),
            "rating": it.get("rating"),
        } for it in manifest.get("items", [])]
        return web.json_response({
            "found": True,
            "key": key,
            "stamp": stamp,
            "subfolder": subfolder,
            "created": manifest.get("created"),
            "scene_prompts": manifest.get("scene_prompts", []),
            "items": items,
            "labels": _rating_labels(),
        })

    @PromptServer.instance.routes.post("/funpack/batch/submit")
    async def funpack_batch_submit(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        key = str(body.get("key", "")).strip()
        stamp = str(body.get("stamp", "")).strip()
        ratings = body.get("ratings", {}) or {}
        if not key or not stamp:
            return web.json_response({"error": "key and stamp required"}, status=400)
        batch_dir = os.path.join(_key_dir(key), stamp)
        if not os.path.exists(os.path.join(batch_dir, "batch.json")):
            return web.json_response({"error": "batch not found"}, status=404)
        manifest = _load_manifest(batch_dir)
        trained, note = _train_batch(key, manifest, batch_dir, {str(k): v for k, v in ratings.items()})
        # Submit clears the batch regardless (rated material is consumed).
        try:
            shutil.rmtree(batch_dir, ignore_errors=True)
        except Exception:
            pass
        if trained == 0:
            return web.json_response({"ok": True, "trained": 0, "warning": note or "nothing trained"})
        return web.json_response({"ok": True, "trained": trained, "info": note})

    @PromptServer.instance.routes.post("/funpack/batch/forget")
    async def funpack_batch_forget(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        key = str(body.get("key", "")).strip()
        stamp = str(body.get("stamp", "")).strip()
        if not key or not stamp:
            return web.json_response({"error": "key and stamp required"}, status=400)
        batch_dir = os.path.join(_key_dir(key), stamp)
        shutil.rmtree(batch_dir, ignore_errors=True)
        return web.json_response({"ok": True})
