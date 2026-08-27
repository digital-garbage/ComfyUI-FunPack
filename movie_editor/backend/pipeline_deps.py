"""Detect missing custom-node packs for the built-in FunPack pipeline and install
them through ComfyUI-Manager's queue (same path as the Manager UI)."""
from __future__ import annotations

import asyncio
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from . import config
from .builder import CORE, family_core

MANAGER_FOLDER = "ComfyUI-Manager"
MANAGER_GIT_URL = "https://github.com/ComfyUI-Manager/ComfyUI-Manager"
MANAGER_TITLE = "ComfyUI-Manager"
STALE_PROGRESS_SEC = 90
POLL_INTERVAL_SEC = 1.5

# Curated mapping: pack id (ComfyUI-Manager / registry) -> node classes used by the
# fixed core graph. FunPack and comfy-core primitives are excluded.
PIPELINE_PACKS: list[dict[str, Any]] = [
    {
        "id": "ComfyUI-LTXVideo",
        "title": "ComfyUI-LTXVideo",
        "classes": frozenset({
            "LTXVConditioning",
            "LTXVConcatAVLatent",
            "LTXVSeparateAVLatent",
            "LTXVAudioVAEDecode",
            "LTXFloatToInt",
        }),
        "git_urls": ["https://github.com/Lightricks/ComfyUI-LTXVideo"],
    },
    {
        "id": "comfyui-videohelpersuite",
        "title": "Video Helper Suite",
        "classes": frozenset({"VHS_VideoCombine"}),
        "git_urls": ["https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"],
    },
    {
        "id": "comfyui-kjnodes",
        "title": "KJNodes",
        "classes": frozenset({"NormalizeAudioLoudness"}),
        "git_urls": ["https://github.com/kijai/ComfyUI-KJNodes"],
    },
]

_PACK_BY_ID = {p["id"]: p for p in PIPELINE_PACKS}
_ALL_MAPPED = frozenset().union(*(p["classes"] for p in PIPELINE_PACKS))


# ── per-family setup: what a project needs before it can generate ─────────────
# Declarative on purpose: both families are real pipelines here, and a project picks one.
# MiniMax H3 shipped in ComfyUI v0.30.0 (PR #15224) with weights on Comfy-Org/MiniMax-H3,
# so its entries below are the real files, not placeholders.
#
# `nodes` are slot node classes (not core graph nodes — those are covered by PIPELINE_PACKS).
# `models` are files the user has to download; `folder` is the ComfyUI models/ subdirectory.
FAMILY_SETUP: dict[str, dict[str, Any]] = {
    "ltxav": {
        "label": "LTX2 / LTX2.3 / LTX2.5",
        "released": True,
        "summary": "Lightricks LTX-2, every point release. Gemma3 text encoder (Gemma4 on 2.5), separate video and audio latents.",
        "nodes": [],
        "models": [
            {"role": "unet", "label": "LTX-2 diffusion model", "folder": "diffusion_models"},
            {"role": "clip", "label": "Gemma3 text encoder (Gemma4 on 2.5)", "folder": "text_encoders"},
            {"role": "video_vae", "label": "LTX-2 video VAE", "folder": "vae"},
            {"role": "audio_vae", "label": "LTX-2 audio VAE", "folder": "vae"},
        ],
    },
    "minimax_h3": {
        "label": "MiniMax H3 (Hailuo)",
        "released": True,
        "summary": "MiniMax H3. Qwen3-VL text encoder, one joint AV latent, 24 fps, "
                   "native reference conditioning (ref2va). Needs ComfyUI v0.30.0 or newer.",
        "note":
            "MiniMax H3 ships as TWO diffusion checkpoints and they are not interchangeable: "
            "minimax_h3_fl2va_* is the text-to-video / first-last-frame model (it is the one "
            "that uses a scene's anchor image), minimax_h3_ref2va_* is the reference model "
            "(<Picture 1> / <Video 1> / <Audio 1> media). Load the one that matches how you "
            "generate — the other reads the same graph without complaining and just conditions "
            "badly. Everything else (text encoder, both VAEs) is shared.",
        "source_url": "https://huggingface.co/Comfy-Org/MiniMax-H3",
        "source_title": "Comfy-Org/MiniMax-H3 — repackaged weights",
        "nodes": [
            {"class": "EmptyMiniMaxH3LatentAV", "label": "Empty MiniMax H3 AV Latent",
             "role": "empty_latent",
             "why": "Makes the joint video + audio latent that feeds the Chain Sampler. Its "
                    "`length` must match the project's frames per scene (both snap to 17k+5)."},
            {"class": "MiniMaxH3SigmaShift", "label": "MiniMax H3 Sigma Shift",
             "role": None, "optional": True,
             "why": "Optional — sets the video/audio flow shifts (defaults 12.0 / 3.0). It also "
                    "tells the DiT which shift the schedule uses, which is how the audio stream "
                    "stays on its own clock."},
        ],
        "models": [
            {"role": "unet", "label": "MiniMax H3 diffusion model (fl2va OR ref2va)",
             "folder": "diffusion_models",
             "hint": "minimax_h3_fl2va_pruned_int8_convrot.safetensors (~21 GB) for t2v/i2v, or "
                     "minimax_h3_ref2va_pruned_int8_convrot.safetensors for reference media. "
                     "_int8_convrot (~34 GB) and _bf16 (~66 GB) are the higher-precision cuts"},
            {"role": "clip", "label": "Qwen3-VL-32B text encoder (50 layers)", "folder": "text_encoders",
             "hint": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors (~16 GB); _int8_convrot and "
                     "_bf16 also published. H3 consumes the unnormalized hidden state after layer 50"},
            {"role": "video_vae", "label": "MiniMax H3 video VAE", "folder": "vae",
             "hint": "minimax_h3_video_vae_fp16.safetensors — 24-channel latent, 16x spatial"},
            {"role": "audio_vae", "label": "MiniMax H3 audio VAE", "folder": "vae",
             "hint": "minimax_h3_audio_vae_fp32.safetensors — 32 kHz stereo; also encodes "
                     "audio references for ref2va"},
        ],
    },
}

DEFAULT_FAMILY = "ltxav"


def family_setup(family: str | None) -> dict:
    key = str(family or DEFAULT_FAMILY).strip().lower()
    return FAMILY_SETUP.get(key) or FAMILY_SETUP[DEFAULT_FAMILY]


def families_payload() -> list[dict]:
    """The setup picker's options, in display order."""
    return [
        {"key": key, "label": spec["label"], "released": bool(spec.get("released")),
         "summary": spec.get("summary", ""),
         "note": spec.get("note"),
         "source_url": spec.get("source_url"), "source_title": spec.get("source_title")}
        for key, spec in FAMILY_SETUP.items()
    ]


def family_readiness(object_info: dict | None, family: str) -> dict:
    """What this family still needs — nodes not installed, models not downloaded.

    Model files cannot be checked from object_info alone (a loader lists them as combo
    choices only once the node exists), so those are reported as EXPECTED rather than
    missing: the honest statement is "here is what you will need", not a false negative.
    """
    oi = object_info or {}
    spec = family_setup(family)
    nodes = []
    for n in spec.get("nodes", []):
        nodes.append({**n, "installed": n["class"] in oi})
    return {
        "family": str(family or DEFAULT_FAMILY).lower(),
        "label": spec["label"],
        "released": bool(spec.get("released")),
        "summary": spec.get("summary", ""),
        "note": spec.get("note"),
        "source_url": spec.get("source_url"),
        "source_title": spec.get("source_title"),
        "nodes": nodes,
        "missing_nodes": [n for n in nodes if not n["installed"] and not n.get("optional")],
        "models": spec.get("models", []),
    }

_install_jobs: dict[str, dict] = {}


def custom_nodes_dir() -> Path:
    """ComfyUI ``custom_nodes`` directory (FunPack lives one level below)."""
    try:
        import folder_paths
        paths = folder_paths.get_folder_paths("custom_nodes")
        if paths:
            return Path(paths[0])
    except Exception:
        pass
    return Path(__file__).resolve().parents[2].parent


def manager_dir() -> Path:
    return custom_nodes_dir() / MANAGER_FOLDER


def manager_dir_on_disk() -> bool:
    d = manager_dir()
    return d.is_dir() and ((d / ".git").exists() or (d / "__init__.py").exists())


def required_core_classes(family: str | None = None) -> list[str]:
    return sorted(set(_core_for(family).values()))


def _core_for(family: str | None) -> dict:
    """The fixed-core node classes for `family` (H3 drops three of LTXAV's)."""
    if not family:
        return CORE
    return family_core(str(family).strip().lower())[0]


def missing_core_classes(object_info: dict | None, family: str | None = None) -> list[str]:
    oi = object_info or {}
    return sorted(cls for cls in _core_for(family).values() if cls not in oi)


def missing_packs(object_info: dict | None, family: str | None = None) -> list[dict]:
    missing = set(missing_core_classes(object_info, family))
    out: list[dict] = []
    for pack in PIPELINE_PACKS:
        hit = pack["classes"] & missing
        if not hit:
            continue
        out.append({
            "id": pack["id"],
            "title": pack["title"],
            "missing_classes": sorted(hit),
        })
    return out


def unmapped_missing_classes(object_info: dict | None, family: str | None = None) -> list[str]:
    missing = set(missing_core_classes(object_info, family))
    return sorted(missing - _ALL_MAPPED)


def status_payload(
    object_info: dict | None,
    *,
    manager_available: bool,
    manager_on_disk: bool | None = None,
    family: str | None = None,
) -> dict:
    packs = missing_packs(object_info, family)
    unmapped = unmapped_missing_classes(object_info, family)
    on_disk = manager_dir_on_disk() if manager_on_disk is None else manager_on_disk
    readiness = family_readiness(object_info, family or DEFAULT_FAMILY)
    # A family whose own nodes are missing (an older ComfyUI, say) is a setup problem too —
    # the modal has to open and say so, rather than reporting a clean pipeline that cannot
    # actually generate.
    needs_setup = bool(packs) or bool(readiness["missing_nodes"])
    needs_manager = bool(packs) and not manager_available
    return {
        "family": readiness["family"],
        "families": families_payload(),
        "readiness": readiness,
        "manager_available": manager_available,
        "manager_on_disk": on_disk,
        "needs_manager_install": needs_manager and not on_disk,
        "needs_manager_restart": needs_manager and on_disk,
        "missing_classes": missing_core_classes(object_info, family),
        "missing_packs": packs,
        "unmapped_classes": unmapped,
        "needs_install": needs_setup,
        "needs_setup": needs_setup,
        "manager_git_url": MANAGER_GIT_URL,
        "manual_urls": [
            {"id": p["id"], "title": p["title"], "url": p["git_urls"][0]}
            for p in PIPELINE_PACKS
            if p["id"] in {x["id"] for x in packs}
        ],
    }


async def manager_available() -> bool:
    status, _ = await _manager_request("GET", "/manager/version")
    if status == 200:
        return True
    status, _ = await _manager_request("GET", "/api/manager/version")
    return status == 200


async def _manager_request(method: str, path: str, *, json_body: dict | None = None, text_body: str | None = None):
    import aiohttp

    base = config.comfy_base_url().rstrip("/")
    url = f"{base}{path}"
    headers: dict[str, str] = {}
    kwargs: dict[str, Any] = {}
    if json_body is not None:
        headers["Content-Type"] = "application/json"
        kwargs["json"] = json_body
    elif text_body is not None:
        headers["Content-Type"] = "text/plain"
        kwargs["data"] = text_body
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.request(method, url, headers=headers, **kwargs) as resp:
            text = await resp.text()
            return resp.status, text


async def _fetch_manager_pack_meta(pack_id: str) -> dict | None:
    for path in ("/customnode/getlist?mode=cache", "/api/customnode/getlist?mode=cache"):
        status, body = await _manager_request("GET", path)
        if status != 200:
            continue
        try:
            import json as _json
            data = _json.loads(body)
        except Exception:
            continue
        node_packs = data.get("node_packs") or {}
        for _key, meta in node_packs.items():
            if not isinstance(meta, dict):
                continue
            if meta.get("id") == pack_id or meta.get("cnr_id") == pack_id or _key == pack_id:
                return meta
    return None


def _install_body(pack: dict, meta: dict | None) -> dict:
    pid = pack["id"]
    title = pack["title"]
    if meta:
        version = str(meta.get("version") or meta.get("ver") or "nightly")
        selected = str(meta.get("selected_version") or "latest")
        body: dict[str, Any] = {
            "id": meta.get("id") or meta.get("cnr_id") or pid,
            "title": meta.get("title") or title,
            "name": meta.get("name") or title,
            "version": version,
            "selected_version": selected,
            "channel": str(meta.get("channel") or "default"),
            "mode": "cache",
            "ui_id": pid,
            "skip_post_install": False,
        }
        if meta.get("repository"):
            body["repository"] = meta["repository"]
        if meta.get("files"):
            body["files"] = meta["files"]
        return body
    return {
        "id": pid,
        "title": title,
        "name": title,
        "version": "unknown",
        "selected_version": "unknown",
        "files": list(pack.get("git_urls") or []),
        "install_type": "git-clone",
        "channel": "default",
        "mode": "cache",
        "ui_id": pid,
        "skip_post_install": False,
    }


async def _queue_install(body: dict) -> tuple[bool, str]:
    for path in ("/manager/queue/install", "/api/manager/queue/install"):
        status, text = await _manager_request("POST", path, json_body=body)
        if status in (200, 201):
            return True, ""
        if status == 403:
            return False, "ComfyUI-Manager blocked the install (security policy). Open Manager and lower the security level, or install the packs manually."
        if status == 404:
            return False, text.strip() or "Pack not found in ComfyUI-Manager database."
    return False, "ComfyUI-Manager install endpoint unavailable."


async def _start_manager_queue() -> None:
    for path in ("/manager/queue/start", "/api/manager/queue/start"):
        status, _ = await _manager_request("GET", path)
        if status in (200, 201):
            return


async def _reset_manager_queue() -> None:
    for path in ("/manager/queue/reset", "/api/manager/queue/reset"):
        await _manager_request("POST", path)


async def _manager_queue_status() -> dict:
    for path in ("/manager/queue/status", "/api/manager/queue/status"):
        status, body = await _manager_request("GET", path)
        if status != 200:
            continue
        try:
            import json as _json
            data = _json.loads(body)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def install_manager_sync() -> tuple[bool, str]:
    """Clone ComfyUI-Manager into custom_nodes (no Manager HTTP required)."""
    import shutil

    if manager_dir_on_disk():
        return True, "already_installed"
    if not shutil.which("git"):
        return False, "git not found on PATH."
    root = custom_nodes_dir()
    root.mkdir(parents=True, exist_ok=True)
    target = manager_dir()
    proc = subprocess.run(
        ["git", "clone", MANAGER_GIT_URL, str(target)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "git clone failed").strip()
        return False, err
    req = target / "requirements.txt"
    if req.is_file():
        pip = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if pip.returncode != 0:
            err = (pip.stderr or pip.stdout or "pip install failed").strip()
            return False, f"ComfyUI-Manager cloned but pip install failed: {err}"
    return True, ""


def _new_job(*, kind: str, packs: list[dict]) -> dict:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "kind": kind,
        "state": "queued",
        "packs": packs,
        "total": max(1, len(packs)),
        "done": 0,
        "current_title": packs[0]["title"] if packs else "",
        "error": "",
        "stale": False,
        "cancel_requested": False,
        "started_at": time.time(),
        "last_progress_at": time.time(),
    }
    _install_jobs[job_id] = job
    return job


def create_manager_install_job() -> dict:
    return _new_job(
        kind="manager",
        packs=[{"id": MANAGER_FOLDER, "title": MANAGER_TITLE}],
    )


def create_install_job(pack_ids: list[str]) -> dict:
    packs = []
    for pid in pack_ids:
        spec = _PACK_BY_ID.get(pid)
        if spec:
            packs.append({"id": pid, "title": spec["title"]})
    return _new_job(kind="packs", packs=packs)


def cancel_install_job(job_id: str) -> bool:
    job = _install_jobs.get(job_id)
    if not job:
        return False
    if job["state"] in ("restarting", "cancelled", "done"):
        return False
    job["cancel_requested"] = True
    return True


def get_install_job(job_id: str) -> Optional[dict]:
    job = _install_jobs.get(job_id)
    if not job:
        return None
    return {
        "job_id": job["job_id"],
        "kind": job.get("kind") or "packs",
        "state": job["state"],
        "packs": job["packs"],
        "total": job["total"],
        "done": job["done"],
        "current_title": job.get("current_title") or "",
        "error": job.get("error") or "",
        "stale": bool(job.get("stale")),
    }




async def _sleep_poll(job: dict) -> bool:
    """Return True if cancel was requested during the wait."""
    steps = max(1, int(POLL_INTERVAL_SEC / 0.25))
    for _ in range(steps):
        if job.get("cancel_requested"):
            return True
        await asyncio.sleep(0.25)
    return bool(job.get("cancel_requested"))


async def _finish_cancelled(job: dict) -> None:
    job["state"] = "cancelled"
    job["current_title"] = ""
    if job.get("kind") == "packs":
        await _reset_manager_queue()


def _touch_progress(job: dict, done: int) -> None:
    now = time.time()
    if done != job.get("_last_done"):
        job["_last_done"] = done
        job["last_progress_at"] = now
        job["stale"] = False
    elif now - float(job.get("last_progress_at") or now) >= STALE_PROGRESS_SEC:
        job["stale"] = True


async def _poll_manager_queue(job: dict, restart_fn: Callable[[], None]) -> None:
    deadline = time.time() + 1800
    while time.time() < deadline:
        if job.get("cancel_requested"):
            await _finish_cancelled(job)
            return
        st = await _manager_queue_status()
        done = int(st.get("done_count") or 0)
        total = int(st.get("total_count") or 0) or job["total"]
        processing = bool(st.get("is_processing"))
        job["total"] = max(job["total"], total)
        job["done"] = min(done, job["total"])
        _touch_progress(job, done)
        if job["packs"]:
            idx = min(max(done, 0), len(job["packs"]) - 1)
            job["current_title"] = job["packs"][idx]["title"]
        if job["total"] > 0 and done >= job["total"] and not processing:
            break
        if await _sleep_poll(job):
            await _finish_cancelled(job)
            return
    else:
        job["state"] = "error"
        job["error"] = "Install timed out. Cancel and try again, or check ComfyUI-Manager in the main ComfyUI window."
        job["stale"] = True
        return
    job["done"] = job["total"]
    job["state"] = "restarting"
    job["current_title"] = ""
    await asyncio.sleep(0.7)
    restart_fn()


async def run_manager_install_job(job_id: str, restart_fn: Callable[[], None]) -> None:
    job = _install_jobs.get(job_id)
    if not job:
        return
    job["state"] = "installing"
    job["current_title"] = MANAGER_TITLE
    try:
        if job.get("cancel_requested"):
            await _finish_cancelled(job)
            return
        ok, err = await asyncio.to_thread(install_manager_sync)
        if job.get("cancel_requested"):
            await _finish_cancelled(job)
            return
        if not ok:
            job["state"] = "error"
            job["error"] = err or "ComfyUI-Manager install failed."
            return
        job["done"] = 1
        job["state"] = "restarting"
        job["current_title"] = ""
        await asyncio.sleep(0.7)
        restart_fn()
    except Exception as e:  # noqa: BLE001
        job["state"] = "error"
        job["error"] = str(e) or "ComfyUI-Manager install failed."


async def run_install_job(job_id: str, restart_fn: Callable[[], None]) -> None:
    job = _install_jobs.get(job_id)
    if not job:
        return
    job["state"] = "installing"
    try:
        for i, pack in enumerate(job["packs"]):
            if job.get("cancel_requested"):
                await _finish_cancelled(job)
                return
            pid = pack["id"]
            spec = _PACK_BY_ID.get(pid)
            if not spec:
                continue
            job["current_title"] = pack["title"]
            job["done"] = i
            meta = await _fetch_manager_pack_meta(pid)
            body = _install_body(spec, meta)
            ok, err = await _queue_install(body)
            if not ok:
                job["state"] = "error"
                job["error"] = err or f"Failed to queue {pack['title']}."
                return
        await _start_manager_queue()
        await _poll_manager_queue(job, restart_fn)
    except Exception as e:  # noqa: BLE001
        job["state"] = "error"
        job["error"] = str(e) or "Install failed."
