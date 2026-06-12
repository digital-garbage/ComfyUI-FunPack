"""Detect missing custom-node packs for the built-in FunPack pipeline and install
them through ComfyUI-Manager's queue (same path as the Manager UI)."""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Optional

from . import config
from .builder import CORE

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

_install_jobs: dict[str, dict] = {}


def required_core_classes() -> list[str]:
    return sorted(set(CORE.values()))


def missing_core_classes(object_info: dict | None) -> list[str]:
    oi = object_info or {}
    return sorted(cls for cls in CORE.values() if cls not in oi)


def missing_packs(object_info: dict | None) -> list[dict]:
    missing = set(missing_core_classes(object_info))
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


def unmapped_missing_classes(object_info: dict | None) -> list[str]:
    missing = set(missing_core_classes(object_info))
    return sorted(missing - _ALL_MAPPED)


def status_payload(object_info: dict | None, *, manager_available: bool) -> dict:
    packs = missing_packs(object_info)
    unmapped = unmapped_missing_classes(object_info)
    return {
        "manager_available": manager_available,
        "missing_classes": missing_core_classes(object_info),
        "missing_packs": packs,
        "unmapped_classes": unmapped,
        "needs_install": bool(packs),
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


def create_install_job(pack_ids: list[str]) -> dict:
    packs = []
    for pid in pack_ids:
        spec = _PACK_BY_ID.get(pid)
        if spec:
            packs.append({"id": pid, "title": spec["title"]})
    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "state": "queued",
        "packs": packs,
        "total": len(packs),
        "done": 0,
        "current_title": packs[0]["title"] if packs else "",
        "error": "",
        "started_at": time.time(),
    }
    _install_jobs[job_id] = job
    return job


def get_install_job(job_id: str) -> Optional[dict]:
    job = _install_jobs.get(job_id)
    if not job:
        return None
    return {
        "job_id": job["job_id"],
        "state": job["state"],
        "packs": job["packs"],
        "total": job["total"],
        "done": job["done"],
        "current_title": job.get("current_title") or "",
        "error": job.get("error") or "",
    }


async def run_install_job(job_id: str, restart_fn) -> None:
    job = _install_jobs.get(job_id)
    if not job:
        return
    job["state"] = "installing"
    try:
        for i, pack in enumerate(job["packs"]):
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
        deadline = time.time() + 1800
        while time.time() < deadline:
            st = await _manager_queue_status()
            done = int(st.get("done_count") or 0)
            total = int(st.get("total_count") or 0) or job["total"]
            processing = bool(st.get("is_processing"))
            job["total"] = max(job["total"], total)
            job["done"] = min(done, job["total"])
            if job["packs"]:
                idx = min(max(done, 0), len(job["packs"]) - 1)
                job["current_title"] = job["packs"][idx]["title"]
            if job["total"] > 0 and done >= job["total"] and not processing:
                break
            await asyncio.sleep(1.5)
        else:
            job["state"] = "error"
            job["error"] = "Install timed out. Check ComfyUI-Manager queue in the main ComfyUI window."
            return
        job["done"] = job["total"]
        job["state"] = "restarting"
        job["current_title"] = ""
        await asyncio.sleep(0.7)
        restart_fn()
    except Exception as e:  # noqa: BLE001
        job["state"] = "error"
        job["error"] = str(e) or "Install failed."
