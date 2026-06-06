"""Async client for a running ComfyUI: native API + FunPack /funpack/* routes.

The sidecar stays light — no torch/comfy. Everything that needs the model goes
through ComfyUI over HTTP.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

import httpx

from . import config


class ComfyError(RuntimeError):
    pass


def _url(path: str) -> str:
    return f"{config.COMFY_URL}{path}"


async def _get_json(path: str, params: Optional[dict] = None) -> Any:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(_url(path), params=params)
        r.raise_for_status()
        return r.json()


async def _post_json(path: str, payload: dict) -> Any:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(_url(path), json=payload)
        r.raise_for_status()
        return r.json()


# ── FunPack routes (parse/library; reuse Studio's logic server-side) ──────────

async def parse_timeline(prompt: str, seed: int = 0) -> dict:
    """{anchor, scenes, transitions} — the canonical split Studio will see."""
    return await _post_json("/funpack/parse_timeline", {"prompt": prompt, "seed": seed})


async def transitions() -> dict:
    return await _get_json("/funpack/transitions")


async def scenes() -> dict:
    return await _get_json("/funpack/scenes")


async def shortcuts() -> dict:
    return await _get_json("/funpack/shortcuts")


# ── Native ComfyUI API (queue + results) ─────────────────────────────────────

async def queue_prompt(graph: dict, client_id: Optional[str] = None) -> dict:
    """POST an API-format graph to /prompt. Returns {prompt_id, number, node_errors}."""
    payload = {"prompt": graph, "client_id": client_id or uuid.uuid4().hex}
    result = await _post_json("/prompt", payload)
    if result.get("node_errors"):
        raise ComfyError(f"ComfyUI rejected the graph: {result['node_errors']}")
    return result


async def history(prompt_id: str) -> dict:
    """/history/{id} — empty dict until the prompt completes."""
    return await _get_json(f"/history/{prompt_id}")


async def queue_state() -> dict:
    return await _get_json("/queue")


async def is_running(prompt_id: str) -> bool:
    state = await queue_state()
    for bucket in ("queue_running", "queue_pending"):
        for item in state.get(bucket, []):
            # item is [number, prompt_id, prompt, extra, outputs]
            if len(item) > 1 and item[1] == prompt_id:
                return True
    return False


def view_url(filename: str, subfolder: str = "", type_: str = "output") -> str:
    """Absolute URL for an output file, so the frontend can stream it via our proxy."""
    from urllib.parse import urlencode
    q = urlencode({"filename": filename, "subfolder": subfolder, "type": type_})
    return _url(f"/view?{q}")


async def fetch_view(filename: str, subfolder: str = "", type_: str = "output") -> tuple[bytes, str]:
    """Download an output file (bytes, content-type) for proxying to the browser."""
    from urllib.parse import urlencode
    q = urlencode({"filename": filename, "subfolder": subfolder, "type": type_})
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.get(_url(f"/view?{q}"))
        r.raise_for_status()
        return r.content, r.headers.get("content-type", "application/octet-stream")
