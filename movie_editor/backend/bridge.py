"""Bridge to ComfyUI from inside its own process.

Parse/library calls go DIRECT to FunPack functions (no HTTP). Queue/history/view
go over loopback HTTP to ComfyUI's own server so we reuse its prompt validation and
output handling exactly.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from . import config


# ── In-process FunPack calls (no HTTP) ───────────────────────────────────────

def _funpack_imports():
    try:
        from conditioning import parse_timeline_segments
        from templates import apply_prompt_shortcuts, load_transition_db, transition_items
    except ImportError:  # imported as a package submodule
        from ...conditioning import parse_timeline_segments  # type: ignore
        from ...templates import (  # type: ignore
            apply_prompt_shortcuts, load_transition_db, transition_items,
        )
    return parse_timeline_segments, apply_prompt_shortcuts, load_transition_db, transition_items


def parse_timeline(prompt: str, seed: int = 0) -> dict:
    """{anchor, scenes, transitions} — the canonical split Studio will see."""
    parse_timeline_segments, apply_prompt_shortcuts, *_ = _funpack_imports()
    expanded, _applied = apply_prompt_shortcuts(str(prompt or ""), seed=int(seed or 0))
    return parse_timeline_segments(expanded)


def transitions() -> dict:
    _, _, load_transition_db, transition_items = _funpack_imports()
    data = load_transition_db()
    return {"data": data, "transitions": transition_items(data)}


# ── Loopback HTTP to ComfyUI (queue + results) ───────────────────────────────

def _url(path: str) -> str:
    return f"{config.comfy_base_url()}{path}"


async def _session():
    import aiohttp
    return aiohttp.ClientSession()


class ComfyError(RuntimeError):
    pass


async def queue_prompt(graph: dict, client_id: Optional[str] = None) -> dict:
    payload = {"prompt": graph, "client_id": client_id or uuid.uuid4().hex}
    async with await _session() as s:
        async with s.post(_url("/prompt"), json=payload) as r:
            data = await r.json()
            if r.status >= 400 or data.get("node_errors"):
                raise ComfyError(data.get("error") or data.get("node_errors") or f"HTTP {r.status}")
            return data


async def history(prompt_id: str) -> dict:
    async with await _session() as s:
        async with s.get(_url(f"/history/{prompt_id}")) as r:
            r.raise_for_status()
            return await r.json()


async def is_running(prompt_id: str) -> bool:
    async with await _session() as s:
        async with s.get(_url("/queue")) as r:
            r.raise_for_status()
            state = await r.json()
    for bucket in ("queue_running", "queue_pending"):
        for item in state.get(bucket, []):
            if len(item) > 1 and item[1] == prompt_id:
                return True
    return False


async def fetch_view(filename: str, subfolder: str = "", type_: str = "output") -> tuple[bytes, str]:
    from urllib.parse import urlencode
    q = urlencode({"filename": filename, "subfolder": subfolder, "type": type_})
    async with await _session() as s:
        async with s.get(_url(f"/view?{q}")) as r:
            r.raise_for_status()
            return await r.read(), r.headers.get("content-type", "application/octet-stream")
