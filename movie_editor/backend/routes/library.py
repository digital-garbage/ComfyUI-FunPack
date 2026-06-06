"""Proxy the FunPack libraries (transitions/scenes/shortcuts) for editor palettes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import comfy_client

router = APIRouter(prefix="/api/library", tags=["library"])


async def _safe(coro, what: str):
    try:
        return await coro
    except Exception as e:  # ComfyUI down / route missing -> empty, not fatal for editing
        raise HTTPException(502, f"ComfyUI {what} unavailable: {e}")


@router.get("/transitions")
async def transitions():
    return await _safe(comfy_client.transitions(), "transitions")


@router.get("/scenes")
async def scenes():
    return await _safe(comfy_client.scenes(), "scenes")


@router.get("/shortcuts")
async def shortcuts():
    return await _safe(comfy_client.shortcuts(), "shortcuts")
