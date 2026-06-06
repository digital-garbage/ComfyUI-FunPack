"""Generate: assemble prompt -> fill template -> queue ComfyUI -> track -> serve result.

V1 drives the EXISTING uniform chain in ONE pass: the whole timeline (minus excluded
scenes) generates together. Per-scene "generate only this" / true selective regen is
Phase 3 — it needs the per-scene latent store. We still accept the request shape now so
the frontend is stable; `only_scene` falls back to a single-scene combined prompt.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Optional

from .. import comfy_client, projects, workflow
from ..timeline import Project, build_combined_prompt

router = APIRouter(prefix="/api/projects", tags=["generate"])


class GenerateBody(BaseModel):
    # Phase-3 forward-compat: a single scene id to generate in isolation. In V1 this
    # builds a one-scene prompt (anchor + that scene) and runs the chain on it.
    only_scene: Optional[str] = None


def _media_from_history(hist_entry: dict) -> list[dict]:
    """Pull output files (videos/gifs/images) from a /history entry's outputs."""
    out: list[dict] = []
    outputs = hist_entry.get("outputs", {})
    for node_id, node_out in outputs.items():
        for key in ("gifs", "videos", "images"):
            for item in node_out.get(key, []) or []:
                fn = item.get("filename")
                if not fn:
                    continue
                out.append({
                    "node_id": node_id,
                    "kind": key,
                    "filename": fn,
                    "subfolder": item.get("subfolder", ""),
                    "type": item.get("type", "output"),
                    "format": item.get("format"),
                })
    return out


def _project_for_request(p: Project, only_scene: Optional[str]) -> Project:
    if not only_scene:
        return p
    scene = next((s for s in p.scenes if s.id == only_scene), None)
    if scene is None:
        raise HTTPException(404, f"Scene {only_scene} not found")
    solo = Project.from_dict(p.to_dict())
    solo.scenes = [scene]
    return solo


@router.post("/{project_id}/generate")
async def generate(project_id: str, body: GenerateBody):
    p = projects.get(project_id)
    if p is None:
        raise HTTPException(404, "Project not found")

    target = _project_for_request(p, body.only_scene)
    prompt = build_combined_prompt(target)
    if not prompt.strip():
        raise HTTPException(400, "Nothing to generate — timeline has no active scene text.")

    try:
        graph = workflow.load_template()
        graph, applied = workflow.inject(graph, {
            "prompt": prompt,
            "seed": target.seed,
            "num_frames_per_scene": target.num_frames_per_scene,
            "frame_rate": target.frame_rate,
            "max_scenes": target.max_scenes,
        })
    except workflow.WorkflowError as e:
        raise HTTPException(400, str(e))

    try:
        result = await comfy_client.queue_prompt(graph)
    except Exception as e:
        raise HTTPException(502, f"Failed to queue on ComfyUI: {e}")

    return {"prompt_id": result.get("prompt_id"), "injected": applied}


@router.get("/{project_id}/status/{prompt_id}")
async def status(project_id: str, prompt_id: str):
    try:
        hist = await comfy_client.history(prompt_id)
    except Exception as e:
        raise HTTPException(502, f"ComfyUI history unavailable: {e}")

    if prompt_id in hist:
        entry = hist[prompt_id]
        media = _media_from_history(entry)
        status_obj = entry.get("status", {})
        return {
            "state": "completed",
            "media": media,
            "status": status_obj,
        }
    try:
        running = await comfy_client.is_running(prompt_id)
    except Exception:
        running = True
    return {"state": "running" if running else "pending", "media": []}


@router.get("/{project_id}/result")
async def result(
    project_id: str,
    filename: str,
    subfolder: str = "",
    type: str = "output",
):
    """Proxy an output file so the browser can play it without CORS to ComfyUI."""
    try:
        data, content_type = await comfy_client.fetch_view(filename, subfolder, type)
    except Exception as e:
        raise HTTPException(502, f"Could not fetch result: {e}")
    return Response(content=data, media_type=content_type)
