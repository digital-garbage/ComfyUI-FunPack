"""Timeline preview: assemble the combined prompt and show the canonical split.

This is the round-trip integrity check from the plan — the editor calls it live so
the user always sees exactly how Studio will split their timeline before generating.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import comfy_client, projects
from ..timeline import build_combined_prompt

router = APIRouter(prefix="/api/projects", tags=["timeline"])


@router.get("/{project_id}/preview")
async def preview(project_id: str, include_excluded: bool = False):
    p = projects.get(project_id)
    if p is None:
        raise HTTPException(404, "Project not found")
    prompt = build_combined_prompt(p, include_excluded=include_excluded)
    result = {"combined_prompt": prompt}
    try:
        parsed = await comfy_client.parse_timeline(prompt, seed=p.seed)
        result["parsed"] = parsed
        active = [s for s in p.scenes if include_excluded or not s.excluded]
        expected = len(active)
        got = len(parsed.get("scenes", []))
        result["expected_scenes"] = expected
        result["parsed_scenes"] = got
        if expected and got != expected:
            result["warning"] = (
                f"Studio split produced {got} scene(s) but the timeline has {expected}. "
                f"Check transition markers (each scene needs a recognized trigger before it)."
            )
    except Exception as e:
        result["parse_error"] = f"ComfyUI parse_timeline unavailable: {e}"
    return result
