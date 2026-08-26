"""On-disk JSON project store. One file per project, keyed by id.

Deliberately simple and synchronous — projects are small. This is the persistence
seam that Phase 2+ extends with cached per-scene latents/assets.
"""
from __future__ import annotations

import json
import time
from typing import Optional

from . import config
from .timeline import Project, Scene, without_key_scoped


def _path(project_id: str):
    return config.PROJECTS_DIR / f"{project_id}.json"


def list_projects() -> list[dict]:
    config.ensure_dirs()
    out: list[dict] = []
    for p in sorted(config.PROJECTS_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        out.append({
            "id": d.get("id", p.stem),
            "name": d.get("name", p.stem),
            "scene_count": len(d.get("scenes", [])),
            "updated_at": d.get("updated_at", 0),
        })
    out.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
    return out


def get(project_id: str) -> Optional[Project]:
    p = _path(project_id)
    if not p.exists():
        return None
    stored = json.loads(p.read_text())
    # Only here, on the way off disk: the rating-learned strengths come from the refinement
    # key, so an opened project must not carry yesterday's copy of them.
    stored["sampler_inputs"] = without_key_scoped(stored.get("sampler_inputs"))
    return Project.from_dict(stored)


def save(project: Project) -> Project:
    config.ensure_dirs()
    project.updated_at = time.time()
    path = _path(project.id)
    payload = json.dumps(project.to_dict(), indent=2)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(payload)
    tmp.replace(path)
    return project


def create(name: str = "Untitled") -> Project:
    project = Project(name=name)
    project.anchor = "Optional: describe the subjects or world shared by all scenes"
    project.scenes = [Scene(text="Describe your first scene here…")]
    return save(project)


def delete(project_id: str) -> bool:
    p = _path(project_id)
    if p.exists():
        p.unlink()
        return True
    return False
