"""Project CRUD + timeline read/write."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import projects
from ..timeline import Project

router = APIRouter(prefix="/api/projects", tags=["projects"])


class CreateBody(BaseModel):
    name: str = "Untitled"


@router.get("")
def list_all():
    return {"projects": projects.list_projects()}


@router.post("")
def create(body: CreateBody):
    return projects.create(body.name).to_dict()


@router.get("/{project_id}")
def get(project_id: str):
    p = projects.get(project_id)
    if p is None:
        raise HTTPException(404, "Project not found")
    return p.to_dict()


@router.put("/{project_id}")
def update(project_id: str, body: dict):
    existing = projects.get(project_id)
    if existing is None:
        raise HTTPException(404, "Project not found")
    body["id"] = project_id  # id is immutable
    body.setdefault("created_at", existing.created_at)
    return projects.save(Project.from_dict(body)).to_dict()


@router.delete("/{project_id}")
def delete(project_id: str):
    if not projects.delete(project_id):
        raise HTTPException(404, "Project not found")
    return {"deleted": project_id}
