"""Projects: an ordered list of scenes, one JSON file each.

A scene owns its prompt text and the result it last produced. Nothing else about
a scene lives here. v4's Scene accumulated three dozen fields over its life --
transitions, effects, audio gain, per-scene frames, guide sources -- and each of
those belongs to the feature that needs it, added when that feature lands, not to
the model every part of the app has to load.

Per-scene length and fps are deliberately absent: regeneration uses the PROJECT's
values, never a scene's own, so a scene cropped on the timeline and regenerated
comes back whole. A crop is a timeline decision; a regenerate is a new scene.

Synchronous, one file per project, stdlib json. Projects are small.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field

from . import config

#: A generated id, never a user-supplied name, is what reaches the filesystem.
_ID = re.compile(r"\A[0-9a-f]{12}\Z")

MAX_NAME = 120


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def is_id(value) -> bool:
    """True for something this module generated. Everything from a request is
    checked against this before it is used to build a path."""
    return isinstance(value, str) and bool(_ID.match(value))


def _clean_name(raw, fallback="Untitled") -> str:
    name = (raw if isinstance(raw, str) else "").strip()
    return name[:MAX_NAME] or fallback


MAX_SETTING = 16384


def _whole(value) -> int | None:
    """A whole positive number, or nothing. Everything read back out of a
    project file goes through here: the file outlives the code that wrote it."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 1 <= number <= MAX_SETTING else None


#: What a person can say about a result. Core keeps the word and nothing else:
#: what any of them MEAN belongs to whatever learns from them, which is not here.
RATINGS = ("perfect", "good", "wrong", "awful")


@dataclass
class Scene:
    id: str = field(default_factory=_new_id)
    text: str = ""
    #: The asset this scene last produced, so a reload shows the timeline the
    #: user left rather than an empty one they have to regenerate.
    result: str | None = None
    #: How long this clip RUNS on the timeline -- a crop, made here. It is not
    #: what a regenerate uses: that reads the project, because the crop was a
    #: timeline decision and a regenerate is a new scene.
    length: int | None = None
    rating: str | None = None

    @staticmethod
    def from_dict(d) -> "Scene":
        d = d if isinstance(d, dict) else {}
        sid = d.get("id")
        result = d.get("result")
        rating = d.get("rating")
        return Scene(
            id=sid if is_id(sid) else _new_id(),
            text=d.get("text") if isinstance(d.get("text"), str) else "",
            result=result if isinstance(result, str) else None,
            length=_whole(d.get("length")),
            rating=rating if rating in RATINGS else None,
        )


def _clean_video(raw) -> dict:
    """What the app is holding for the pipeline's `project.video` inputs.

    Core does not know what a video setting IS. Which of them exist is the
    pipeline's business -- it says so with a role -- and naming width and height
    here would be core naming an implementation. What it knows is that a project
    file is the one thing in this app that outlives the code that wrote it, so
    everything read back out of one is checked: a whole positive number, or gone.
    """
    if not isinstance(raw, dict):
        return {}
    clean = {}
    for key, value in raw.items():
        # `True` is an int in Python and would land in a width.
        if not isinstance(key, str) or isinstance(value, bool):
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= MAX_SETTING:
            clean[key] = number
    return clean


@dataclass
class Project:
    id: str = field(default_factory=_new_id)
    name: str = "Untitled"
    scenes: list[Scene] = field(default_factory=list)
    #: Settings the whole project is generated at -- size, length -- rather than
    #: any one scene. A scene cropped on the timeline and regenerated comes back
    #: at the project's length: the crop was a timeline decision and a regenerate
    #: is a new scene.
    video: dict = field(default_factory=dict)
    updated_at: float = 0.0

    @staticmethod
    def from_dict(d) -> "Project":
        d = d if isinstance(d, dict) else {}
        pid = d.get("id")
        raw = d.get("scenes")
        return Project(
            id=pid if is_id(pid) else _new_id(),
            name=_clean_name(d.get("name")),
            scenes=[Scene.from_dict(s) for s in (raw if isinstance(raw, list) else [])],
            video=_clean_video(d.get("video")),
            updated_at=float(d.get("updated_at") or 0.0),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _dir():
    config.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    return config.PROJECTS_DIR


def _path(project_id: str):
    # is_id, not a sanitiser: a name that merely survives cleaning can still be
    # "..", and a store that repairs a bad id quietly is one that writes
    # somewhere nobody asked for.
    if not is_id(project_id):
        raise ValueError(f"not a project id: {project_id!r}")
    return _dir() / f"{project_id}.json"


def listing() -> list[dict]:
    """Every project, newest first, as the little the picker needs."""
    out = []
    for p in _dir().glob("*.json"):
        if not is_id(p.stem):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue  # an unreadable file is not a reason to have no list
        if not isinstance(d, dict):
            continue
        out.append({
            "id": p.stem,
            "name": _clean_name(d.get("name"), p.stem),
            "scenes": len(d.get("scenes") or []),
            "updated_at": float(d.get("updated_at") or 0.0),
        })
    out.sort(key=lambda x: x["updated_at"], reverse=True)
    return out


def get(project_id: str) -> Project | None:
    try:
        path = _path(project_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        return Project.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def save(project: Project) -> Project:
    project.updated_at = time.time()
    path = _path(project.id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(project.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(path)  # a concurrent read never sees a half-written project
    return project


def create(name=None) -> Project:
    """A new project with one empty scene: a timeline with nothing on it cannot
    be typed into, so a brand-new project would need an Add before it could be
    used at all."""
    return save(Project(name=_clean_name(name), scenes=[Scene()]))


def delete(project_id: str) -> bool:
    try:
        path = _path(project_id)
    except ValueError:
        return False
    if not path.exists():
        return False
    path.unlink()
    return True
