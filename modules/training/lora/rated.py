"""Results the person rated, found in their saved projects.

A render's rating lives on its scene (and on each earlier take); this reads the project files and
lists every rated result that still has a file, newest project first. The wording of a rating is
the rating feature's (`app/ui/features/rating/choices.js`): a number from 6 up is liked, a lower
one disliked, and "Disliked: bad image" / "Disliked: bad composition" say which way it went.
"""

from __future__ import annotations

from .store import kind_of

FORGET = "-Just forget it-"
LABEL_TITLES = {"liked": "Liked results", "disliked": "Disliked results", "bad_composition": "Bad composition results",
                "bad_image": "Bad image results"}


def label_of(rating) -> str | None:
    v = str(rating or "").replace("|loved", "").strip()
    if not v or v == FORGET:
        return None
    if v == "Disliked: bad image":
        return "bad_image"
    if v == "Disliked: bad composition":
        return "bad_composition"
    if v.isdigit():
        return "liked" if int(v) >= 6 else "disliked"
    return "disliked" if v.startswith("Disliked") else None


def _media(raw):
    if not isinstance(raw, dict):
        return None
    name = raw.get("filename")
    if not isinstance(name, str) or not kind_of(name):
        return None
    return {"filename": name, "subfolder": str(raw.get("subfolder") or ""), "type": raw.get("type") if raw.get("type") in ("output", "temp", "input") else "output"}


def scan(projects) -> list[dict]:
    """[{project, project_name, label, rating, media, caption}] for every rated result in these project documents."""
    out, seen = [], set()

    def take(doc, rating, media, caption):
        label, media = label_of(rating), _media(media)
        if label is None or media is None:
            return
        key = (media["type"], media["subfolder"], media["filename"])
        if key in seen:
            return
        seen.add(key)
        out.append({"project": doc.get("id", ""), "project_name": doc.get("name", ""), "label": label, "rating": str(rating),
                    "media": media, "caption": " ".join(str(caption or "").split())})

    for doc in projects:
        if not isinstance(doc, dict):
            continue
        scenes = {s.get("id"): s for s in doc.get("scenes") or [] if isinstance(s, dict)}
        for sid, render in (doc.get("scene_renders") or {}).items():
            scene, render = scenes.get(sid) or {}, render if isinstance(render, dict) else {}
            rated = scene.get("rated") if isinstance(scene.get("rated"), dict) else {}
            take(doc, scene.get("rating"), render.get("media"), rated.get("text") or render.get("text") or scene.get("text"))
        for sid, takes in (doc.get("scene_variants") or {}).items():
            for t in takes if isinstance(takes, list) else []:
                if isinstance(t, dict):
                    rated = t.get("rated") if isinstance(t.get("rated"), dict) else {}
                    take(doc, t.get("rating"), t.get("media"), rated.get("text") or t.get("text"))
    return out
