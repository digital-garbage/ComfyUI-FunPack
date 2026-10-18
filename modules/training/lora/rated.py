"""Results the person rated, found in their saved projects.

A render's rating lives on its scene (and on each earlier take); this reads the project files and
lists every rated result that still has a file, newest project first. The wording of a rating is
the rating feature's (`app/ui/features/rating/choices.js`): a number from 6 up is liked, a lower
one disliked, and "Disliked: bad image" / "Disliked: bad composition" say which way it went.
"""

from __future__ import annotations

import json
import re

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


INCLUDE = ("prefix", "main", "postfix", "variables")


def prompt_caption(text, doc=None, include=("main",), expand=None, resolve=None) -> str:
    """A rated result's prompt as a caption, from the parts asked for.

    `include` names the parts: prefix and postfix are the project's anchor and postfix, main is the scene's own words,
    variables fills in `$name` (left as written when not asked for). `expand(text)` is the app's shortcut
    expansion; `resolve(text, variables)` fills the variables in.
    """
    doc = doc if isinstance(doc, dict) else {}
    want = set(include or ())
    parts = []
    if "prefix" in want:
        parts.append(doc.get("anchor"))
    if "main" in want:
        parts.append(text)
    if "postfix" in want and doc.get("postfix_enabled") is not False:
        parts.append(doc.get("postfix"))
    joined = " ".join(" ".join(str(p or "").split()) for p in parts).strip()
    if expand and joined:
        joined = " ".join(str(expand(joined)).split())
    if "variables" in want and resolve:
        joined = resolve(joined, doc.get("variables"))
    return " ".join(joined.split())


def _graph_of(path):
    raw = None
    try:
        if str(path).lower().endswith((".png", ".webp", ".jpg", ".jpeg")):
            from PIL import Image
            with Image.open(path) as img:
                raw = (img.info or {}).get("prompt")
        else:
            import av
            with av.open(str(path)) as container:
                raw = (container.metadata or {}).get("prompt")
        graph = json.loads(raw) if raw else None
    except Exception:                                      # noqa: BLE001 -- a file that cannot be read says nothing
        return None
    return graph if isinstance(graph, dict) else None


def embedded_prompt(path) -> str:
    """The prompt the file was made from, read from its own queued graph: the text typed into the prompt node (before any rewrite), else a text encoder's. "" when it says nothing."""
    graph = _graph_of(path)
    if graph is None:
        return ""
    nodes = [(i, n) for i, n in graph.items() if isinstance(n, dict) and isinstance((n.get("inputs") or {}).get("text"), str)]
    for wanted in ("FunPackEnhancePrompt",):
        for _, n in nodes:
            if n.get("class_type") == wanted and n["inputs"]["text"].strip():
                return " ".join(n["inputs"]["text"].split())
    encoders = [n["inputs"]["text"] for i, n in nodes if n.get("class_type") == "CLIPTextEncode" and i != "negative" and n["inputs"]["text"].strip()]
    return " ".join(max(encoders, key=len).split()) if encoders else ""


def embedded_text(path) -> str | None:
    """Every piece of text the file's own queued graph carried (ComfyUI keeps it in the video or picture), or None when the file says nothing."""
    graph = _graph_of(path)
    if graph is None:
        return None
    texts = []
    for node in graph.values():
        for value in ((node or {}).get("inputs") or {}).values() if isinstance(node, dict) else ():
            if isinstance(value, str):
                texts.append(value)
    return " ".join(texts)


def _words(text) -> set[str]:
    return set(re.findall(r"[a-z0-9']{3,}", str(text or "").lower()))


def same_video(prompt_text, path, minimum: float = 0.5) -> bool:
    """False only when the file proves it was made from another prompt: file names repeat between ComfyUI instances, so a name alone does not say which video was rated.

    Judged on the words of the prompt the project kept for it; a file that carries no prompt of its own, or a rating with no words, is given the benefit of the doubt.
    """
    wanted = _words(prompt_text)
    if not wanted:
        return True
    found = embedded_text(path)
    if found is None:
        return True
    return len(wanted & _words(found)) / len(wanted) >= minimum


def is_this_video(result, path, slack_ms: int = 120_000) -> bool:
    """Whether the file under this name is the video the rating was given to. A rating is stamped with the time it was given, and a file is
    never made after the rating of it; one that is newer is another video that took the name (names repeat between ComfyUI instances),
    unless its own embedded prompt still says it is the same."""
    at = result.get("rated_at")
    try:
        newer = at is None or path.stat().st_mtime * 1000 > at + slack_ms      # a rating with no time on it cannot say: the prompt has to
    except OSError:
        return False
    return not newer or same_video(result.get("prompt_text"), path)


def scan(projects, expand=None, resolve=None, include=("main",)) -> list[dict]:
    """[{project, project_name, label, rating, media, caption}] for every rated result in these project documents.

    The caption is the prompt that made the result, built by `prompt_caption` from the parts in `include`.
    """
    out, seen = [], set()

    def take(doc, rating, media, caption, at=None):
        label, media = label_of(rating), _media(media)
        if label is None or media is None:
            return
        key = (media["type"], media["subfolder"], media["filename"])
        if key in seen:
            return
        seen.add(key)
        out.append({"project": doc.get("id", ""), "project_name": doc.get("name", ""), "label": label, "rating": str(rating),
                    "media": media, "caption": prompt_caption(caption, doc, include, expand, resolve), "prompt_text": " ".join(str(caption or "").split()),
                    "rated_at": at if isinstance(at, (int, float)) else None})

    for doc in projects:
        if not isinstance(doc, dict):
            continue
        scenes = {s.get("id"): s for s in doc.get("scenes") or [] if isinstance(s, dict)}
        for sid, render in (doc.get("scene_renders") or {}).items():
            scene, render = scenes.get(sid) or {}, render if isinstance(render, dict) else {}
            rated = scene.get("rated") if isinstance(scene.get("rated"), dict) else {}
            take(doc, scene.get("rating"), render.get("media"), rated.get("text") or render.get("text") or scene.get("text"), rated.get("at"))
        for sid, takes in (doc.get("scene_variants") or {}).items():
            for t in takes if isinstance(takes, list) else []:
                if isinstance(t, dict):
                    rated = t.get("rated") if isinstance(t.get("rated"), dict) else {}
                    take(doc, t.get("rating"), t.get("media"), rated.get("text") or t.get("text"), rated.get("at"))
    return out
