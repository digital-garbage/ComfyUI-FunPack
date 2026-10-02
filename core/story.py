"""The Story view: every scene's text as one box, cut by typed marker words.

`Scene.text` stays the truth (core/projects.py). A story is only a second way
to look at and edit the same scenes: `join` lays them out with a marker between
each, `split` reads that text back into scenes. A marker is a word the person
types (`qcut` by default); it is consumed on split, so a cut is never part of
any scene's text, and only typed words cut -- shortcut expansion happens later
(core/prompt_build.py) and never moves a boundary.

Lossless: split(join(scenes)) == scenes for any scenes that do not themselves
contain a marker word (an empty scene included -- "a qcut qcut b" is three
scenes, the middle one empty). The anchor is not part of the story; it has its
own field.
"""

from __future__ import annotations

import json
import re

from . import config

DEFAULT_MARKERS = ["qcut"]
MAX_MARKER = 60


def markers() -> list[str]:
    try:
        data = json.loads(config.MARKERS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return list(DEFAULT_MARKERS)
    return _clean(data) or list(DEFAULT_MARKERS)


def _clean(raw) -> list[str]:
    out = []
    for m in raw if isinstance(raw, list) else []:
        m = " ".join(str(m).split())[:MAX_MARKER]
        if m and m.lower() not in (o.lower() for o in out):
            out.append(m)
    return out


def save_markers(raw) -> list[str]:
    """A story with no marker could never be cut, so an empty list is refused
    rather than quietly replaced by the default."""
    clean = _clean(raw)
    if not clean:
        raise ValueError("at least one marker word is needed to cut scenes")
    config.ROOT.mkdir(parents=True, exist_ok=True)
    tmp = config.MARKERS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    tmp.replace(config.MARKERS_FILE)
    return clean


def _pattern(words: list[str]):
    # Longest first so "scene cut" wins over "cut"; whole words only, so
    # "qcutting" is not a cut. Spaces in a marker match any run of whitespace.
    alts = [r"\s+".join(map(re.escape, w.split()))
            for w in sorted(words, key=len, reverse=True)]
    return re.compile(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)", re.IGNORECASE)


def split(text: str, words: list[str] | None = None) -> list[str]:
    parts = _pattern(words or markers()).split(str(text or ""))
    return [p.strip() for p in parts]


def join(scenes, words: list[str] | None = None) -> str:
    marker = (words or markers())[0]
    return f"\n{marker}\n".join(str(s or "").strip() for s in scenes)
