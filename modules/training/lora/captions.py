"""Captions as tags: the small text helpers the tag editor and the trainer share. No ComfyUI, no I/O."""

from __future__ import annotations

MAX_CAPTION = 4000


def split_tags(text) -> list[str]:
    """"a, b ,, a" -> ["a", "b"]: split at commas, spaces collapsed, no empties, first occurrence kept. For counting, not for rewriting."""
    seen, out = set(), []
    for part in str(text or "").replace("\n", ",").split(","):
        tag = " ".join(part.split())
        if tag and tag.lower() not in seen:
            seen.add(tag.lower())
            out.append(tag)
    return out


def join_tags(tags) -> str:
    return ", ".join(split_tags(", ".join(str(t) for t in tags)))[:MAX_CAPTION]


def parts(text) -> list[str]:
    """The comma separated pieces of a caption as typed: trimmed, no empties, nothing dropped or reworded."""
    return [p.strip() for p in str(text or "").split(",") if p.strip()]


def _same(a, b) -> bool:
    return " ".join(a.split()).lower() == " ".join(b.split()).lower()


def with_parts(caption, prefix="", postfix="") -> str:
    """The caption between the prefix and the postfix, joined with commas. Empty parts are left out; the caption is not reworded."""
    pieces = [" ".join(str(x or "").split()) for x in (prefix,)] + [str(caption or "").strip()] + [" ".join(str(postfix or "").split())]
    return ", ".join(p.strip(", ") for p in pieces if p.strip(", "))


def add_tags(caption, tags) -> str:
    """The caption with these tags added at its end, after a comma, skipping any it already has. The caption itself is kept as typed."""
    have = parts(caption)
    new = [t for t in split_tags(", ".join(str(t) for t in tags)) if not any(_same(t, h) for h in have)]
    text = str(caption or "").rstrip().rstrip(",").rstrip()
    return (", ".join([text, *new]) if text else ", ".join(new))[:MAX_CAPTION]


def frequency(captions) -> list[tuple[str, int]]:
    """Every tag with the number of captions it is in, most common first, then A to Z."""
    counts: dict[str, list] = {}
    for caption in captions:
        for tag in split_tags(caption):
            counts.setdefault(tag.lower(), [tag, 0])[1] += 1
    return sorted(((tag, n) for tag, n in counts.values()), key=lambda x: (-x[1], x[0].lower()))


def bulk(items, op, tag, new=None, ids=None) -> int:
    """Change captions in place; returns how many items changed.

    `op` is add (at the end), prepend, remove or replace (tag -> new). `ids` limits it to those items.
    Only the comma separated piece that matches is touched; the rest of a caption keeps its words.
    """
    changed = 0
    tag = " ".join(str(tag or "").split())
    if not tag or op not in ("add", "prepend", "remove", "replace"):
        return 0
    if op == "replace":
        new = " ".join(str(new or "").split())
        if not new:
            op = "remove"
    for item in items:
        if ids is not None and item.get("id") not in ids:
            continue
        pieces = parts(item.get("caption"))
        has = any(_same(p, tag) for p in pieces)
        if op == "add" and not has:
            pieces.append(tag)
        elif op == "prepend":
            pieces = [tag] + [p for p in pieces if not _same(p, tag)]
        elif op == "remove" and has:
            pieces = [p for p in pieces if not _same(p, tag)]
        elif op == "replace" and has:
            pieces = [new if _same(p, tag) else p for p in pieces]
        else:
            continue
        text = ", ".join(pieces)[:MAX_CAPTION]
        if text != item.get("caption"):
            item["caption"] = text
            changed += 1
    return changed
