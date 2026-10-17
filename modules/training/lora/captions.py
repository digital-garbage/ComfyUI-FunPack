"""Captions as tags: the small text helpers the tag editor and the trainer share. No ComfyUI, no I/O."""

from __future__ import annotations

MAX_CAPTION = 4000


def split_tags(text) -> list[str]:
    """"a, b ,, a" -> ["a", "b"]: trimmed, no empties, first occurrence kept."""
    seen, out = set(), []
    for part in str(text or "").replace("\n", ",").split(","):
        tag = " ".join(part.split())
        if tag and tag.lower() not in seen:
            seen.add(tag.lower())
            out.append(tag)
    return out


def join_tags(tags) -> str:
    return ", ".join(split_tags(", ".join(str(t) for t in tags)))[:MAX_CAPTION]


def with_trigger(caption, trigger) -> str:
    """The trigger word first, once. An empty trigger changes nothing."""
    trigger = " ".join(str(trigger or "").split())
    if not trigger:
        return str(caption or "")
    rest = [t for t in split_tags(caption) if t.lower() != trigger.lower()]
    return ", ".join([trigger, *rest])


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
        tags = split_tags(item.get("caption"))
        low = [t.lower() for t in tags]
        if op == "add" and tag.lower() not in low:
            tags.append(tag)
        elif op == "prepend":
            tags = [tag] + [t for t in tags if t.lower() != tag.lower()]
        elif op == "remove" and tag.lower() in low:
            tags = [t for t in tags if t.lower() != tag.lower()]
        elif op == "replace" and tag.lower() in low:
            tags = [new if t.lower() == tag.lower() else t for t in tags]
        else:
            continue
        text = join_tags(tags)
        if text != item.get("caption"):
            item["caption"] = text
            changed += 1
    return changed

