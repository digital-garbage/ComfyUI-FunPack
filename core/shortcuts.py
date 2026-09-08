"""The shortcut library: trigger -> replacement text, expanded into a prompt at
generation time. Global, not per-project -- a shortcut ("cyberpunk" -> a long
style phrase) is reused across every project, not redefined in each one.

Storage is one JSON file holding the whole list, not one-per-item: the picker
browses all of it and export/import round-trips all of it, so there is no
case where only part of the library is ever read or written.

The expansion algorithm (trigger regex, longest-first, seeded multi-choice) and
`$variable` resolution (recursive, cycle-safe, undefined -> literal) are a
close port of v4's `templates.py` (`apply_prompt_shortcuts`/`resolve_variables`)
-- proven logic, not reinvented. Not ported: the shortcut revolver's
no-repeat cycling (a stateful commit-vs-preview split) and managed category
lists (empty categories kept around for a picker to offer) -- both are real
v4 features, cut here to keep this a first pass rather than a rewrite of
the whole prompt-craft surface at once.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
from dataclasses import asdict, dataclass, field

from . import config

_LOCK = threading.Lock()

MAX_NAME = 120
MAX_ITEM = 4096


def _clean_list(raw, *, keep_empty=False) -> list[str]:
    """One-per-line strings: trimmed, internal whitespace collapsed, exact
    duplicates dropped. Triggers and replacements are both lists of whole
    phrases -- comma-splitting a replacement tears one prose phrase into
    bogus variants, so callers never do that upstream of here.

    `keep_empty` is for replacements only: an empty string there is a
    deliberate "remove this phrase" entry (`expand()`'s `_cleanup_removed_
    phrases`), not a blank line to discard -- a trigger has no such meaning
    and always drops empties."""
    if isinstance(raw, str):
        raw = raw.split("\n")
    if not isinstance(raw, list):
        return []
    seen, out = set(), []
    for item in raw:
        text = re.sub(r"\s+", " ", str(item if item is not None else "")).strip()[:MAX_ITEM]
        if (text or keep_empty) and text not in seen:
            seen.add(text)
            out.append(text)
    return out


@dataclass
class Shortcut:
    name: str = ""
    triggers: list[str] = field(default_factory=list)
    replacements: list[str] = field(default_factory=list)
    enabled: bool = True
    #: Free-text grouping for the picker UI. Plain strings, not a managed
    #: list -- a category exists the moment a shortcut is saved under it, and
    #: goes away the moment nothing uses it any more, no separate CRUD needed.
    category: str = ""
    sub_category: str = ""

    @staticmethod
    def from_dict(d) -> "Shortcut":
        d = d if isinstance(d, dict) else {}
        triggers = _clean_list(d.get("triggers"))
        name = str(d.get("name") or "").strip()[:MAX_NAME] or (triggers[0] if triggers else "")
        return Shortcut(
            name=name,
            triggers=triggers,
            replacements=_clean_list(d.get("replacements"), keep_empty=True),
            enabled=bool(d.get("enabled", True)),
            category=str(d.get("category") or "").strip()[:MAX_NAME],
            sub_category=str(d.get("sub_category") or "").strip()[:MAX_NAME],
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _path():
    config.ROOT.mkdir(parents=True, exist_ok=True)
    return config.SHORTCUTS_FILE


def listing() -> list[Shortcut]:
    p = config.SHORTCUTS_FILE
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return []  # an unreadable library is not a reason to have no list
    items = data if isinstance(data, list) else []
    return [Shortcut.from_dict(it) for it in items]


def _save_all(items: list[Shortcut]) -> None:
    path = _path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([it.to_dict() for it in items], indent=2), encoding="utf-8")
    tmp.replace(path)  # a concurrent read never sees a half-written library


def save(payload: dict, original_name: str | None = None) -> list[Shortcut]:
    """Insert, or update in place when `original_name` names an existing
    entry -- a rename keeps its position; without this a save-under-a-new-
    name would move the entry to the end, reordering the picker under the
    user's feet for no reason they asked for. Raises ValueError for a
    shortcut with no triggers: nothing can ever match it, so saving one
    would be silent data loss dressed up as success."""
    item = Shortcut.from_dict(payload)
    if not item.triggers:
        raise ValueError("a shortcut needs at least one trigger")
    with _LOCK:
        items = listing()
        idx = next((i for i, it in enumerate(items)
                    if it.name == (original_name or item.name)), None)
        if idx is None:
            items.append(item)
        else:
            items[idx] = item
        _save_all(items)
        return items


def delete(name: str) -> list[Shortcut]:
    with _LOCK:
        items = [it for it in listing() if it.name != name]
        _save_all(items)
        return items


def clear() -> list[Shortcut]:
    with _LOCK:
        _save_all([])
        return []


# --- shortcut expansion ------------------------------------------------------

def _trigger_pattern(trigger: str) -> str:
    words = [re.escape(w) for w in re.split(r"\s+", trigger.strip()) if w]
    if not words:
        return ""
    body = r"\s+".join(words)
    return rf"(?<![\w'’-])({body})(?![\w'’-])"


def _cleanup_removed_phrases(text: str) -> str:
    """Fix punctuation/spacing left behind when a replacement is "" (a
    deliberate 'remove this phrase' entry)."""
    text = re.sub(r"[ \t]+([,;])", r"\1", text)
    text = re.sub(r"([,;])\s*([,;])+", r"\1", text)
    text = re.sub(r"^[\s,;]+", "", text)
    text = re.sub(r"[\s,;]+$", "", text)
    return re.sub(r"[ \t]{2,}", " ", text)


def expand(text: str, shortcuts: list[Shortcut] | None = None, seed: int = 0) -> str:
    """Every enabled shortcut's trigger, replaced with one of its
    replacements (random when there is more than one). Deterministic: the
    pick is seeded from `seed`, or from the text itself when seed is 0 --
    so calling this twice on the same text and seed always agrees, without a
    database round-trip to remember what was drawn last time.

    Longest trigger wins first ("golden hour" is tried before "golden"), so
    a short trigger can never shadow a longer one that contains it.
    """
    original = str(text or "")
    if not original:
        return original
    items = listing() if shortcuts is None else shortcuts
    candidates = []
    for sc in items:
        if not sc.enabled or not sc.replacements:
            continue
        for trigger in sc.triggers:
            pattern = _trigger_pattern(trigger)
            if pattern:
                candidates.append((trigger, pattern, sc.replacements))
    if not candidates:
        return original

    candidates.sort(key=lambda c: len(c[0]), reverse=True)
    combined = "|".join(f"(?P<t{i}>{p})" for i, (_, p, _) in enumerate(candidates))
    rng_seed = int(seed or 0) or int(hashlib.md5(original.encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(rng_seed)
    removed = False

    def replace(m):
        nonlocal removed
        for i, (_, _, replacements) in enumerate(candidates):
            if m.group(f"t{i}") is None:
                continue
            choice = rng.choice(replacements)
            if not choice:
                removed = True
            return choice
        return m.group(0)  # unreachable: `combined` only matches a known group

    expanded = re.sub(combined, replace, original, flags=re.IGNORECASE | re.UNICODE)
    return _cleanup_removed_phrases(expanded) if removed else expanded


# --- $variables --------------------------------------------------------------

_VARIABLE_TOKEN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_VARIABLE_MAX_DEPTH = 64


def resolve_variables(text: str, variables) -> str:
    """Substitute `$name` tokens from a project's variables ([{"name",
    "value"}, ...]). Recursive -- a variable's value may itself reference
    another -- cycle-safe (a name reappearing in its own expansion chain is
    left literal rather than recursed into), and undefined names are left as
    literal `$name`: an unset variable is a typo to notice and fix, not a
    blank the prompt swallows silently.
    """
    var_map: dict[str, str] = {}
    for v in (variables or []):
        name = str((v or {}).get("name") or "").lstrip("$").strip()
        if name:
            var_map[name] = str((v or {}).get("value") or "")
    if not var_map:
        return str(text or "")

    def _expand(s, stack, depth):
        if depth > _VARIABLE_MAX_DEPTH:
            return s

        def _repl(m):
            name = m.group(1)
            if name not in var_map or name in stack:
                return m.group(0)
            return _expand(var_map[name], stack | {name}, depth + 1)

        return _VARIABLE_TOKEN.sub(_repl, s)

    return _expand(str(text or ""), frozenset(), 0)
