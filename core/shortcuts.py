"""The shortcut library: trigger -> replacement text, expanded into a prompt at
generation time. Global, not per-project -- a shortcut ("cyberpunk" -> a long
style phrase) is reused across every project, not redefined in each one.

Storage is one JSON file holding the whole list, not one-per-item: the picker
browses all of it and export/import round-trips all of it, so there is no
case where only part of the library is ever read or written.

The expansion algorithm (trigger regex, longest-first, seeded multi-choice) and
`$variable` resolution (recursive, cycle-safe, undefined -> literal) are a
close port of v4's `templates.py` (`apply_prompt_shortcuts`/`resolve_variables`)
-- proven logic, not reinvented. Not ported yet: the shortcut revolver's
no-repeat cycling (a stateful commit-vs-preview split).
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
# One draw-and-save at a time: two Generates (or a Generate and a toggle) must not read the
# same cycle position, nor let a stale copy write the old settings back.
_REVOLVER_LOCK = threading.RLock()

MAX_NAME = 120
MAX_ITEM = 4096


def _as_list(raw, sep=r"[,;\n]+"):
    """A string is one separated list (hand-written v4 files); anything that is
    not a list or string is nothing, never its repr. Replacements are prose
    and may hold commas, so they split on newlines only -- as v4 did."""
    if isinstance(raw, str):
        return re.split(sep, raw)
    return [x for x in raw if isinstance(x, (str, int, float))] if isinstance(raw, list) else []


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
        # v4 files spelled these differently; an import must read them.
        triggers = _clean_list(_as_list(d.get("triggers", d.get("activation_words", d.get("activation")))))
        name = str(d.get("name") or "").strip()[:MAX_NAME] or (triggers[0] if triggers else "")
        return Shortcut(
            name=name,
            triggers=triggers,
            replacements=_clean_list(_as_list(d.get("replacements", d.get("replacement")), r"\n+"), keep_empty=True),
            enabled=bool(d.get("enabled", True)),
            category=_label(d.get("category")),
            sub_category=_label(d.get("sub_category")),
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
        same = lambda a, b: a.lower() == b.lower()  # noqa: E731 -- one identity rule, shared with import
        target = original_name or item.name
        # An exact-case match first: a library written before names were one identity may
        # hold "Fox" and "fox", and an edit must land on the one that was edited.
        idx = next((i for i, it in enumerate(items) if it.name == target), None)
        if idx is None:
            idx = next((i for i, it in enumerate(items) if same(it.name, target)), None)
        clash = next((i for i, it in enumerate(items) if same(it.name, item.name)), None)
        renamed = idx is None or items[idx].name != item.name
        if renamed and clash is not None and clash != idx:
            raise ValueError(f"a shortcut named {items[clash].name!r} already exists")
        if idx is None:
            items.append(item)
        else:
            items[idx] = item
        _save_all(items)
        return items


def delete(name: str) -> list[Shortcut]:
    with _LOCK:
        items = listing()
        hit = next((i for i, it in enumerate(items) if it.name == name), None)
        if hit is None:
            hit = next((i for i, it in enumerate(items) if it.name.lower() == name.lower()), None)
        if hit is not None:
            del items[hit]
        _save_all(items)
        return items


def clear() -> list[Shortcut]:
    with _LOCK:
        _save_all([])
        _save_categories([])
        return []


# --- categories --------------------------------------------------------------

def _label(v) -> str:
    if not isinstance(v, str):
        return ""
    return re.sub(r"\s+", " ", str(v or "").strip())[:MAX_NAME]


def _union(cats: list[dict], name, sub="") -> None:
    name, sub = _label(name), _label(sub)
    if not name:
        return
    entry = next((c for c in cats if c["name"].lower() == name.lower()), None)
    if entry is None:
        entry = {"name": name, "sub_categories": []}
        cats.append(entry)
    if sub and sub.lower() not in (s.lower() for s in entry["sub_categories"]):
        entry["sub_categories"].append(sub)


def _saved_categories() -> list[dict]:
    try:
        data = json.loads(config.SHORTCUT_CATEGORIES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    out: list[dict] = []
    for e in data if isinstance(data, list) else []:
        if isinstance(e, str):
            e = {"name": e}
        if isinstance(e, dict):
            _union(out, e.get("name"))
            for s in e.get("sub_categories") or []:
                _union(out, e.get("name"), s)
    return out


def _save_categories(cats: list[dict]) -> None:
    config.ROOT.mkdir(parents=True, exist_ok=True)
    tmp = config.SHORTCUT_CATEGORIES_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cats, indent=2), encoding="utf-8")
    tmp.replace(config.SHORTCUT_CATEGORIES_FILE)


def categories() -> list[dict]:
    """[{name, sub_categories}] -- the ones a person made plus every one a
    shortcut names, so a grouping in use can never be missing from the picker."""
    cats = _saved_categories()
    for it in listing():
        _union(cats, it.category, it.sub_category)
    return cats


def add_category(name, sub_category="") -> list[dict]:
    if not _label(name):
        raise ValueError("a category needs a name")
    with _LOCK:
        cats = _saved_categories()
        _union(cats, name, sub_category)
        _save_categories(cats)
    return categories()


# --- revolver ----------------------------------------------------------------
# A shortcut with several replacements normally draws one at random, so the same
# one can come up twice running. With the revolver on, each shortcut instead
# walks its replacements in turn (first, second, ... or shuffled once per round)
# and repeats none until all have been used.

def _fingerprint(replacements) -> str:
    return hashlib.md5(json.dumps(list(replacements), ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


def load_revolver() -> dict:
    try:
        data = json.loads(config.REVOLVER_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        data = {}
    data = data if isinstance(data, dict) else {}
    return {"enabled": data.get("enabled") is True, "random": data.get("random") is True,
            "state": data["state"] if isinstance(data.get("state"), dict) else {}}


def _save_revolver(data: dict) -> None:
    config.ROOT.mkdir(parents=True, exist_ok=True)
    tmp = config.REVOLVER_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(config.REVOLVER_FILE)


def revolver_settings() -> dict:
    d = load_revolver()
    return {"enabled": d["enabled"], "random": d["random"]}


def set_revolver_settings(enabled=None, random_order=None) -> dict:
    """Any real change restarts every cycle: an order made under one mode means
    nothing under the other. A value that is not a real bool is ignored."""
    with _REVOLVER_LOCK:
        d = load_revolver()
        changed = False
        for key, new in (("enabled", enabled), ("random", random_order)):
            if isinstance(new, bool) and new != d[key]:
                d[key], changed = new, True
        if changed:
            d["state"] = {}
            _save_revolver(d)
        return {"enabled": d["enabled"], "random": d["random"]}


def _draw(state: dict, key: str, replacements: list[str], shuffled: bool, rng) -> str:
    fp = _fingerprint(replacements)
    entry = state.get(key)
    queue = []
    if isinstance(entry, dict) and entry.get("fp") == fp and isinstance(entry.get("queue"), list):
        queue = [i for i in entry["queue"] if isinstance(i, int) and 0 <= i < len(replacements)]
    if not queue:
        queue = list(range(len(replacements)))
        if shuffled:
            rng.shuffle(queue)
    index = queue.pop(0)
    state[key] = {"fp": fp, "queue": queue}
    return replacements[index]


# --- export / import ---------------------------------------------------------

def export_payload() -> dict:
    return {"version": 2, "shortcuts": [s.to_dict() for s in listing()], "categories": categories()}


def import_payload(data, mode: str = "merge") -> int:
    """Read a file written by export_payload() or by v4 (shortcuts as a list or a
    name-keyed object, plus categories). `merge` keeps what is here and
    replaces same-named entries; `replace` starts from the file alone.
    Entries without a trigger are skipped, not fatal -- one bad row in a
    long library is not a reason to refuse the rest. Everything is read and
    checked BEFORE anything is written, so a refused file changes nothing.
    Returns how many shortcuts the library gained or updated."""
    if mode not in ("merge", "replace"):
        raise ValueError("mode is merge or replace")
    raw = data.get("shortcuts") if isinstance(data, dict) else data
    if isinstance(raw, dict):
        # v4 keyed by name and let the key stand in for a row's own empty name.
        raw = [dict(v, name=v.get("name") or k) if isinstance(v, dict) else v for k, v in raw.items()]
    if not isinstance(raw, list):
        raise ValueError("that file holds no shortcuts")
    read: dict[str, Shortcut] = {}
    for r in raw:
        s = Shortcut.from_dict(r)
        if s.triggers:
            read[s.name.lower()] = s      # a name twice in one file: the later row wins
    if not read:
        raise ValueError("that file holds no shortcut with a trigger")
    wanted: list[dict] = []
    for c in (data.get("categories") if isinstance(data, dict) else None) or []:
        c = {"name": c} if isinstance(c, str) else c
        if isinstance(c, dict):
            _union(wanted, c.get("name"))
            subs = c.get("sub_categories")
            for sub in subs if isinstance(subs, list) else []:
                _union(wanted, c.get("name"), sub)
    with _LOCK:
        items = [] if mode == "replace" else listing()
        for s in read.values():
            i = next((i for i, it in enumerate(items) if it.name.lower() == s.name.lower()), None)
            if i is None:
                items.append(s)
            else:
                items[i] = s
        cats = [] if mode == "replace" else _saved_categories()
        for c in wanted:
            _union(cats, c["name"])
            for sub in c["sub_categories"]:
                _union(cats, c["name"], sub)
        _save_all(items)
        _save_categories(cats)
    return len(read)


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


def _combined(items) -> tuple[list, str]:
    """Every enabled trigger with something to put in, longest first, as one pattern: -> (candidates, pattern)."""
    candidates = [(t, _trigger_pattern(t), sc.replacements, sc.name) for sc in items
                  if sc.enabled and sc.replacements for t in sc.triggers if _trigger_pattern(t)]
    candidates.sort(key=lambda c: len(c[0]), reverse=True)
    return candidates, "|".join(f"(?P<t{i}>{p})" for i, (_, p, _, _) in enumerate(candidates))


_TOKEN = re.compile(r"[\w'’-]+|[^\w\s]")


def matcher(items=None):
    """-> fired(text): the names of the shortcuts expand() would replace in `text`, by its own rules. A trigger can only
    fire when every one of its tokens stands whole in the text, so each text is searched for those triggers alone."""
    candidates, _ = _combined(listing() if items is None else items)
    compiled = [re.compile(pattern, re.IGNORECASE) for _, pattern, _, _ in candidates]
    needs = [set(_TOKEN.findall(trigger.lower())) for trigger, *_rest in candidates]
    by_first: dict[str, list[int]] = {}
    always = []           # a trigger lower() reshapes ("İ" -> "i̇") matches case-blind in ways tokens cannot tell
    for i, (trigger, *_rest) in enumerate(candidates):
        if len(trigger.lower()) != len(trigger):
            always.append(i)
        else:
            by_first.setdefault(_TOKEN.findall(trigger.lower())[0], []).append(i)

    def fired(text) -> set:
        text = str(text or "")
        tokens = set(_TOKEN.findall(text.lower()))
        near = [i for t in tokens for i in by_first.get(t, ()) if needs[i] <= tokens] + always
        # As the expander's one pattern does: the leftmost match wins, the longest trigger (lowest i) first at a tie,
        # and the search goes on after it.
        out, pos = set(), 0
        for start, i, end in sorted((m.start(), i, m.end()) for i in near for m in compiled[i].finditer(text)):
            if start >= pos:
                out.add(candidates[i][3])
                pos = end
        return out
    return fired


def expand(text: str, shortcuts: list[Shortcut] | None = None, seed: int = 0, commit: bool = False) -> str:
    with _REVOLVER_LOCK:
        return _expand(text, shortcuts, seed, commit)


def _expand(text: str, shortcuts, seed: int, commit: bool) -> str:
    """Every enabled shortcut's trigger, replaced with one of its
    replacements (random when there is more than one). Deterministic: the
    pick is seeded from `seed`, or from the text itself when seed is 0 --
    so calling this twice on the same text and seed always agrees, without a
    database round-trip to remember what was drawn last time.

    Longest trigger wins first ("golden hour" is tried before "golden"), so
    a short trigger can never shadow a longer one that contains it.

    With the revolver on, a multi-replacement shortcut takes the next one in
    its cycle instead. A preview only PEEKS (reads the stored cycle, saves
    nothing) so it shows exactly what the next generation will draw; only
    `commit=True` -- a real generation -- moves the cycle on.
    """
    original = str(text or "")
    if not original:
        return original
    candidates, combined = _combined(listing() if shortcuts is None else shortcuts)
    if not candidates:
        return original
    revolver = load_revolver()
    drew = False
    rng_seed = int(seed or 0) or int(hashlib.md5(original.encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(rng_seed)
    removed = False

    def replace(m):
        nonlocal removed, drew
        for i, (_, _, replacements, name) in enumerate(candidates):
            if m.group(f"t{i}") is None:
                continue
            key = name.lower()
            if revolver["enabled"] and len(replacements) > 1:
                choice = _draw(revolver["state"], key, replacements, revolver["random"], rng)
                drew = True
            else:
                choice = rng.choice(replacements)
            if not choice:
                removed = True
            return choice
        return m.group(0)  # unreachable: `combined` only matches a known group

    expanded = re.sub(combined, replace, original, flags=re.IGNORECASE | re.UNICODE)
    if drew and commit:
        with _LOCK:
            _save_revolver(revolver)
    return _cleanup_removed_phrases(expanded) if removed else expanded


# --- $variables --------------------------------------------------------------

_VARIABLE_TOKEN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_VARIABLE_MAX_DEPTH = 64
#: A ceiling on total resolved length, in characters -- far past anything a
#: real prompt needs, and not a working limit for one. It exists because a
#: variable referencing the same other variable more than once is not a
#: cycle (see below) and the OUTPUT for that shape is legitimately
#: exponential even once the per-(name,stack) evaluations are memoized: each
#: level's re.sub still has to concatenate the (now O(1)-computed) child
#: result multiple times, so the memo bounds compute but not output size.
_VARIABLE_MAX_OUTPUT = 100_000


def resolve_variables(text: str, variables) -> str:
    """Substitute `$name` tokens from a project's variables ([{"name",
    "value"}, ...]). Recursive -- a variable's value may itself reference
    another -- cycle-safe (a name reappearing in its own expansion chain is
    left literal rather than recursed into), and undefined names are left as
    literal `$name`: an unset variable is a typo to notice and fix, not a
    blank the prompt swallows silently.

    Memoized per (name, stack): a value referencing the SAME variable more
    than once (`v0 = "$v1 $v1"`) is not a cycle -- `name in stack` never
    trips -- so without this a chain of only two dozen such variables costs
    2**24 evaluations of the last one and tens of megabytes of output before
    _VARIABLE_MAX_DEPTH ever gets the chance to matter (that cap bounds
    depth, not the branching this shape produces). The cache key includes
    `stack`, not just `name`, because the SAME variable can legitimately
    resolve differently depending on which cycle it is being asked from
    (A="$B", B="$A": resolving from A ends in literal "$A", from B in
    literal "$B") -- collapsing by name alone would answer one of those with
    the other's result. Past memoization, `_VARIABLE_MAX_OUTPUT` bounds the
    total characters produced -- once hit, remaining tokens are left literal
    rather than expanded.
    """
    var_map: dict[str, str] = {}
    for v in (variables or []):
        name = str((v or {}).get("name") or "").lstrip("$").strip()
        if name:
            var_map[name] = str((v or {}).get("value") or "")
    if not var_map:
        return str(text or "")

    memo: dict[tuple[str, frozenset], str] = {}
    budget = [_VARIABLE_MAX_OUTPUT]

    def _expand(s, stack, depth):
        if depth > _VARIABLE_MAX_DEPTH:
            return s

        def _repl(m):
            if budget[0] <= 0:
                return m.group(0)
            name = m.group(1)
            if name not in var_map or name in stack:
                return m.group(0)
            key = (name, stack)
            if key in memo:
                result = memo[key]
            else:
                result = _expand(var_map[name], stack | {name}, depth + 1)
                memo[key] = result
            budget[0] -= len(result)
            return result

        return _VARIABLE_TOKEN.sub(_repl, s)

    return _expand(str(text or ""), frozenset(), 0)
