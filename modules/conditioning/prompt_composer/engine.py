"""Local, reviewed prompt drafts built from shortcuts and exact user edits."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import threading
from collections import Counter

from ....core import config, prompt_build, shortcuts

_LOCK = threading.RLock()
_MAX_EXAMPLES = 300
_MAX_TEXT = 32768
_MAX_CAPTURES = 500
_STOP = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "is", "of", "on", "or", "the", "to", "with"}
_WORD = re.compile(r"[\w'-]+", re.UNICODE)
_EDIT_TOKEN = re.compile(r"[\w'-]+|[^\w\s]", re.UNICODE)
_SHOT = re.compile(r"\[Shot\s+\d+\]", re.IGNORECASE)
_CUT_TIME = re.compile(r"\bAt\s+\d{2}:\d{2}(?:\.\d+)?", re.IGNORECASE)
_nlp = None
_nlp_tried = False


def _path():
    return config.ROOT / "prompt_composer_memory.json"


def _read():
    try:
        value = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        value = {}
    if not isinstance(value, dict):
        value = {}
    examples = value.get("examples")
    if isinstance(examples, list):
        for row in examples[-_MAX_EXAMPLES:]:
            if isinstance(row, dict) and not isinstance(row.get("prompt_ratings"), dict):
                old = row.get("composition_ratings")
                row["prompt_ratings"] = {key: "bad" for key, vote in old.items()
                                          if vote == "bad"} if isinstance(old, dict) else {}
    return {"enabled": value.get("enabled") is True,
            "examples": examples[-_MAX_EXAMPLES:] if isinstance(examples, list) else [],
            "captures": value.get("captures", [])[-_MAX_CAPTURES:] if isinstance(value.get("captures"), list) else []}


def _write(data):
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _key(text):
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _nlp_model():
    global _nlp, _nlp_tried
    if not _nlp_tried:
        _nlp_tried = True
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except Exception:  # noqa: BLE001 -- analysis still works without spaCy/model
            _nlp = None
    return _nlp


def _body_text(body):
    body = body if isinstance(body, dict) else {}
    return prompt_build.build(
        str(body.get("text") or "")[:_MAX_TEXT],
        anchor=str(body.get("anchor") or "")[:_MAX_TEXT],
        postfix=str(body.get("postfix") or "")[:_MAX_TEXT],
        postfix_enabled=body.get("postfix_enabled") is not False,
        variables=body.get("variables") if isinstance(body.get("variables"), list) else None,
        seed=body.get("seed") if isinstance(body.get("seed"), int) and not isinstance(body.get("seed"), bool) else 1,
    )


def _postfix_text(body):
    """Resolve the project's tail by itself so learned edits can never rewrite it."""
    body = body if isinstance(body, dict) else {}
    if body.get("postfix_enabled") is False:
        return ""
    return prompt_build.build(
        "", postfix=str(body.get("postfix") or "")[:_MAX_TEXT],
        variables=body.get("variables") if isinstance(body.get("variables"), list) else None,
        seed=body.get("seed") if isinstance(body.get("seed"), int) and not isinstance(body.get("seed"), bool) else 1,
    )


def _remove_current_anchor(source, body):
    anchor = " ".join(str(body.get("anchor") or "").split())
    source = str(source or "").strip()
    if anchor and source.startswith(anchor) and (len(source) == len(anchor) or source[len(anchor)].isspace()):
        return source[len(anchor):].strip()
    return source


def _preserve_postfix(accepted, postfix):
    """Keep a resolved sound/postfix variable byte-for-byte at the end of a review."""
    if not postfix or accepted.endswith(postfix):
        return accepted
    # If the user removed only the end of the tail, replace that partial remnant
    # instead of duplicating it. The complete configured tail is always restored.
    overlap = min(len(postfix), len(accepted))
    while overlap and not accepted.endswith(postfix[:overlap]):
        overlap -= 1
    if overlap:
        accepted = accepted[:-overlap]
    return accepted.rstrip() + (" " if accepted.rstrip() else "") + postfix


def _shot_structure(text):
    text = str(text or "")
    markers = list(_SHOT.finditer(text))
    return {"shots": len(markers), "cut_times": len(_CUT_TIME.findall(text)),
            "shot_camera_ready": bool(markers)}


def _shot_camera_run(prompt_id):
    try:
        from ..shot_camera import memory as shot_memory
        run = shot_memory.run_for_prompt(prompt_id)
    except Exception:  # noqa: BLE001 -- optional module and old memory files are non-fatal
        return None
    if not isinstance(run, dict):
        return None
    arms = run.get("arms") if isinstance(run.get("arms"), list) else []
    views = run.get("views") if isinstance(run.get("views"), list) else []
    details = run.get("details") if isinstance(run.get("details"), list) else []
    return {"views": [str(row.get("view")) for row in views if isinstance(row, dict) and row.get("view")],
            "cuts_added": sum(str(arm) == "split:yes" for arm in arms),
            "camera_moves_added": sum(str(arm) == "move:yes" for arm in arms),
            "details_added": [str(row.get("phrase")) for row in details if isinstance(row, dict) and row.get("phrase")],
            "final_text": run.get("text") if isinstance(run.get("text"), str) else None}


def _shortcut_units(text, items=None):
    """Find the user's actual shortcut triggers in a source prompt, preserving order."""
    items = items if items is not None else [s for s in shortcuts.listing() if s.enabled]
    candidates = sorted(((trigger, item.name) for item in items
                         for trigger in item.triggers if trigger), key=lambda row: -len(row[0]))
    if not candidates:
        return []
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(trigger) for trigger, _name in candidates) + r")(?!\w)", re.IGNORECASE)
    names = {trigger.casefold(): name for trigger, name in candidates}
    return [names.get(match.group(0).casefold(), match.group(0)) for match in pattern.finditer(str(text or ""))]


def _shortcut_trigger(name, items):
    for item in items:
        if item.name.casefold() == str(name).casefold() and item.triggers:
            return item.triggers[0]
    return str(name)


def _training_counts(data):
    """Rating-weighted shortcut and ordering counts from recorded user runs."""
    items = [s for s in shortcuts.listing() if s.enabled]
    use = Counter()
    transitions = Counter()
    neutral = Counter()
    for row in data.get("captures", []):
        if not isinstance(row, dict):
            continue
        units = _shortcut_units(row.get("source_text"), items)
        rating = row.get("rating")
        weight = 2 if rating == "good" else -2 if rating == "bad" else 0
        for unit in set(units):
            if weight > 0:
                use[(unit.casefold(), "good")] += weight
            elif weight < 0:
                use[(unit.casefold(), "bad")] += -weight
            else:
                neutral[unit.casefold()] += 1
        for left, right in zip(units, units[1:]):
            if weight > 0:
                transitions[(left.casefold(), right.casefold(), "good")] += weight
            elif weight < 0:
                transitions[(left.casefold(), right.casefold(), "bad")] += -weight
    return item_by_name, use, transitions, neutral


def generate(body):
    """Build a prompt with rated shortcut order and replacement relevance; no language model."""
    body = body if isinstance(body, dict) else {}
    idea = str(body.get("text") or "")[:_MAX_TEXT].strip()
    items = [s for s in shortcuts.listing() if s.enabled and s.replacements and s.triggers]
    if not items:
        return {"why": "There are no enabled shortcuts with replacements yet.", "draft": _body_text(body), "generated": False}
    with _LOCK:
        data = _read()
        item_by_name, usage, transitions, neutral = _training_counts(data)
    units = _shortcut_units(idea, items)
    idea_words = {word.casefold() for word in _WORD.findall(idea) if word.casefold() not in _STOP}
    chosen = []
    reasons = []
    if units:
        chosen.extend(units)
    elif not idea:
        # With no seed idea, reuse the strongest liked source composition as the
        # starting grammar. It remains the user's macro prompt, including $vars.
        rated = [r for r in data.get("captures", []) if isinstance(r, dict) and r.get("rating") == "good" and r.get("source_text")]
        if rated:
            frequency = Counter(r.get("source_text") for r in rated)
            last_seen = {r.get("source_text"): i for i, r in enumerate(rated)}
            source = max(frequency, key=lambda value: (frequency[value], len(_shortcut_units(value, items)), last_seen[value]))
            idea = _remove_current_anchor(str(source or "")[:_MAX_TEXT], body)
            chosen.extend(_shortcut_units(idea, items))
            units = _shortcut_units(idea, items)
            reasons.append("started from a liked prompt structure")
        else:
            remembered = [r for r in data.get("captures", []) if isinstance(r, dict) and r.get("source_text")]
            if remembered:
                idea = _remove_current_anchor(str(remembered[-1].get("source_text") or "")[:_MAX_TEXT], body)
                chosen.extend(_shortcut_units(idea, items))
                units = _shortcut_units(idea, items)
                reasons.append("started from a recent prompt structure while ratings are still sparse")
    if not chosen and idea_words:
        candidates = []
        for item in items:
            replacement_words = {word.casefold() for phrase in item.replacements
                                 for word in _WORD.findall(phrase) if word.casefold() not in _STOP}
            overlap = len(idea_words & replacement_words) / max(1, len(idea_words | replacement_words) ** 0.5)
            good = usage[(item.name.casefold(), "good")]
            bad = usage[(item.name.casefold(), "bad")]
            score = overlap + 0.08 * (good - bad) + 0.015 * neutral[item.name.casefold()]
            if overlap > 0:
                candidates.append((score, item))
        candidates.sort(key=lambda row: (-row[0], row[1].name.casefold()))
        for _score, item in candidates[:2]:
            chosen.append(item.name)
        if candidates:
            reasons.append("matched shortcut actions to the idea and rating history")
    if chosen:
        last = chosen[-1].casefold()
        next_candidates = []
        for item in items:
            name = item.name.casefold()
            if name in {x.casefold() for x in chosen}:
                continue
            good = transitions[(last, name, "good")]
            bad = transitions[(last, name, "bad")]
            score = good - bad
            if good or bad:
                next_candidates.append((score, good, item))
        next_candidates.sort(key=lambda row: (-row[0], -row[1], row[2].name.casefold()))
        if next_candidates and next_candidates[0][0] > 0 and len(chosen) < 3:
            chosen.append(next_candidates[0][2].name)
            reasons.append("continued a shortcut order that received positive ratings")
    if not idea:
        # A cold start still gives a useful, editable draft from the available
        # library; rated runs will gradually replace this inventory-based pick.
        item = max(items, key=lambda row: (sum(len(x) for x in row.replacements), row.name.casefold()))
        chosen = [item.name]
        idea = _shortcut_trigger(item.name, items)
        units = list(chosen)
        reasons.append("cold start: selected from the enabled shortcut library")
    elif chosen:
        additions = chosen[len(units):]
        if additions:
            triggers = [_shortcut_trigger(name, items) for name in additions]
            idea = " ".join(part for part in (idea, " ".join(triggers)) if part).strip()
    base = _body_text({**body, "text": idea})
    return {"base": base, "draft": base, "generated": True, "source_text": idea,
            "shortcuts": chosen, "reasons": reasons,
            "training_runs": sum(isinstance(r, dict) and r.get("rating") in ("good", "bad") for r in data.get("captures", [])),
            "shot_structure": _shot_structure(base),
            "postfix_preserved": bool(_postfix_text(body))}


def _learned(base, data=None):
    data = data or _read()
    key = _key(base)
    for row in reversed(data["examples"]):
        if isinstance(row, dict) and row.get("key") == key:
            return row
    return None


def _prompt_votes(row):
    ratings = row.get("prompt_ratings", {}) if isinstance(row, dict) else {}
    if not isinstance(ratings, dict):
        return {"good": 0, "bad": 0}
    return {"good": sum(value == "good" for value in ratings.values()),
            "bad": sum(value == "bad" for value in ratings.values())}


def _needs_review(row):
    votes = _prompt_votes(row)
    return votes["bad"] - votes["good"] >= 2


def transform(prompt):
    """Optional core hook: replay a reviewed edit only for the exact same prompt."""
    with _LOCK:
        data = _read()
        if not data["enabled"]:
            return None
        row = _learned(prompt, data)
        # A single dislike can be seed luck. Repeated negative prompt feedback
        # stops automatic replay until the user reviews and saves a new edit.
        return (_preserve_postfix(row.get("accepted", prompt), row.get("postfix", ""))
                if row and not _needs_review(row) else None)


def draft(body):
    base = _body_text(body)
    with _LOCK:
        data = _read()
        row = _learned(base, data)
        votes = _prompt_votes(row)
        review_needed = row is not None and _needs_review(row)
        use_learned = data["enabled"] and row is not None and not review_needed
        accepted = _preserve_postfix(row.get("accepted", base), _postfix_text(body)) if use_learned else base
        return {"base": base, "draft": accepted,
                "learned": use_learned, "enabled": data["enabled"],
                "edits": row.get("edits", []) if use_learned else [],
                "prompt_votes": votes,
                "prompt_review_needed": review_needed,
                "shot_structure": _shot_structure(base),
                "postfix_preserved": bool(_postfix_text(body))}


def _edit_summary(before, after):
    a, b = _EDIT_TOKEN.findall(before), _EDIT_TOKEN.findall(after)
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            out.append({"change": tag, "before": " ".join(a[i1:i2]),
                        "after": " ".join(b[j1:j2]), "position": i1})
    return out[:100]


def learn(body):
    body = body if isinstance(body, dict) else {}
    accepted = body.get("accepted")
    if not isinstance(accepted, str) or not accepted.strip():
        raise ValueError("The reviewed prompt cannot be empty.")
    accepted = accepted[:_MAX_TEXT]
    base = _body_text(body)
    if not base.strip():
        raise ValueError("Enter a prompt idea before saving a learned edit.")
    accepted = _preserve_postfix(accepted, _postfix_text(body))
    row = {"key": _key(base), "base": base, "accepted": accepted,
           "postfix": _postfix_text(body),
           "edits": _edit_summary(base, accepted), "prompt_ratings": {}}
    with _LOCK:
        data = _read()
        data["examples"] = [r for r in data["examples"]
                            if not isinstance(r, dict) or r.get("key") != row["key"]]
        data["examples"].append(row)
        data["examples"] = data["examples"][-_MAX_EXAMPLES:]
        _write(data)
        return {"learned": True, "edit_count": len(row["edits"]),
                "examples": len(data["examples"]), "enabled": data["enabled"]}


def capture(prompt_id, prompt_text, source_text=None):
    """Keep the exact prompt sent to ComfyUI until its render is rated."""
    if not isinstance(prompt_id, str) or not prompt_id:
        return {"captured": False}
    prompt_text = str(prompt_text or "")[:_MAX_TEXT]
    if not prompt_text.strip():
        return {"captured": False}
    with _LOCK:
        data = _read()
        example = next((r for r in reversed(data["examples"])
                        if isinstance(r, dict) and r.get("accepted") == prompt_text), None)
        camera = _shot_camera_run(prompt_id)
        record = {"prompt_id": prompt_id, "text": prompt_text,
                  "source_text": str(source_text or "")[:_MAX_TEXT],
                  "rating": None,
                  "example_key": example.get("key") if example else None,
                  "example_text": example.get("accepted") if example else None,
                  "shot_structure": _shot_structure(prompt_text),
                  "shot_camera": {key: value for key, value in (camera or {}).items() if key != "final_text"}}
        data["captures"] = [r for r in data["captures"]
                            if not isinstance(r, dict) or r.get("prompt_id") != prompt_id]
        data["captures"].append(record)
        data["captures"] = data["captures"][-_MAX_CAPTURES:]
        _write(data)
        return {"captured": True, "linked": True,
                "shot_structure": record["shot_structure"],
                "shot_camera": record["shot_camera"]}


def rate(prompt_id, rating, axis=None):
    """Attach prompt-relevant ratings to the reviewed prompt that ran."""
    if not isinstance(prompt_id, str) or not prompt_id:
        return {"linked": False}
    with _LOCK:
        data = _read()
        capture_row = next((r for r in reversed(data["captures"])
                           if isinstance(r, dict) and r.get("prompt_id") == prompt_id), None)
        if not capture_row:
            return {"linked": False}
        value = ("good" if rating == "liked" else
                 "bad" if rating == "disliked" and axis != "image" else None)
        capture_row["rating"] = value
        example = next((r for r in data["examples"]
                        if isinstance(r, dict) and r.get("key") == capture_row.get("example_key")
                        and r.get("accepted") == capture_row.get("example_text")), None)
        if example:
            ratings = example.setdefault("prompt_ratings", {})
        else:
            ratings = None
        if ratings is not None and value == "good":
            ratings[prompt_id] = "good"
        elif ratings is not None and value == "bad":
            # A general dislike may include bad composition. An explicit bad
            # image is excluded because it does not identify a prompt problem.
            ratings[prompt_id] = "bad"
        elif ratings is not None:
            # Clearing or an image-only dislike removes this run's prompt vote.
            ratings.pop(prompt_id, None)
        _write(data)
        return {"linked": True, "prompt_votes": _prompt_votes(example) if example else {"good": 0, "bad": 0},
                "prompt_review_needed": _needs_review(example) if example else False,
                "shot_camera": capture_row.get("shot_camera", {})}


def set_enabled(enabled):
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be true or false")
    with _LOCK:
        data = _read()
        data["enabled"] = enabled
        _write(data)
        return status(data)


def clear():
    with _LOCK:
        data = {"enabled": False, "examples": [], "captures": []}
        _write(data)
        return status(data)


def status(data=None):
    data = data or _read()
    return {"enabled": data["enabled"], "examples": len(data["examples"]),
            "captured_prompts": len(data["captures"]),
            "prompt_likes": sum(_prompt_votes(row)["good"] for row in data["examples"] if isinstance(row, dict)),
            "prompt_dislikes": sum(_prompt_votes(row)["bad"] for row in data["examples"] if isinstance(row, dict)),
            "prompt_review_needed": sum(_needs_review(row) for row in data["examples"] if isinstance(row, dict)),
            "shot_camera_runs": sum(bool(row.get("shot_camera")) for row in data["captures"] if isinstance(row, dict)),
            "shot_views_added": sum(len(row.get("shot_camera", {}).get("views", [])) for row in data["captures"] if isinstance(row, dict)),
            "shot_cuts_added": sum(int(row.get("shot_camera", {}).get("cuts_added", 0)) for row in data["captures"] if isinstance(row, dict)),
            "shot_camera_moves_added": sum(int(row.get("shot_camera", {}).get("camera_moves_added", 0)) for row in data["captures"] if isinstance(row, dict)),
            "shot_details_added": sum(len(row.get("shot_camera", {}).get("details_added", [])) for row in data["captures"] if isinstance(row, dict)),
            "rated_prompt_runs": sum(row.get("rating") == "good" for row in data["captures"] if isinstance(row, dict)),
            "disliked_prompt_runs": sum(row.get("rating") == "bad" for row in data["captures"] if isinstance(row, dict))}


def analysis():
    items = [s for s in shortcuts.listing() if s.enabled and s.replacements]
    categories, phrases, words, patterns, positions, neighbors = Counter(), Counter(), Counter(), Counter(), {}, Counter()
    phrase_rows = []
    for item in items:
        category = " / ".join(x for x in (item.category, item.sub_category) if x) or "Uncategorised"
        categories[category] += 1
        for replacement in item.replacements:
            clean = " ".join(replacement.split())
            if not clean:
                continue
            phrases[clean.casefold()] += 1
            phrase_rows.append((clean, category))
            tokens = [x.casefold() for x in _WORD.findall(clean) if x.casefold() not in _STOP]
            words.update(tokens)
            for i, token in enumerate(tokens):
                bucket = "single" if len(tokens) == 1 else "start" if i == 0 else "end" if i == len(tokens) - 1 else "middle"
                positions.setdefault(token, Counter())[bucket] += 1
                if i:
                    neighbors[(tokens[i - 1], token)] += 1
            nlp = _nlp_model()
            if nlp is not None:
                pattern = " ".join(t.pos_ or "X" for t in nlp(clean) if not t.is_punct)
                if pattern:
                    patterns[pattern] += 1
    top_phrases = []
    seen = set()
    for phrase, category in sorted(phrase_rows, key=lambda r: (-phrases[r[0].casefold()], r[0].casefold())):
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        top_phrases.append({"text": phrase, "category": category, "count": phrases[key]})
        if len(top_phrases) == 12:
            break
    with _LOCK:
        state = status()
    return {"shortcuts": len(items), "categories": [{"name": k, "count": v} for k, v in categories.most_common(12)],
            "phrases": top_phrases, "words": [{"text": k, "count": v} for k, v in words.most_common(16)],
            "positions": [{"text": word, **counts} for word, counts in sorted(
                positions.items(), key=lambda item: (-sum(item[1].values()), item[0]))[:8]],
            "neighbors": [{"left": left, "right": right, "count": count}
                          for (left, right), count in neighbors.most_common(10)],
            "patterns": [{"pattern": k, "count": v} for k, v in patterns.most_common(10)],
            "spacy": _nlp is not None, **state}
