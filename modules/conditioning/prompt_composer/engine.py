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
        return row.get("accepted") if row and not _needs_review(row) else None


def draft(body):
    base = _body_text(body)
    with _LOCK:
        data = _read()
        row = _learned(base, data)
        votes = _prompt_votes(row)
        review_needed = row is not None and _needs_review(row)
        use_learned = data["enabled"] and row is not None and not review_needed
        return {"base": base, "draft": row["accepted"] if use_learned else base,
                "learned": use_learned, "enabled": data["enabled"],
                "edits": row.get("edits", []) if use_learned else [],
                "prompt_votes": votes,
                "prompt_review_needed": review_needed}


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
    row = {"key": _key(base), "base": base, "accepted": accepted,
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


def capture(prompt_id, prompt_text):
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
        record = {"prompt_id": prompt_id, "text": prompt_text,
                  "example_key": example.get("key") if example else None,
                  "example_text": example.get("accepted") if example else None}
        data["captures"] = [r for r in data["captures"]
                            if not isinstance(r, dict) or r.get("prompt_id") != prompt_id]
        data["captures"].append(record)
        data["captures"] = data["captures"][-_MAX_CAPTURES:]
        _write(data)
        return {"captured": True, "linked": bool(record["example_key"])}


def rate(prompt_id, rating, axis=None):
    """Attach prompt-relevant ratings to the reviewed prompt that ran."""
    if not isinstance(prompt_id, str) or not prompt_id:
        return {"linked": False}
    with _LOCK:
        data = _read()
        capture_row = next((r for r in reversed(data["captures"])
                           if isinstance(r, dict) and r.get("prompt_id") == prompt_id), None)
        if not capture_row or not capture_row.get("example_key"):
            return {"linked": False}
        example = next((r for r in data["examples"]
                        if isinstance(r, dict) and r.get("key") == capture_row["example_key"]
                        and r.get("accepted") == capture_row.get("example_text")), None)
        if not example:
            return {"linked": False}
        ratings = example.setdefault("prompt_ratings", {})
        if rating == "liked":
            ratings[prompt_id] = "good"
        elif rating == "disliked" and axis != "image":
            # A general dislike may include bad composition. An explicit bad
            # image is excluded because it does not identify a prompt problem.
            ratings[prompt_id] = "bad"
        else:
            # Clearing or an image-only dislike removes this run's prompt vote.
            ratings.pop(prompt_id, None)
        _write(data)
        return {"linked": True, "prompt_votes": _prompt_votes(example),
                "prompt_review_needed": _needs_review(example)}


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
            "prompt_review_needed": sum(_needs_review(row) for row in data["examples"] if isinstance(row, dict))}


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
