"""Which words the user's prompts keep dwelling on, for choosing where a camera move aims.

Each candidate target already gets a score from its own prompt (an owned part beats a bare
object, objects of a verb beat doers). This adds a second: how many DIFFERENT prompts the word
turned up in. "It keeps coming back, so it matters" is fused into the first score as a bonus.
Silent until MIN_PROMPTS prompts have been seen: one prompt's words are not a habit. Kept per
refinement key in `<key>.focus_memory.json` (keys are disposable; nothing is migrated). A prompt
seen again counts nothing, so regenerating does not inflate a word.
"""

import json
import math
import os

MIN_PROMPTS = 3
FREQ_W = 0.6            # max bonus: reorders near-equals, never outranks an owned part
MAX_HASHES = 300


def _path(key):
    try:
        from .conditioning import refinement_state_path
    except ImportError:
        from conditioning import refinement_state_path
    return refinement_state_path(key, "focus_memory", prefix="refine_v2", extension="json")


def _read(key):
    try:
        with open(_path(key), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError):
        return {}


def prior(key):
    """{lemma: bonus} for `key`, or {} while too few prompts have been seen."""
    if not key:
        return {}
    data = _read(key)
    seen = data.get("seen") or {}
    if int(data.get("prompts", 0)) < MIN_PROMPTS or not seen:
        return {}
    top = math.log1p(max(seen.values()))
    return {lemma: FREQ_W * math.log1p(n) / top for lemma, n in seen.items()}


def observe(key, prompt_hash, lemmas):
    """Count each of `lemmas` once for this prompt. -> prompts seen so far (0 = nothing kept)."""
    if not key or not lemmas:
        return 0
    data = _read(key)
    hashes = data.get("hashes") or []
    if prompt_hash in hashes:
        return int(data.get("prompts", 0))
    seen = data.get("seen") or {}
    for lemma in set(lemmas):
        seen[lemma] = seen.get(lemma, 0) + 1
    data.update(seen=seen, prompts=int(data.get("prompts", 0)) + 1,
                hashes=(hashes + [prompt_hash])[-MAX_HASHES:])
    path = _path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return data["prompts"]
