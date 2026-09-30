"""Which words a user's prompts keep dwelling on, for choosing where a camera move aims.

Each candidate target already gets a score from its own prompt (an owned part beats a bare
object, objects of a verb beat doers). This adds a second: how many DIFFERENT prompts the word
turned up in. "It keeps coming back, so it matters" is fused into the first score as a bonus.
Silent until MIN_PROMPTS prompts have been seen: one prompt's words are not a habit. A prompt
seen again counts nothing, so regenerating does not inflate a word.

Self-contained: no dependency on the host pack. One store per user, at
`<ComfyUI user dir>/shot_camera/focus_memory.json` (else `~/.shot_camera/focus_memory.json`);
the SHOT_CAMERA_MEMORY environment variable names another file. Delete the file to forget.
"""

import json
import math
import os

MIN_PROMPTS = 3
FREQ_W = 0.6            # max bonus: reorders near-equals, never outranks an owned part
MAX_HASHES = 300


def _path():
    override = os.environ.get("SHOT_CAMERA_MEMORY")
    if override:
        return override
    try:
        import folder_paths
        base = folder_paths.get_user_directory()
    except Exception:  # noqa: BLE001 — not inside ComfyUI, or an older one
        base = os.path.join(os.path.expanduser("~"), ".shot_camera")
        return os.path.join(base, "focus_memory.json")
    return os.path.join(base, "shot_camera", "focus_memory.json")


def _read():
    try:
        with open(_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def prior():
    """{lemma: bonus}, or {} while too few prompts have been seen."""
    data = _read()
    seen = data.get("seen") or {}
    if int(data.get("prompts", 0)) < MIN_PROMPTS or not seen:
        return {}
    top = math.log1p(max(seen.values()))
    return {lemma: FREQ_W * math.log1p(n) / top for lemma, n in seen.items()}


def observe(prompt_hash, lemmas):
    """Count each of `lemmas` once for this prompt. -> prompts seen so far (0 = nothing kept)."""
    if not lemmas:
        return 0
    data = _read()
    hashes = data.get("hashes") or []
    if prompt_hash in hashes:
        return int(data.get("prompts", 0))
    seen = data.get("seen") or {}
    for lemma in set(lemmas):
        seen[lemma] = seen.get(lemma, 0) + 1
    data.update(seen=seen, prompts=int(data.get("prompts", 0)) + 1,
                hashes=(hashes + [prompt_hash])[-MAX_HASHES:])
    path = _path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return data["prompts"]
