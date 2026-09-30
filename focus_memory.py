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
PICK_W = 0.9            # bonus a word the user has chosen grows toward (tanh of picks)
REJECT_W = 0.6          # penalty a word the rewriter proposed and the user replaced grows toward
MIN_SHOTS_FOR_RATE = 10 # shots reviewed before "how often the user wants a move" is trusted


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
    """{lemma: bonus}: words that keep coming back across prompts (once enough prompts were
    seen), plus what the user's own choices said: up for words they picked over the rewriter's
    proposal, down for the words they replaced."""
    data = _read()
    seen = data.get("seen") or {}
    out = {}
    if int(data.get("prompts", 0)) >= MIN_PROMPTS and seen:
        top = math.log1p(max(seen.values()))
        out = {lemma: FREQ_W * math.log1p(n) / top for lemma, n in seen.items()}
    for lemma, n in (data.get("picks") or {}).items():
        out[lemma] = out.get(lemma, 0.0) + PICK_W * math.tanh(n / 2.0)
    for lemma, n in (data.get("rejects") or {}).items():
        out[lemma] = out.get(lemma, 0.0) - REJECT_W * math.tanh(n / 2.0)
    return out


def learn(decisions):
    """Remember what a person chose. `decisions`: [{"auto": lemma|None, "picked": lemma|None,
    "mode": "auto"|"move"|"hold"|"none"}], one per shot shown. A picked word is reinforced; the
    word the rewriter proposed instead is marked down when they differ; "none" counts against
    moves in general (see move_rate). -> shots remembered."""
    data = _read()
    picks, rejects = data.get("picks") or {}, data.get("rejects") or {}
    total, kept = int(data.get("shots", 0)), int(data.get("kept", 0))
    n = 0
    for d in decisions or []:
        mode = d.get("mode") or "auto"
        n += 1
        total += 1
        kept += mode != "none"
        if mode == "none":
            continue
        picked, auto = d.get("picked"), d.get("auto")
        if picked:
            picks[picked] = picks.get(picked, 0) + 1
            if auto and auto != picked:
                rejects[auto] = rejects.get(auto, 0) + 1
    data.update(picks=picks, rejects=rejects, shots=total, kept=kept)
    _save(data)
    return n


def effective_chance(chance):
    """The configured chance of a move, pulled toward how often the user actually keeps one
    (of the shots they reviewed) once enough were reviewed."""
    data = _read()
    total, kept = int(data.get("shots", 0)), int(data.get("kept", 0))
    if total < MIN_SHOTS_FOR_RATE:
        return chance
    return max(0.05, min(1.0, (chance + kept / total) / 2.0))


def _save(data):
    path = _path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


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
    _save(data)
    return data["prompts"]
