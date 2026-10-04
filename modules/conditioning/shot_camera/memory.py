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
import functools
import os
import threading

MIN_PROMPTS = 3
FREQ_W = 0.6            # max bonus: reorders near-equals, never outranks an owned part
MAX_HASHES = 300
PICK_W = 0.9            # bonus a word the user has chosen grows toward (tanh of picks)
REJECT_W = 0.6          # penalty a word the rewriter proposed and the user replaced grows toward
MIN_SHOTS_FOR_RATE = 10 # shots reviewed before "how often the user wants a move" is trusted
WORD_RATE_W = 0.5       # max bonus/penalty a word earns from the ratings of runs that aimed at it


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


_LOCK = threading.RLock()      # the worker thread (observe, record_run) and the rating route write the same file
_DICTS = ("picks", "rejects", "seen", "views", "arms", "runs", "rated")


def _locked(fn):
    @functools.wraps(fn)
    def run(*args, **kwargs):
        with _LOCK:
            return fn(*args, **kwargs)
    return run


def _read():
    try:
        with open(_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    for key in _DICTS:                     # a hand-edited file of the wrong shape reads as "nothing learned there"
        if key in data and not isinstance(data[key], dict):
            data.pop(key)
    for key in ("picks", "rejects", "seen"):
        if key in data:
            data[key] = {k: v for k, v in data[key].items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if not isinstance(data.get("hashes", []), list):
        data.pop("hashes")
    return data


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
    for key, (g, b) in arm_stats().items():       # what ratings taught about aiming at a word
        if key.startswith("word:"):
            lemma = key[5:]
            out[lemma] = out.get(lemma, 0.0) + WORD_RATE_W * 2.0 * (arm_rate(key) - 0.5)
    return out


@_locked
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
        picked = [picked] if isinstance(picked, str) else list(picked or [])
        for word in picked:
            picks[word] = picks.get(word, 0) + 1
        if picked and auto and auto not in picked:
            rejects[auto] = rejects.get(auto, 0) + 1
    data.update(picks=picks, rejects=rejects, shots=total, kept=kept)
    _save(data)
    return n


def view_stats():
    """{key: (good, bad)} for each view and each "view@trait" the ratings and picks taught."""
    return {k: (float(v[0]), float(v[1])) for k, v in (_read().get("views") or {}).items()
            if isinstance(v, (list, tuple)) and len(v) == 2}


def _bump_views(data, used, good, bad):
    # a negative amount undoes an earlier one, never below zero
    views = data.setdefault("views", {})
    for u in used:
        for key in [u["view"]] + [f"{u['view']}@{t}" for t in (u.get("traits") or ["none"])]:
            g, b = views.get(key, [0.0, 0.0])
            views[key] = [max(0.0, round(g + good, 4)), max(0.0, round(b + bad, 4))]


@_locked
def rate_views(used, sign, k=1.0):
    """A rated generation: its views share the credit (sign > 0) or the blame (sign < 0), so a
    run with four views does not punish each as hard as a run with one. `used`: [{"view",
    "traits"}]. -> views rated."""
    used = [u for u in used or [] if isinstance(u, dict) and u.get("view")]
    if not used or not sign:
        return 0
    data = _read()
    share = k / len(used)
    _bump_views(data, used, share if sign > 0 else 0.0, share if sign < 0 else 0.0)
    _save(data)
    return len(used)


@_locked
def learn_views(decisions):
    """What a person chose: [{"auto": view|None, "picked": view|None, "traits": [...]}]. A pick
    counts as a good sign for that view; the view the rewriter proposed instead, as half a bad."""
    data = _read()
    n = 0
    for d in decisions or []:
        tr = d.get("traits") or []
        picked = [d["picked"]] if isinstance(d.get("picked"), str) else list(d.get("picked") or [])
        if picked:
            for v in picked:
                _bump_views(data, [{"view": v, "traits": tr}], 1.0 / len(picked), 0.0)
            n += 1
            if d.get("auto") and d["auto"] not in picked:
                _bump_views(data, [{"view": d["auto"], "traits": tr}], 0.0, 0.5)
    _save(data)
    return n


def arm_stats():
    """{arm: (good, bad)} for what ratings taught about the camera's own choices: "move:yes" /
    "move:no" (a shot got a move / was left alone), "style:hold|travel|k1|k2|k3", "word:<lemma>",
    "split:yes|no" (a shot whose point changes was cut in two / kept whole)."""
    return {k: (float(v[0]), float(v[1])) for k, v in (_read().get("arms") or {}).items()
            if isinstance(v, (list, tuple)) and len(v) == 2}


def arm_rate(key, stats=None):
    """Smoothed good-rate of one arm: 0.5 until rated."""
    g, b = (stats if stats is not None else arm_stats()).get(key, (0.0, 0.0))
    return (g + 1.0) / (g + b + 2.0)


@_locked
def rate_arms(used, sign, k=1.0):
    """A rated generation: every choice the camera made in it shares the credit (sign > 0) or
    the blame (sign < 0), so a run with many choices does not punish each as hard as a run with
    one. `used`: [arm key]. -> arms rated."""
    used = [u for u in used or [] if isinstance(u, str) and u]
    if not used or not sign:
        return 0
    data = _read()
    arms = data.setdefault("arms", {})
    share = k / len(used)
    for u in used:
        g, b = arms.get(u, [0.0, 0.0])
        arms[u] = [max(0.0, round(g + (share if sign > 0 else 0.0), 4)), max(0.0, round(b + (share if sign < 0 else 0.0), 4))]
    _save(data)
    return len(used)


def summary(limit=40):
    """What is remembered, for the person to read and prune: words (picked / replaced / merely
    recurring) and views (good / bad, with the per-trait rows folded under their view)."""
    data = _read()
    picks, rejects, seen = (data.get("picks") or {}), (data.get("rejects") or {}), (data.get("seen") or {})
    words = [{"word": w, "picks": int(picks.get(w, 0)), "rejects": int(rejects.get(w, 0)), "seen": int(seen.get(w, 0))}
             for w in {*picks, *rejects, *seen}]
    words.sort(key=lambda r: -(r["picks"] * 3 + r["rejects"] * 3 + r["seen"]))
    views = [{"view": k, "good": g, "bad": b} for k, (g, b) in view_stats().items() if "@" not in k]
    views.sort(key=lambda r: -(r["good"] + r["bad"]))
    return {"words": words[:limit], "more": max(0, len(words) - limit), "views": views,
            "prompts": int(data.get("prompts", 0)), "shots": int(data.get("shots", 0)),
            "kept": int(data.get("kept", 0))}


@_locked
def forget(kind, name=None):
    """Drop one remembered word (`kind="word"`), one view and its per-trait rows (`"view"`), or
    everything (`"all"`). -> True if anything was removed."""
    data = _read()
    if kind == "all":
        existed = bool(data)
        data = {}
    elif kind == "word":
        existed = any(name in (data.get(k) or {}) for k in ("picks", "rejects", "seen"))
        for k in ("picks", "rejects", "seen"):
            (data.get(k) or {}).pop(name, None)
        existed = existed or (data.get("arms") or {}).pop(f"word:{name}", None) is not None
    elif kind == "view":
        views = data.get("views") or {}
        gone = [k for k in views if k == name or k.startswith(f"{name}@")]
        existed = bool(gone)
        for k in gone:
            views.pop(k)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    _save(data)
    return existed


def effective_chance(chance):
    """The configured chance of a move, pulled toward how often the user actually keeps one
    (of the shots they reviewed) once enough were reviewed."""
    data = _read()
    total, kept = int(data.get("shots", 0)), int(data.get("kept", 0))
    if total >= MIN_SHOTS_FOR_RATE:
        chance = (chance + kept / total) / 2.0
    return max(0.05, min(1.0, chance * _lean("move:yes", "move:no")))


def _lean(yes, no):
    """Factor (0.5-2) that tilts an odds toward the arm the ratings liked more, 1 when unrated."""
    stats = arm_stats()
    ry, rn = arm_rate(yes, stats), arm_rate(no, stats)
    return 2.0 * ry / (ry + rn)


def split_chance(chance):
    """The chance of cutting a shot in two, tilted by whether split runs were liked."""
    return max(0.0, min(1.0, chance * _lean("split:yes", "split:no")))


def _save(data):
    path = _path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


@_locked
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


MAX_RUNS = 64


@_locked
def record_run(prompt_id, chose):
    """What one run chose ({"views": [...], "arms": [...]}), kept until it is rated."""
    data = _read()
    runs = data.get("runs") or {}
    runs[str(prompt_id)] = chose
    data["runs"] = dict(list(runs.items())[-MAX_RUNS:])
    _save(data)


def _sign(rating, axis):
    """+1 liked; -1 disliked for the camera; 0 for no rating, or a dislike that blames the picture alone."""
    return 1 if rating == "liked" else -1 if rating == "disliked" and axis != "image" else 0


@_locked
def on_rating(prompt_id, rating, axis=None):
    """A clip was rated: the views and camera choices of the run that made it share the credit or the blame.
    Changing or clearing the rating takes the earlier lesson back first, so the last word counts and a
    forgotten rating leaves nothing behind. -> choices now rated."""
    data = _read()
    pid = str(prompt_id)
    run = (data.get("runs") or {}).get(pid)
    if not isinstance(run, dict):
        return 0
    before, sign = (data.get("rated") or {}).get(pid, 0), _sign(rating, axis)
    if before == sign:
        return 0
    if before:
        rate_views(run.get("views"), before, -1.0)
        rate_arms(run.get("arms"), before, -1.0)
    n = (rate_views(run.get("views"), sign) + rate_arms(run.get("arms"), sign)) if sign else 0
    data = _read()
    data.setdefault("rated", {})[pid] = sign
    data["rated"] = dict(list(data["rated"].items())[-MAX_RUNS:])
    _save(data)
    return n
