"""What a person's own prompting habits say about shortcuts: which ones land in
the same scene (pairs), which shot follows which (follows), and how the scenes
using them were rated (scores, rated_pairs: how much likelier to pick, see weigh). Mined on demand
by reading every saved project's scene text for the triggers the expander would replace, so
there is nothing learned to store and nothing to go stale. Keyed by shortcut
name, which is what the picker identifies a shortcut by."""

from __future__ import annotations

import math
import re

from . import projects, shortcuts

_WORD = re.compile(r"[^\W_][\w'’-]*")        # JS twin in shell/habits.js words()


def words(text: str) -> set[str]:
    return {w.casefold() for w in _WORD.findall(text or "")}


def guess(rated) -> dict:
    """Which words go with a like: per-word log-odds (naive Bayes) over (text, +1/-1) votes. It owns up to how good
    it is: each distinct prompt's votes are guessed with that prompt left out, against always guessing the commoner
    vote. `ready` only when it clearly beats that, so a guess that knows nothing stays off."""
    docs = [(words(t), v) for t, v in rated]
    pos: dict[str, int] = {}
    neg: dict[str, int] = {}
    for ws, v in docs:
        for w in ws:
            (pos if v > 0 else neg)[w] = (pos if v > 0 else neg).get(w, 0) + 1
    likes = sum(1 for _, v in docs if v > 0)

    def score(ws, P, N, L, D):
        return math.log((L + 1) / (D + 1)) + sum(math.log((P.get(w, 0) + 1) / (L + 2)) - math.log((N.get(w, 0) + 1) / (D + 2)) for w in ws)

    groups: dict[str, list] = {}
    for (t, _), d in zip(rated, docs):
        groups.setdefault(" ".join((t or "").casefold().split()), []).append(d)
    right = 0
    for held in groups.values():                     # leave one prompt out: its seeds' votes never grade themselves
        P, N, L = dict(pos), dict(neg), likes
        for ws, v in held:
            C = P if v > 0 else N
            for w in ws:
                C[w] -= 1
            L -= v > 0
        D = len(docs) - len(held) - L
        for ws, v in held:
            right += (score(ws, P, N, L, D) > 0) == (v > 0)
    n = len(docs)
    base = max(likes, n - likes)        # not left-out: a left-out majority tips against the vote it leaves out
    # 3 standard deviations of a coin-flip count (sqrt(n)/2 each): noise turns it on about 1 time in 1000.
    ready = len(groups) >= 5 and right - base >= max(2, 1.5 * math.sqrt(n))
    D = n - likes
    weights = {w: round(math.log((pos.get(w, 0) + 1) / (likes + 2)) - math.log((neg.get(w, 0) + 1) / (D + 2)), 4)
               for w in pos.keys() | neg.keys()} if ready else {}
    return {"ready": ready, "right": right, "of": n, "prompts": len(groups), "weights": weights}


def vote(label: str) -> int:
    """A scene's saved rating as +1 / -1 / 0. A bad-image dislike blames the picture, not the prompt: 0."""
    v = str(label or "").removesuffix("|loved").strip()
    if v.isascii() and v.isdigit():
        return 1 if int(v) >= 6 else -1
    return -1 if v == "Disliked: bad composition" else 0


def weigh(votes) -> float:
    """How much likelier a shortcut (or pair) is to be picked, from its votes oldest first (1 = never rated): each like
    x1.5; a dislike x0.5, the next one in a row x0.25, then x0.125... A like ends the run. Nothing reaches zero."""
    w, run = 1.0, 0
    for v in votes:
        if v > 0:
            w, run = w * 1.5, 0
        else:
            run += 1
            w = max(w * 0.5 ** run, 1e-12)        # a long run would round to 0: a ban, which no rating is
    return w


def stats() -> dict:
    fired = shortcuts.matcher()
    counts: dict[str, int] = {}
    pairs: dict[tuple, int] = {}
    follows: dict[tuple, int] = {}
    votes: dict = {}                  # shortcut name, or (a, b) pair -> [(when, +1/-1)]
    voted: list = []                  # (text, +1/-1): what the like-guess learns from
    scanned = 0
    for meta in projects.listing():
        proj = projects.get(meta["id"])
        if proj is None:
            continue
        prev: set[str] | None = None
        by_id = {sc.id: sc for sc in proj.scenes}
        # Timeline (cut) order, as v4 mined it; the plan order until a clip was moved by hand.
        order = [i for i in proj.timeline_order if i in by_id] or [sc.id for sc in proj.scenes]
        for sid in order:
            sc = by_id[sid]
            # Editorial cuts share the root's text and rating: only the root owns them.
            if sc.gen_unit_id and sc.cut_offset_frames:
                continue
            # Every rating counts for the text it was given to (older ones carry none: the text as it is now): the
            # clip's own, and each earlier take's -- regenerating moves a rating there. A clip left out still was rated.
            now = proj.scene_renders.get(sc.id) if isinstance(proj.scene_renders.get(sc.id), dict) else {}
            on_clip = (lambda t: t.get("promptId") == now.get("promptId")) if now.get("promptId") else \
                (lambda t: (t.get("media") or {}).get("filename") == (now.get("media") or {}).get("filename"))
            takes = [(t.get("rating"), t.get("rated")) for t in proj.scene_variants.get(sc.id) or []
                     if isinstance(t, dict) and isinstance(t.get("rated"), dict) and not (now and on_clip(t))]
            for label, rated in [(sc.rating, sc.rated or {"text": sc.text}), *takes]:
                v, text, at = vote(label), rated.get("text"), rated.get("at")
                ok = v and isinstance(text, str) and text.strip()
                hit = fired(text) if ok else ()
                if ok:
                    voted.append((text, v))
                when = at if isinstance(at, (int, float)) and not isinstance(at, bool) else 0      # undated: oldest
                for k in hit:
                    votes.setdefault(k, []).append((when, v))
                    for b in hit:
                        if k < b:
                            votes.setdefault((k, b), []).append((when, v))
            # Habits: what the cut uses now.
            if sc.excluded or not (sc.text or "").strip():
                continue
            present = fired(sc.text)
            scanned += 1
            for k in present:
                counts[k] = counts.get(k, 0) + 1
            for a in present:
                for b in present:
                    if a < b:
                        pairs[(a, b)] = pairs.get((a, b), 0) + 1
            for a in prev or ():
                for b in present:
                    follows[(a, b)] = follows.get((a, b), 0) + 1
            prev = present
    return {"scenes": scanned, "counts": counts,
            "pairs": [[a, b, n] for (a, b), n in pairs.items()],
            "follows": [[a, b, n] for (a, b), n in follows.items()],
            "scores": {k: weigh(v for _, v in sorted(h, key=lambda x: x[0])) for k, h in votes.items() if isinstance(k, str)},
            "guess": guess(voted),
            "rated_pairs": [[*k, weigh(v for _, v in sorted(h, key=lambda x: x[0]))] for k, h in votes.items() if isinstance(k, tuple)]}
