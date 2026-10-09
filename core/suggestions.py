"""What a person's own prompting habits say about shortcuts: which ones land in
the same scene (pairs), which shot follows which (follows), and how the scenes
using them were rated (scores, rated_pairs: liked +1, disliked -1). Mined on demand
by reading every saved project's scene text for the triggers the expander would replace, so
there is nothing learned to store and nothing to go stale. Keyed by shortcut
name, which is what the picker identifies a shortcut by."""

from __future__ import annotations

from . import projects, shortcuts


def vote(label: str) -> int:
    """A scene's saved rating as +1 / -1 / 0. A bad-image dislike blames the picture, not the prompt: 0."""
    v = str(label or "").removesuffix("|loved").strip()
    if v.isdigit():
        return 1 if int(v) >= 6 else -1
    return -1 if v == "Disliked: bad composition" else 0


def stats() -> dict:
    fired = shortcuts.matcher()
    counts: dict[str, int] = {}
    pairs: dict[tuple, int] = {}
    follows: dict[tuple, int] = {}
    scores: dict[str, int] = {}
    rated_pairs: dict[tuple, int] = {}
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
            # Editorial cuts share the root's text: only the root owns the prompt.
            if sc.excluded or (sc.gen_unit_id and sc.cut_offset_frames):
                continue
            if not (sc.text or "").strip():
                continue
            present = fired(sc.text)
            scanned += 1
            v = vote(sc.rating)
            # A rating is about the text it was given to (older ratings carry none: the text as it is now).
            rated = fired(sc.rated_text) if v and sc.rated_text.strip() else present
            for k in present:
                counts[k] = counts.get(k, 0) + 1
            for a in present:
                for b in present:
                    if a < b:
                        pairs[(a, b)] = pairs.get((a, b), 0) + 1
            for k in rated if v else ():
                scores[k] = scores.get(k, 0) + v
                for b in rated:
                    if k < b:
                        rated_pairs[(k, b)] = rated_pairs.get((k, b), 0) + v
            for a in prev or ():
                for b in present:
                    follows[(a, b)] = follows.get((a, b), 0) + 1
            prev = present
    return {"scenes": scanned, "counts": counts,
            "pairs": [[a, b, n] for (a, b), n in pairs.items()],
            "follows": [[a, b, n] for (a, b), n in follows.items()],
            "scores": scores, "rated_pairs": [[a, b, n] for (a, b), n in rated_pairs.items()]}
