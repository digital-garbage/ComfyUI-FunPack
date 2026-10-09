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
            # Editorial cuts share the root's text and rating: only the root owns them.
            if sc.gen_unit_id and sc.cut_offset_frames:
                continue
            # Every rating counts for the text it was given to (older ones carry none: the text as it is now): the
            # clip's own, and each earlier take's -- regenerating moves a rating there. A clip left out still was rated.
            now = proj.scene_renders.get(sc.id) if isinstance(proj.scene_renders.get(sc.id), dict) else {}
            on_clip = (lambda t: t.get("promptId") == now.get("promptId")) if now.get("promptId") else \
                (lambda t: (t.get("media") or {}).get("filename") == (now.get("media") or {}).get("filename"))
            takes = [(t.get("rating"), t.get("rated_text")) for t in proj.scene_variants.get(sc.id) or []
                     if isinstance(t, dict) and isinstance(t.get("rated_text"), str) and not (now and on_clip(t))]
            for label, text in [(sc.rating, sc.rated_text or sc.text), *takes]:
                v = vote(label)
                rated = fired(text) if v and str(text or "").strip() else ()
                for k in rated:
                    scores[k] = scores.get(k, 0) + v
                    for b in rated:
                        if k < b:
                            rated_pairs[(k, b)] = rated_pairs.get((k, b), 0) + v
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
            "scores": scores, "rated_pairs": [[a, b, n] for (a, b), n in rated_pairs.items()]}
