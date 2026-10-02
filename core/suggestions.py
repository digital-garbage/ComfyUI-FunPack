"""What a person's own prompting habits say about shortcuts: which ones land in
the same scene (pairs) and which shot follows which (follows). Mined on demand
by reading every saved project's scene text for triggers typed verbatim, so
there is nothing learned to store and nothing to go stale. Keyed by shortcut
name, which is what the picker identifies a shortcut by."""

from __future__ import annotations

import re

from . import projects, shortcuts


def stats() -> dict:
    pats = []
    for sc in shortcuts.listing():
        if not sc.enabled:
            continue
        for trig in sc.triggers:
            t = trig.strip().lower()
            if t:
                pats.append((sc.name, re.compile(r"(?<![a-z0-9_])" + re.escape(t) + r"(?![a-z0-9_])")))
    counts: dict[str, int] = {}
    pairs: dict[tuple, int] = {}
    follows: dict[tuple, int] = {}
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
            text = (sc.text or "").strip().lower()
            if not text:
                continue
            present = {k for k, pat in pats if pat.search(text)}
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
            "follows": [[a, b, n] for (a, b), n in follows.items()]}
