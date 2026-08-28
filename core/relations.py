"""Ordering by declared relations, never by priority numbers.

A number is a global namespace: inserting one module between two others means
renumbering everything after it, and the numbers live in files that know nothing
about each other. A relation is local -- a module names only the neighbours it
actually cares about, and everything else sorts itself.
"""

from typing import Dict, Iterable, List, Tuple

from .contract import STAGES, ModuleSpec


class CycleError(ValueError):
    """Relations that cannot be satisfied in any order."""


def order(specs: Iterable[ModuleSpec]) -> Tuple[List[ModuleSpec], List[Tuple[ModuleSpec, str]]]:
    """(ordered, rejected) -- rejected carries the reason it could not be placed.

    A module in a cycle is dropped rather than placed arbitrarily: an arbitrary
    order is one that works until it does not, and by then nobody remembers the
    order was never real.
    """
    specs = list(specs)
    by_id = {spec.id: spec for spec in specs}
    rejected: List[Tuple[ModuleSpec, str]] = []

    # Edges point from "runs earlier" to "runs later". `before` is expressed as
    # the same edge from the other end, so the two spellings cannot disagree.
    edges: Dict[str, set] = {spec.id: set() for spec in specs}
    for spec in specs:
        for earlier in spec.after:
            if earlier in by_id:
                edges[spec.id].add(earlier)
        for later in spec.before:
            if later in by_id:
                edges[later].add(spec.id)

    # A relation naming an absent module is not an error: modules come and go,
    # and "after audio_clock" simply has nothing to wait for when audio_clock
    # failed to load or was never installed.

    stage_of = {spec.id: STAGES.index(spec.stage) for spec in specs}
    ordered: List[ModuleSpec] = []
    placed: set = set()

    # Stage first, then declaration order, so a run without any relations is
    # stable and readable rather than dependent on filesystem order.
    remaining = sorted(specs, key=lambda s: (stage_of[s.id], s.id))

    while remaining:
        ready = [s for s in remaining if edges[s.id] <= placed]
        if not ready:
            for spec in remaining:
                unmet = sorted(edges[spec.id] - placed)
                rejected.append((spec, f"ordering cycle with {', '.join(unmet)}"))
            break
        # Among the ready, respect stage so a later stage never jumps ahead.
        nxt = min(ready, key=lambda s: (stage_of[s.id], s.id))
        ordered.append(nxt)
        placed.add(nxt.id)
        remaining.remove(nxt)

    return ordered, rejected
