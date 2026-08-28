"""Ordering by declared relations, never by priority numbers.

A number is a global namespace: inserting one module between two others means
renumbering everything after it, and the numbers live in files that know nothing
about each other. A relation is local -- a module names only the neighbours it
actually cares about, and everything else sorts itself.
"""

from typing import Dict, Iterable, List, Set, Tuple

from . import log
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

    stage_of = {spec.id: STAGES.index(spec.stage) for spec in specs}

    # Edges point from "runs earlier" to "runs later". `before` is expressed as
    # the same edge from the other end, so the two spellings cannot disagree.
    edges: Dict[str, set] = {spec.id: set() for spec in specs}

    def _link(later: str, earlier: str) -> None:
        # Stage is the coarse order and a relation may not overturn it: a load
        # module cannot meaningfully run after a conditioning one, and honouring
        # that would put a whole stage out of sequence to satisfy one module.
        # The relation is dropped and said out loud rather than silently obeyed.
        if stage_of[later] < stage_of[earlier]:
            log.note(
                f"ignoring {later!r} after {earlier!r}: {later!r} is in stage "
                f"{STAGES[stage_of[later]]!r}, which runs before {STAGES[stage_of[earlier]]!r}"
            )
            return
        edges[later].add(earlier)

    for spec in specs:
        # A relation naming an absent module is not an error: modules come and
        # go, and "after audio_clock" has nothing to wait for when audio_clock
        # failed to load or was never installed.
        for earlier in spec.after:
            if earlier in by_id:
                _link(spec.id, earlier)
        for later in spec.before:
            if later in by_id:
                _link(later, spec.id)
    ordered: List[ModuleSpec] = []
    placed: set = set()

    # Stage first, then declaration order, so a run without any relations is
    # stable and readable rather than dependent on filesystem order.
    remaining = sorted(specs, key=lambda s: (stage_of[s.id], s.id))

    while remaining:
        ready = [s for s in remaining if edges[s.id] <= placed]
        if not ready:
            # Drop only the modules actually ON a cycle. Dropping everything
            # still waiting takes healthy modules with it -- one that merely
            # says "after B" is not itself circular, and once B is gone its
            # relation simply has nothing to wait for.
            live = {s.id for s in remaining}
            looping = _on_a_cycle(edges, live)
            for spec in [s for s in remaining if s.id in looping]:
                others = sorted(edges[spec.id] & looping)
                rejected.append((spec, f"ordering cycle with {', '.join(others)}"))
                remaining.remove(spec)
            # A dropped module stops being something to wait for. Otherwise
            # everything downstream of the cycle waits on an id that will never
            # arrive, and the collateral damage spreads instead of stopping.
            for waiting in edges.values():
                waiting -= looping
            if not looping:
                # Nothing is circular and nothing is ready: unreachable, but
                # looping forever would be worse than saying so.
                for spec in remaining:
                    rejected.append((spec, "could not be ordered"))
                break
            continue
        # Among the ready, respect stage so a later stage never jumps ahead.
        nxt = min(ready, key=lambda s: (stage_of[s.id], s.id))
        ordered.append(nxt)
        placed.add(nxt.id)
        remaining.remove(nxt)

    return ordered, rejected


def _on_a_cycle(edges: Dict[str, set], live: Set[str]) -> Set[str]:
    """Ids that can reach themselves through the edges still in play."""
    on_cycle: Set[str] = set()
    for start in live:
        seen: Set[str] = set()
        stack = [n for n in edges[start] if n in live]
        while stack:
            node = stack.pop()
            if node == start:
                on_cycle.add(start)
                break
            if node in seen:
                continue
            seen.add(node)
            stack.extend(n for n in edges.get(node, ()) if n in live)
    return on_cycle
