"""Modifiers that belong to the sampler rather than to the model.

Some cannot ride on a ModelPatcher and it is not a limitation of the plumbing:
they need the step index, the schedule, or the chance to run the whole thing
twice. A model patch has none of those.

So a sampler offers HOOK POINTS by name and core hands it the modifiers that
asked for one it offers. Two properties fall out, and both are the point:

* A sampler file never names a modifier. Five samplers and six modifiers stay
  zero hand-written call sites rather than thirty.
* The sampler keeps the veto -- it says what it can host -- and core keeps the
  plumbing. A modifier wanting a hook this sampler does not offer is absent, and
  said, rather than half-working.

The call site is deliberately ONE, and shaped so it cannot grow:

    latent = chain.process(ctx, latent)

`ctx` carries the step, not the signature. v4 passed each technique's parameters
down as keyword arguments and `_sample_chunk` ended up with six ALG arguments
before anyone noticed the shape was wrong.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from . import log, patching


@dataclass(frozen=True)
class Step:
    """Everything a per-step modifier is allowed to know about where it is.

    A small closed object rather than loose arguments: adding a field here is
    one edit, whereas widening a call signature is an edit in every sampler and
    every modifier that ever passed it along.
    """

    index: int
    sigma: Any
    sigmas: Any
    total: int
    run: Optional[str] = None
    anchor: Any = None                 # the pinned latent, where there is one


class Chain:
    """The modifiers a sampler is actually running, in order."""

    def __init__(self, members: Sequence[Tuple[str, Any]], dropped: patching.Dropped):
        self.members = list(members)
        self.dropped = dropped

    def __bool__(self) -> bool:
        return bool(self.members)

    @property
    def ids(self) -> List[str]:
        return [name for name, _ in self.members]

    def process(self, ctx: Step, latent):
        """Every member that wants this step, in declared order.

        A member that raises is dropped for the rest of the run and the run
        continues -- the same rule the model-side hooks follow, for the same
        reason: a modifier is an opinion about the picture and is never worth
        the picture.
        """
        for name, member in self.members:
            key = f"funpack.{name}"
            if key in self.dropped:
                continue
            try:
                if not member.active(ctx):
                    continue
                changed = member.process(ctx, latent)
            except Exception as exc:             # noqa: BLE001
                if self.dropped.record(key, exc):
                    import traceback
                    log.warning(key, "failed during sampling and is now OFF for the "
                                     "rest of this run; the run continues without it\n"
                                     + "".join(traceback.format_exception(exc)).rstrip())
                continue
            if changed is not None:
                latent = changed
        return latent


def build(specs, values, accepts: Sequence[str], dropped: patching.Dropped,
          capability: str = "sampler_modifier") -> Tuple[Chain, List[str]]:
    """(chain, notes) for the modifiers this sampler can host.

    `specs` are already filtered by trait and ordered by relation -- this only
    decides who fits THIS sampler, and builds them.
    """
    offered = set(accepts or ())
    members, notes = [], []

    for spec in specs:
        make = spec.provides.get(capability)
        if make is None:
            continue

        wanted = tuple(getattr(spec, "hooks", ()) or ())
        missing = [hook for hook in wanted if hook not in offered]
        if missing:
            # Absent rather than half-working: a modifier that needs a second
            # pass cannot be usefully run by a sampler that makes one.
            notes.append(f"{spec.id}: needs {', '.join(missing)}, which this sampler "
                         f"does not offer")
            continue

        try:
            member = make(values.get(spec.id) or {}, tuple(offered))
        except Exception as exc:                 # noqa: BLE001
            log.broke(f"{spec.id}.{capability}", exc, "starting up for this run")
            notes.append(f"{spec.id}: failed to start -- {type(exc).__name__}: {exc}")
            continue

        if member is None:
            continue                             # the module decided it is off
        for required in ("active", "process"):
            if not callable(getattr(member, required, None)):
                notes.append(f"{spec.id}: is not a sampler modifier (no {required}())")
                member = None
                break
        if member is None:
            continue

        members.append((spec.id, member))
        notes.append(f"{spec.id}: on")

    return Chain(members, dropped), notes
