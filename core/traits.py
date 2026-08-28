"""Compatibility by trait, never by model name.

A module listing model names has to be edited every time a model ships, and the
one nobody remembers to edit is the one that silently stops appearing. A model
says what it IS; a module says what it NEEDS.
"""

from typing import Iterable, List, Sequence, Set, Tuple

from .contract import ModuleSpec


def split(specs: Iterable[ModuleSpec], available: Sequence[str]) -> Tuple[List[ModuleSpec], List[ModuleSpec]]:
    """(compatible, incompatible) for the traits currently on offer."""
    have: Set[str] = set(available or ())
    compatible, incompatible = [], []
    for spec in specs:
        (compatible if set(spec.requires) <= have else incompatible).append(spec)
    return compatible, incompatible


def missing_for(spec: ModuleSpec, available: Sequence[str]) -> List[str]:
    """Which traits a module wanted and did not get. For the modules dump."""
    return sorted(set(spec.requires) - set(available or ()))
