"""The modifier nodes. A node-only module."""

from .nodes import FunPackLoadModifiers, FunPackModifierSettings

ID = "sampling_modifiers"
TITLE = "Modifiers"
STAGE = "sampling"
STATUS = "proven"

NODES = [FunPackModifierSettings, FunPackLoadModifiers]


def settings_sink():
    """Where the values a person picked belong in a graph.

    Announced rather than known: core places the app's settings into whatever a
    module says accepts them, so this node is not privileged, it is merely the
    one that volunteered.
    """
    return {"node": "FunPackModifierSettings", "input": "settings"}


PROVIDES = {"settings_sink": settings_sink}
