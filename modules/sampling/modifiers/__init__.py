"""The modifier nodes. A node-only module."""

from .nodes import FunPackLoadModifiers, FunPackModifierSettings

ID = "sampling_modifiers"
TITLE = "Modifiers"
STAGE = "sampling"
STATUS = "proven"

NODES = [FunPackModifierSettings, FunPackLoadModifiers]
