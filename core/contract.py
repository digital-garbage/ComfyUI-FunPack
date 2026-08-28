"""What a module declares, and the vocabulary it may use.

Core holds no list of features. A module announces itself and this file says
what a valid announcement looks like -- nothing here names an implementation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Bumped when the shape of an announcement changes. A module declaring an older
# version is not quietly adapted: it is refused, because guessing at what an old
# declaration meant is how silent behaviour changes happen.
CONTRACT_VERSION = 1

# Closed set. A type outside it has no renderer and no validation, so accepting
# one would mean rendering something approximate.
TYPES = frozenset({"bool", "int", "float", "enum", "text", "multiline", "path"})

NUMERIC = frozenset({"int", "float"})

# Which renderers each type allows, first being the default. A hint outside its
# type's list is refused at import: the alternative is a panel that silently
# renders something other than what was asked for.
UI_HINTS: Dict[str, List[str]] = {
    "bool": ["checkboxRow", "toggle"],
    "int": ["number", "slider", "stepper", "macroSlider"],
    "float": ["number", "slider", "stepper", "macroSlider"],
    "enum": ["select", "segmented", "radioGroup", "filterList", "wheel"],
    "text": ["input", "search"],
    "multiline": ["textarea", "autoTextarea"],
    "path": ["filterList"],
}

# Ordering is coarse-grained by stage, then by declared relations within it.
# Numbers are deliberately absent: a priority number is a global namespace, so
# inserting one module means renumbering the others.
STAGES: List[str] = ["load", "conditioning", "latent", "guidance", "sampling", "post"]


@dataclass(frozen=True)
class ModuleSpec:
    """One module's announcement, after validation."""

    id: str
    title: str
    mount: str
    settings: Dict[str, dict] = field(default_factory=dict)
    requires: List[str] = field(default_factory=list)   # model traits
    after: List[str] = field(default_factory=list)      # module ids
    before: List[str] = field(default_factory=list)
    stage: str = "sampling"
    ui: Optional[str] = None                            # served path to its ui.js
    status: str = "experimental"                        # or "proven"
    source: str = ""                                    # dotted import path

    def defaults(self) -> Dict[str, Any]:
        """The values a headless run gets when no panel has been rendered.

        Derived from the same declaration the panel renders, so the two cannot
        disagree about what a setting means when nobody has touched it.
        """
        return {key: spec["default"] for key, spec in self.settings.items()}

    def to_manifest(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "mount": self.mount,
            "settings": self.settings,
            "requires": list(self.requires),
            "after": list(self.after),
            "before": list(self.before),
            "stage": self.stage,
            "ui": self.ui,
            "status": self.status,
        }
