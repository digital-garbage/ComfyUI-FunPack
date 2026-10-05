"""What a module declares, and the vocabulary it may use.

Core holds no list of features. A module announces itself and this file says
what a valid announcement looks like -- nothing here names an implementation.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Bumped when the shape of an announcement changes. A module declaring an older
# version is not quietly adapted: it is refused, because guessing at what an old
# declaration meant is how silent behaviour changes happen.
CONTRACT_VERSION = 1

# Closed set. A type outside it has no renderer and no validation, so accepting
# one would mean rendering something approximate.
# A setting is a PREFERENCE. Anything that is data -- an image, a drawn mask,
# a video -- travels as a ComfyUI socket instead, because it is a tensor the app
# uploads and the graph wires, not a value that belongs in a JSON payload. That
# is why there is no "image" or "mask" here despite both being planned features.
TYPES = frozenset({"bool", "int", "float", "enum", "text", "multiline", "path", "color"})

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
    "color": ["swatch"],
}

# Ordering is coarse-grained by stage, then by declared relations within it.
# Numbers are deliberately absent: a priority number is a global namespace, so
# inserting one module means renumbering the others.
STAGES: List[str] = ["load", "conditioning", "latent", "guidance", "sampling", "post"]

# Presentation grouping, independent of STAGE: stage is execution order, category
# is "what is this module FOR" -- a module can be stage="sampling" and
# category="continuity" at once. Closed set for the same reason TYPES is closed:
# an unrecognised category has no place in the UI to render into. Empty/absent is
# valid and means "uncategorised", never a hard error -- most modules will not
# need this until the panel that groups by it actually exists.
CATEGORIES = frozenset({"continuity", "guidance", "conditioning", "sampling", "post", "system"})


@dataclass(frozen=True)
class ModuleSpec:
    """One module's announcement, after validation."""

    id: str
    title: str
    mount: str
    settings: Dict[str, dict] = field(default_factory=dict)
    requires: List[str] = field(default_factory=list)   # model traits
    uses: List[str] = field(default_factory=list)       # capabilities this module asks other modules for
    uses_when: Dict[str, Any] = field(default_factory=dict)   # its own setting values under which it asks (USES_WHEN absent: whenever on)
    roles: List[str] = field(default_factory=list)      # pipeline inputs (role `at`) it acts through: inert in a pipeline with none
    after: List[str] = field(default_factory=list)      # module ids
    before: List[str] = field(default_factory=list)
    stage: str = "sampling"
    category: str = ""                                   # "" means uncategorised
    # ComfyUI node classes this module contributes. Kept out of `settings` on
    # purpose: a setting is a preference, a node is graph structure, and letting
    # a new setting change a node's socket list would rot every saved workflow.
    nodes: List[type] = field(default_factory=list)
    # A callable(model) -> iterable[str], contributing traits core cannot read on
    # its own. This is how a model's own module teaches the system to recognise
    # it, so supporting a new model is a new folder and never an edit to core.
    traits: Optional[Callable] = None
    # Named capabilities this module offers to OTHER modules. Core never reads a
    # name out of here and never defines one: it is a lookup, so a node can ask
    # "who can build one of these" without core learning what the thing is.
    provides: Dict[str, Callable] = field(default_factory=dict)
    # Hook points a sampler must offer before this module can run. Empty means
    # "any sampler will do", the same narrowing rule traits follow.
    hooks: List[str] = field(default_factory=list)
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
        # `nodes` as their ids: a module whose settings are read only by its own node is inert, so
        # hidden, in a pipeline that does not contain that node.
        return {
            "id": self.id,
            "title": self.title,
            "mount": self.mount,
            "settings": self.settings,
            "requires": list(self.requires),
            "uses": list(self.uses),
            "uses_when": dict(self.uses_when),
            "roles": list(self.roles),
            "after": list(self.after),
            "before": list(self.before),
            "stage": self.stage,
            "category": self.category,
            "ui": self.ui,
            "status": self.status,
            "nodes": [_node_id(n) for n in self.nodes],
        }


def _node_id(node) -> str:
    """A node's id for the browser. A node whose schema fails is not registered (core/nodes.py says why);
    here it must not take the whole module list down with it."""
    try:
        return getattr(node.GET_SCHEMA(), "node_id", None) or node.__name__
    except Exception:                                    # noqa: BLE001
        return getattr(node, "__name__", "?")
