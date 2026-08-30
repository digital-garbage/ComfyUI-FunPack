"""What a node's inputs look like to a person editing them.

`graph.py` reads a node's inputs to decide whether a wire is legal: it needs the
TYPE and nothing else. Editing a value needs the rest -- the choices a combo
offers, the bounds on a number, the default, whether the box is multiline -- and
that is a different question about the same declaration, so it is asked here
rather than widening the one graph.py asks.

Read from ComfyUI's own INPUT_TYPES, the same source its own frontend uses, so a
node written for ComfyUI is editable here without knowing anything about
FunPack. Nothing in this file names a node.

A socket is not a widget. An input whose type is a socket type (MODEL, LATENT,
IMAGE...) is filled by a wire, and offering a text box for it would invite a
value the graph builder then refuses -- so those are reported, and reported as
sockets, rather than made editable.
"""

from typing import Any, Dict, List, Optional

# Declared by a list of choices rather than a type name: the list IS the type.
COMBO = "COMBO"

# The types a person types INTO. Everything else is filled by a wire.
#
# A short list core holds, deliberately: these five are ComfyUI's primitives and
# the same set its own frontend edits, so a node written for ComfyUI is editable
# here. Deciding by case instead ("socket types are upper case") looked
# type-agnostic and was simply wrong -- STRING is upper case, and every prompt
# box in the pipeline came back as a socket nobody could type in.
PRIMITIVE = frozenset({"STRING", "INT", "FLOAT", "BOOLEAN", COMBO})

# What a widget's options dict may say that is worth carrying to a form. Named
# rather than passed through wholesale: an options dict can hold anything a node
# author put there, and forwarding all of it makes the payload a node's private
# business rather than a contract.
KEPT = ("default", "min", "max", "step", "round", "multiline", "dynamicPrompts",
        "tooltip", "placeholder", "control_after_generate", "image_upload",
        "label_on", "label_off", "precision", "forceInput")


def _one(name: str, declared: Any, required: bool) -> Dict[str, Any]:
    kind = declared[0] if isinstance(declared, (list, tuple)) and declared else declared
    options = {}
    if isinstance(declared, (list, tuple)) and len(declared) > 1 and isinstance(declared[1], dict):
        options = declared[1]

    widget: Dict[str, Any] = {"name": name, "required": required}
    if isinstance(kind, str):
        widget["type"] = kind
    else:
        # A list of choices. It may be empty -- a file picker on a machine with
        # no files -- and that is worth saying rather than rendering a select
        # with nothing in it and no explanation.
        widget["type"] = COMBO
        widget["choices"] = list(kind) if isinstance(kind, (list, tuple)) else []

    for key in KEPT:
        if key in options:
            widget[key] = options[key]
    return widget


def describe(class_type: str) -> Optional[dict]:
    """One node, as a form: its widgets, its sockets and what it is called.

    None when no such node is installed -- which is not an error here. A slot may
    point at a node from a pack that is not present, and saying so is the job of
    whoever assembled the graph.
    """
    try:
        import nodes as comfy_nodes
    except Exception:                            # noqa: BLE001 -- not inside ComfyUI
        return None

    node = comfy_nodes.NODE_CLASS_MAPPINGS.get(class_type)
    if node is None:
        return None

    try:
        spec = node.INPUT_TYPES()
    except Exception as exc:                     # noqa: BLE001
        # A node whose declaration raises is a node nobody can edit. Said, not
        # swallowed: it will also be unusable in ComfyUI's own editor.
        from . import log
        log.broke(class_type, exc, "describing its inputs")
        return None

    widgets: List[dict] = []
    sockets: List[dict] = []
    for section in ("required", "optional"):
        for name, declared in (spec.get(section) or {}).items():
            described = _one(name, declared, section == "required")
            # forceInput turns a primitive into a socket: the node author is
            # saying this one is wired, not typed, and offering a box for it
            # would invite a value the graph builder then refuses.
            editable = (described["type"] in PRIMITIVE
                        and not described.get("forceInput"))
            (widgets if editable else sockets).append(described)

    return {
        "node": class_type,
        "title": _display_name(class_type, node),
        "description": getattr(node, "DESCRIPTION", "") or "",
        "category": getattr(node, "CATEGORY", "") or "",
        "widgets": widgets,
        "sockets": sockets,
        "outputs": list(getattr(node, "RETURN_TYPES", ()) or ()),
    }


def _display_name(class_type: str, node: Any) -> str:
    try:
        import nodes as comfy_nodes
        name = comfy_nodes.NODE_DISPLAY_NAME_MAPPINGS.get(class_type)
        if name:
            return name
    except Exception:                            # noqa: BLE001
        pass
    return getattr(node, "DISPLAY_NAME", None) or class_type


def describe_all(class_types) -> Dict[str, Optional[dict]]:
    """Several at once, because a pipeline is edited as a whole.

    Absent nodes are present in the answer with a null, so the caller can tell
    "not installed" from "not asked about" -- one reads as a slot to fix and the
    other as a bug in the request.
    """
    return {name: describe(name) for name in dict.fromkeys(class_types)}
