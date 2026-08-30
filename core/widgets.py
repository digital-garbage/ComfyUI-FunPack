"""What a node's inputs look like to a person editing them.

`graph.py` reads a node's inputs to decide whether a wire is legal: it needs the
TYPE and nothing else. Editing a value needs the rest -- the choices a combo
offers, the bounds on a number, the default, whether the box is multiline -- and
that is a different question about the same declaration, so it is asked here
rather than widening the one graph.py asks. What both of them need to know about
the SPELLING of a type is in `comfy_types`, so they cannot disagree about it.

Read from ComfyUI's own INPUT_TYPES, the same source its own frontend uses, so a
node written for ComfyUI is editable here without knowing anything about
FunPack. Nothing in this file names a node.

A socket is not a widget. An input whose type is a socket type (MODEL, LATENT,
IMAGE...) is filled by a wire, and offering a text box for it would invite a
value the graph builder then refuses -- so those are reported, and reported as
sockets, rather than made editable.
"""

from typing import Any, Dict, List, Optional

from . import comfy_types

COMBO = comfy_types.COMBO
PRIMITIVE = comfy_types.PRIMITIVE

# What a widget's options dict may say that is worth carrying to a form. Named
# rather than passed through wholesale: an options dict can hold anything a node
# author put there, and forwarding all of it makes the payload a node's private
# business rather than a contract.
KEPT = ("default", "min", "max", "step", "round", "multiline", "dynamicPrompts",
        "tooltip", "placeholder", "control_after_generate", "image_upload",
        "label_on", "label_off", "precision", "forceInput", "multiselect")


def _one(name: str, declared: Any, required: bool) -> Dict[str, Any]:
    kind, options = comfy_types.declared(declared)

    widget: Dict[str, Any] = {"name": name, "required": required}
    editable = comfy_types.widget_type(kind, options)
    # The type as it is EDITED, not as it is declared. A MultiType wrapped
    # around a number arrives as "FLOAT,INT" and is a box you type into; read
    # literally it becomes a socket demanding a source for a field nobody wires.
    widget["type"] = editable if editable else (kind if isinstance(kind, str) else COMBO)

    if widget["type"] == COMBO:
        # A combo may be empty -- a file picker on a machine with no files --
        # and that is worth saying rather than rendering a select with nothing
        # in it and no explanation.
        widget["choices"] = (list(kind) if isinstance(kind, (list, tuple))
                             else comfy_types.choices(options))
        if comfy_types.reveals(options):
            # Each choice carries its own further inputs. Nothing renders those
            # yet, and a form that dropped them silently would offer an
            # incomplete node as a complete one.
            widget["reveals_more"] = True

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


def search(query: str, limit: int = 40) -> Dict[str, Any]:
    """Installed nodes whose name, title or category matches, for a picker.

    Names only -- not their inputs. Describing every installed node is what
    ComfyUI's own /object_info does, and on a machine with a few packs that is
    megabytes to answer "what could go in this slot"; the one the user picks is
    then described on its own.

    `total` is the count BEFORE the cut, so a picker can say the list is not all
    of it rather than implying forty is everything there is.

    Ranked so an exact name beats a name that merely contains the query: typing
    "KSampler" and getting KSamplerAdvanced first is the picker being unhelpful
    about the thing it was told.
    """
    try:
        import nodes as comfy_nodes
    except Exception:                            # noqa: BLE001 -- not inside ComfyUI
        return {"nodes": [], "total": 0}

    needle = (query or "").strip().lower()
    found = []
    for class_type, node in comfy_nodes.NODE_CLASS_MAPPINGS.items():
        title = _display_name(class_type, node)
        category = getattr(node, "CATEGORY", "") or ""
        if needle and needle not in f"{class_type}\n{title}\n{category}".lower():
            continue
        lowered = class_type.lower()
        rank = (0 if lowered == needle
                else 1 if lowered.startswith(needle)
                else 2 if needle in lowered
                else 3)
        found.append((rank, lowered, {
            "node": class_type,
            "title": title,
            "category": category,
            "outputs": list(getattr(node, "RETURN_TYPES", ()) or ()),
        }))

    found.sort(key=lambda entry: entry[:2])
    return {"nodes": [entry[2] for entry in found[:max(0, limit)]],
            "total": len(found)}
