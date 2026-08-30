"""How ComfyUI spells a type, in one place.

Two files here ask questions about the same declaration -- `graph.py` asks
whether a wire is legal, `widgets.py` asks whether a person can type into it --
and both had their own idea of what a type string means. That is the drift this
codebase keeps warning about, and it had already happened: a combo was
"a list where the type goes" in one and "the string COMBO" in the other, and
between them they missed the three shapes below.

The shapes, all current, all present in a stock install:

* **A combo** is a dropdown. Spelled as a list in the old schema
  (`(["a", "b"], {...})`), as the string `"COMBO"` with the choices under
  `options` in V3, and as `"COMFY_DYNAMICCOMBO_V3"` -- 140 inputs in a stock
  install -- where each option may itself be a dict carrying further inputs.
  Anything with COMBO in its type name is a widget, never a wire.
* **A union** is a comma-joined type string: `"IMAGE,MASK"` takes either. An
  exact string comparison refuses a MASK feeding it, which is a legal wire
  refused -- as bad a failure as an illegal one accepted.
* **A MultiType wrapped around a widget** carries `widgetType` in its options
  and lists the widget's own type first: `("FLOAT,INT", {"widgetType": "FLOAT"})`
  is a number you type, not a socket. Read as a socket it demands a source for a
  field nobody needs to wire.

Nothing here names a node.
"""

from typing import Any, Dict, List, Optional, Tuple

COMBO = "COMBO"

# What a person types INTO. Everything else arrives on a wire.
PRIMITIVE = frozenset({"STRING", "INT", "FLOAT", "BOOLEAN", COMBO})


def declared(spec_entry: Any) -> Tuple[Any, Dict[str, Any]]:
    """(type, options) out of whichever shape a declaration arrived in."""
    if isinstance(spec_entry, (list, tuple)) and spec_entry:
        kind = spec_entry[0]
        options = spec_entry[1] if len(spec_entry) > 1 and isinstance(spec_entry[1], dict) else {}
        return kind, options
    return spec_entry, {}


def is_combo(kind: Any) -> bool:
    """Any dropdown: a list of choices, "COMBO", or a V3 dynamic combo."""
    if isinstance(kind, (list, tuple)):
        return True
    return isinstance(kind, str) and COMBO in kind.upper()


def members(kind: Any) -> List[str]:
    """The member types of a type string. A union names several."""
    if not isinstance(kind, str):
        return []
    return [part.strip() for part in kind.split(",") if part.strip()]


def widget_type(kind: Any, options: Optional[dict] = None) -> Optional[str]:
    """The type a person edits this as, or None if it is a socket."""
    options = options or {}
    if is_combo(kind):
        return COMBO
    # The node author saying so outright. A MultiType wrapped around a widget
    # input is a widget in ComfyUI's own frontend, and this is how it says so.
    declared_widget = options.get("widgetType")
    if isinstance(declared_widget, str):
        if is_combo(declared_widget):
            return COMBO
        if declared_widget in PRIMITIVE:
            return declared_widget
    parts = members(kind)
    if parts and all(part in PRIMITIVE for part in parts):
        return parts[0]
    return None


def is_widget(kind: Any, options: Optional[dict] = None) -> bool:
    return widget_type(kind, options) is not None


def choices(options: Optional[dict]) -> List[Any]:
    """A combo's choices, whichever way they were written.

    A dynamic combo's option is a dict whose `key` may be an Enum member, and
    `str()` on one of those is "ResizeType.SCALE_DIMENSIONS" -- a value ComfyUI
    would then refuse, because what it wants is the enum's value.
    """
    options = options or {}
    raw = options.get("options")
    if raw is None:
        raw = options.get("choices")
    found = []
    for option in raw or []:
        if isinstance(option, dict):
            for key in ("key", "value", "content", "name", "label"):
                if key in option:
                    found.append(_plain(option[key]))
                    break
        else:
            found.append(_plain(option))
    return found


def reveals(options: Optional[dict]) -> bool:
    """Whether picking a choice brings inputs with it.

    A dynamic combo's options can each carry their own `inputs`, so the form
    changes with the choice. Nothing here renders those yet, and a window that
    dropped them silently would be offering an incomplete node as a complete
    one.
    """
    options = options or {}
    raw = options.get("options") or []
    return any(isinstance(option, dict) and option.get("inputs") for option in raw)


def accepts(wanted: Any, given: Any) -> bool:
    """Whether an output of type `given` may feed an input of type `wanted`."""
    if wanted == "*" or given == "*":
        return True
    left, right = set(members(wanted)), set(members(given))
    if not left or not right:
        return wanted == given
    return bool(left & right)


def _plain(value: Any) -> Any:
    """An Enum member as the value ComfyUI wants, anything else untouched."""
    inner = getattr(value, "value", None)
    if inner is not None and not isinstance(value, (str, int, float, bool)):
        return inner
    return value
