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
* **A MatchType** (V3's `io.MatchType`) is one socket standing in for several
  interchangeable types -- ImageTransformKJ's `image` input takes an IMAGE or a
  MASK through the one plug. Its own type name, `COMFY_MATCHTYPE_V3`, says
  nothing about which: the real types live in a `template` alongside it (a
  string `"IMAGE,MASK"` on an input's own `options`, a list of io type classes
  on an output's V3 schema entry -- two different shapes for the same idea,
  because one comes from the V1-compat `INPUT_TYPES()` dict and the other from
  the V3 schema object directly). Read literally, the bare type name is neither
  a union nor a widget nor anything `accepts()` can compare -- it would refuse
  every wire into or out of a MatchType socket, including a completely legal
  one.
* **An Autogrow** (V3's `io.Autogrow`) is not one input, it is a TEMPLATE for
  as many numbered ones as get used -- MiniMaxH3ReferenceToVideo's
  `ref_images` is really `ref_image_0`, `ref_image_1`, ... up to its own `max`,
  each an ordinary IMAGE socket that happens to not exist until something
  wires it. `INPUT_TYPES()` reports the group as one entry of type
  `COMFY_AUTOGROW_V3`, carrying the template (the WRAPPED input's own
  declaration, a `prefix`, and `min`/`max`) rather than a real type at all --
  there is no socket named literally "ref_images" to wire into. Left alone,
  every numbered instance is unreachable: not present in `declared` at all, so
  a wire naming one is refused as an input the node does not have, for a node
  that plainly does.

Nothing here names a node.
"""

from typing import Any, Dict, List, Optional, Tuple

COMBO = "COMBO"
MATCH_TYPE = "COMFY_MATCHTYPE_V3"
AUTOGROW = "COMFY_AUTOGROW_V3"

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


def autogrow_instances(options: Optional[dict]) -> List[Tuple[str, Any, Dict[str, Any]]]:
    """(dotted_name, kind, options) for every numbered instance an Autogrow's
    `template` allows -- "ref_images" carrying `{"template": {"input":
    {"required": {"ref_image": ("IMAGE", {...})}}, "prefix": "ref_image_",
    "min": 0, "max": 9}}` becomes ("ref_image_0", "IMAGE", {...}) through
    ("ref_image_9", "IMAGE", {...}).

    Every instance is wireable regardless of `min`: an autogrow's own `min` is
    how many the ORIGINAL node would insist on before it runs, which is not
    this codebase's concern here -- core's `required` list is built from
    `INPUT_TYPES()`'s "required"/"optional" split at the GROUP level, and this
    only widens what a slot's `inputs` dict is allowed to name, never what it
    must.

    The wrapped input can itself be any single-value declaration this module
    understands (a widget, a union, ...); it is run back through `declared()`
    the same as any other input would be. A malformed or unrecognised template
    (a future Autogrow shape this codebase has not seen) yields no instances
    rather than a guess -- the group stays unreachable exactly as it was
    before, not reachable under a wrong assumption about its shape.
    """
    template = (options or {}).get("template")
    if not isinstance(template, dict):
        return []
    wrapped = (template.get("input") or {}).get("required") or {}
    if len(wrapped) != 1:
        return []
    (inner_name, inner_decl), = wrapped.items()
    prefix = template.get("prefix")
    lo, hi = template.get("min"), template.get("max")
    if (not isinstance(prefix, str) or not isinstance(lo, int) or not isinstance(hi, int)
            or isinstance(lo, bool) or isinstance(hi, bool) or lo > hi):
        return []
    kind, inner_options = declared(inner_decl)
    return [(f"{prefix}{i}", kind, inner_options) for i in range(lo, hi + 1)]


def match_type_union(allowed: Any) -> Optional[str]:
    """A MatchType's real wire type, as a union string `accepts()` understands.

    `allowed` arrives as either shape described above: a ready-made string off
    an input's `options["template"]["allowed_types"]`, or a list of io type
    classes off an output's V3 schema `template.allowed_types` -- each with its
    own type name on `.io_type` (`io.Image.io_type == "IMAGE"`). Anything else
    (a template this codebase has not seen, a future SDK change) resolves to
    None rather than a guess, which callers turn into "no comparison possible"
    the same way an unrecognised type already does.
    """
    if isinstance(allowed, str):
        return allowed or None
    if isinstance(allowed, (list, tuple)):
        names = [getattr(t, "io_type", None) for t in allowed]
        names = [n for n in names if isinstance(n, str) and n]
        return ",".join(names) if names else None
    return None


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
