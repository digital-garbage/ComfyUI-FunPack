"""Validating an announcement.

Everything here refuses rather than repairs. A module whose declaration is wrong
does not get a best-effort panel -- it does not load at all, and says why once.
That is the same rule the UI follows: absent, not approximate.
"""

import math
from typing import Any, Dict, List, Tuple

from .contract import NUMERIC, STAGES, TYPES, UI_HINTS, ModuleSpec


class SchemaError(ValueError):
    """A declaration core will not accept."""


SCALARS = (str, int, float, bool)

# Content the UI renders as text. Anything non-scalar here would have to be
# interpreted to be displayed, and interpreting module-supplied content is the
# one thing the kit will not do.
TEXT_KEYS = ("label", "hint", "unit", "placeholder")


def _require_text(value: Any, where: str) -> None:
    if not isinstance(value, str):
        raise SchemaError(f"{where} must be a string; got {type(value).__name__}.")


def validate_setting(key: str, spec: Any, siblings: Dict[str, dict]) -> dict:
    where = f"setting {key!r}"
    if not isinstance(spec, dict):
        raise SchemaError(f"{where} must be a dict; got {type(spec).__name__}.")

    kind = spec.get("type")
    if kind not in TYPES:
        raise SchemaError(f"{where} has unknown type {kind!r}. Known: {', '.join(sorted(TYPES))}.")

    if "default" not in spec:
        # Without one there is no value for a headless run, and the panel would
        # have to invent a starting point that apply() never agreed to.
        raise SchemaError(f"{where} has no default.")

    for text_key in TEXT_KEYS:
        if text_key in spec:
            _require_text(spec[text_key], f"{where}'s {text_key}")

    hint = spec.get("ui")
    if hint is not None and hint not in UI_HINTS[kind]:
        raise SchemaError(
            f"{where} asks for ui {hint!r}, which is not available for a {kind}. "
            f"Allowed: {', '.join(UI_HINTS[kind])}."
        )

    if kind == "enum":
        _validate_enum(spec, where)
    elif kind in NUMERIC:
        _validate_numeric(spec, where, kind)
    elif kind == "bool":
        if not isinstance(spec["default"], bool):
            raise SchemaError(f"{where} is a bool but its default is {spec['default']!r}.")
    else:
        if not isinstance(spec["default"], str):
            raise SchemaError(f"{where} is a {kind} but its default is not a string.")

    if "when" in spec:
        _validate_when(spec["when"], where, siblings, key)

    return spec


def _validate_enum(spec: dict, where: str) -> None:
    options = spec.get("options")
    if not isinstance(options, (list, tuple)) or not options:
        raise SchemaError(f"{where} is an enum with no options.")

    values = []
    for i, option in enumerate(options):
        if not isinstance(option, dict):
            raise SchemaError(f"{where} option {i} must be a dict with value and label.")
        if "value" not in option or "label" not in option:
            raise SchemaError(f"{where} option {i} needs both a value and a label.")
        if not isinstance(option["value"], SCALARS):
            raise SchemaError(f"{where} option {i}'s value must be a scalar.")
        _require_text(option["label"], f"{where} option {i}'s label")
        values.append(option["value"])

    if len(set(values)) != len(values):
        raise SchemaError(f"{where} has duplicate option values.")
    if spec["default"] not in values:
        raise SchemaError(f"{where}'s default {spec['default']!r} is not one of its options.")


def _validate_numeric(spec: dict, where: str, kind: str) -> None:
    default = spec["default"]
    if isinstance(default, bool) or not isinstance(default, (int, float)):
        raise SchemaError(f"{where} is a {kind} but its default is {default!r}.")
    if kind == "int" and not isinstance(default, int):
        raise SchemaError(f"{where} is an int but its default is a float.")

    low, high = spec.get("min"), spec.get("max")
    for name, bound in (("min", low), ("max", high)):
        if bound is not None and (isinstance(bound, bool) or not isinstance(bound, (int, float))):
            raise SchemaError(f"{where}'s {name} must be a number.")
    if low is not None and high is not None and low > high:
        raise SchemaError(f"{where} has min {low} above max {high}.")
    if low is not None and default < low:
        raise SchemaError(f"{where}'s default {default} is below its min {low}.")
    if high is not None and default > high:
        raise SchemaError(f"{where}'s default {default} is above its max {high}.")

    step = spec.get("step")
    if step is not None and (isinstance(step, bool) or not isinstance(step, (int, float)) or step <= 0):
        raise SchemaError(f"{where}'s step must be a positive number.")


def _validate_when(when: Any, where: str, siblings: Dict[str, dict], key: str) -> None:
    if not isinstance(when, dict) or not when:
        raise SchemaError(f"{where}'s when must be a non-empty dict of sibling conditions.")
    for other, expected in when.items():
        if other == key:
            raise SchemaError(f"{where}'s when refers to itself.")
        if other not in siblings:
            # A condition on a key that does not exist is always false, so the
            # row would simply never appear -- a typo that reads as a decision.
            raise SchemaError(f"{where}'s when refers to {other!r}, which this module does not declare.")
        if not isinstance(expected, SCALARS) and not isinstance(expected, (list, tuple)):
            raise SchemaError(f"{where}'s when value for {other!r} must be a scalar or a list.")


def validate_nodes(value: Any, module_id: str) -> list:
    """The node classes a module contributes.

    Checked structurally, not by isinstance against `io.ComfyNode`, so core stays
    importable without ComfyUI -- the same property that lets the whole suite run
    outside it. The ComfyUI-specific checks (a FunPack-prefixed node_id, and no
    two modules claiming one id) happen in `core.nodes`, where ComfyUI exists.
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise SchemaError(
            f"module {module_id!r}'s nodes must be a list; got {value!r}."
        )

    for node in value:
        if not isinstance(node, type):
            raise SchemaError(
                f"module {module_id!r} lists {node!r} as a node, which is not a class."
            )
        for method in ("define_schema", "execute"):
            if not callable(getattr(node, method, None)):
                raise SchemaError(
                    f"module {module_id!r}'s node {node.__name__} has no {method}(). "
                    f"A FunPack node is a comfy_api io.ComfyNode subclass."
                )
    return list(value)


def validate_traits(value: Any, module_id: str):
    """A module's trait provider: callable(model) -> iterable of trait names."""
    if value is None:
        return None
    if not callable(value):
        raise SchemaError(
            f"module {module_id!r}'s traits must be a callable(model), got "
            f"{type(value).__name__}."
        )
    return value


def validate_provides(value: Any, module_id: str) -> dict:
    """Named capabilities: {name: callable}."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SchemaError(
            f"module {module_id!r}'s provides must be a dict of name -> callable; "
            f"got {type(value).__name__}.")
    for name, fn in value.items():
        if not isinstance(name, str) or not name.strip():
            raise SchemaError(f"module {module_id!r} provides a capability with no name.")
        if not callable(fn):
            raise SchemaError(
                f"module {module_id!r}'s {name!r} capability is not callable.")
    return dict(value)


def check_values(spec, values: Any) -> Tuple[Dict[str, Any], List[str]]:
    """One module's live values, checked against its own declaration.

    Returns (values, problems). Anything missing falls back to the declared
    default, so a partial payload is fine; anything WRONG is reported rather than
    coerced. v4 carried these as JSON in a string widget that nothing validated,
    which is why it needed a hand-maintained mirror of every default and a list of
    keys that no longer meant anything.
    """
    problems: List[str] = []
    if values is None:
        return spec.defaults(), problems
    if not isinstance(values, dict):
        return spec.defaults(), [f"{spec.id}: settings must be an object, got {type(values).__name__}."]

    clean = spec.defaults()
    for key, value in values.items():
        declared = spec.settings.get(key)
        if declared is None:
            # Not repaired by dropping it silently: a key nobody declares is
            # either a typo or a leftover, and both are worth saying once.
            problems.append(f"{spec.id}: no setting named {key!r}.")
            continue
        problem = _value_problem(declared, key, value, spec.id)
        if problem:
            problems.append(problem)
            continue
        clean[key] = value
    return clean, problems


def _value_problem(declared: dict, key: str, value: Any, module_id: str):
    kind = declared["type"]
    where = f"{module_id}.{key}"

    if kind == "bool":
        if not isinstance(value, bool):
            return f"{where} is a bool, got {value!r}."
        return None

    if kind == "enum":
        allowed = [option["value"] for option in declared["options"]]
        if value not in allowed:
            return f"{where} must be one of {allowed}, got {value!r}."
        return None

    if kind in NUMERIC:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{where} is a {kind}, got {value!r}."
        # NaN and the infinities compare False against BOTH bounds, so a range
        # check alone lets them through -- and json.loads accepts all three by
        # default. A NaN strength reaches sampling and poisons it silently.
        if not math.isfinite(value):
            return f"{where} must be a finite number, got {value!r}."
        if kind == "int" and not isinstance(value, int):
            return f"{where} is an int, got {value!r}."
        low, high = declared.get("min"), declared.get("max")
        if low is not None and value < low:
            return f"{where} is below its minimum {low}: {value}."
        if high is not None and value > high:
            return f"{where} is above its maximum {high}: {value}."
        return None

    if not isinstance(value, str):
        return f"{where} is a {kind}, got {value!r}."
    return None


def validate(announcement: Dict[str, Any], source: str = "") -> ModuleSpec:
    """A validated ModuleSpec, or SchemaError naming what is wrong."""
    for required in ("id", "title"):
        value = announcement.get(required)
        if not isinstance(value, str) or not value.strip():
            raise SchemaError(f"module {source or '?'} has no {required}.")

    # A mount point is only meaningful for a module with something to render.
    # A loader or a node-only module has no panel, and forcing it to name a
    # region would make it invent one -- which then shows up as "hidden", i.e.
    # a module reported as broken for having nothing to show.
    mount = announcement.get("mount")
    renders = bool(announcement.get("settings")) or bool(announcement.get("ui"))
    if renders and (not isinstance(mount, str) or not mount.strip()):
        raise SchemaError(
            f"module {announcement.get('id') or source or '?'} has settings to render "
            f"but no mount."
        )
    if mount is not None and not isinstance(mount, str):
        raise SchemaError(f"module {announcement.get('id')!r}'s mount must be a string.")

    stage = announcement.get("stage", "sampling")
    if stage not in STAGES:
        raise SchemaError(f"module {announcement['id']!r} has unknown stage {stage!r}. Known: {', '.join(STAGES)}.")

    status = announcement.get("status", "experimental")
    if status not in ("proven", "experimental"):
        raise SchemaError(f"module {announcement['id']!r} has unknown status {status!r}.")

    raw = announcement.get("settings") or {}
    if not isinstance(raw, dict):
        raise SchemaError(f"module {announcement['id']!r} has a non-dict settings block.")
    settings = {key: validate_setting(key, spec, raw) for key, spec in raw.items()}

    def _ids(name: str) -> list:
        value = announcement.get(name)
        if value is None:
            return []
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise SchemaError(f"module {announcement['id']!r}'s {name} must be a list.")
        for item in value:
            if not isinstance(item, str):
                raise SchemaError(f"module {announcement['id']!r}'s {name} must contain strings.")
        return list(value)

    return ModuleSpec(
        nodes=validate_nodes(announcement.get("nodes"), announcement["id"]),
        traits=validate_traits(announcement.get("traits"), announcement["id"]),
        provides=validate_provides(announcement.get("provides"), announcement["id"]),
        id=announcement["id"],
        title=announcement["title"],
        mount=announcement.get("mount") or "",
        settings=settings,
        requires=_ids("requires"),
        after=_ids("after"),
        before=_ids("before"),
        stage=stage,
        ui=announcement.get("ui"),
        status=status,
        source=source,
    )
