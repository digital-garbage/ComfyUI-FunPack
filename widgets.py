"""Declarative list inputs: one STRING widget holding a JSON array of rows.

ComfyUI has no repeatable input, so a node that needs N of something (text encoders,
LoRAs) carries ONE string widget whose value is a JSON array. The `funpack_list` entry
in that widget's options describes a row, so both frontends can render it as rows with
an Add button instead of raw JSON: the canvas widget in `web/funpack_list.js`, and the
FunPack Editor's node page (`movie_editor/frontend/models.js`).

The node itself never has to care which frontend wrote the value — `parse_rows` accepts
the JSON string, an already-parsed list, or a single row.
"""
import json

# Row field kinds a frontend must know how to draw.
FIELD_KINDS = {"combo", "string", "int", "float", "boolean"}


def field(name, kind, *, label=None, choices=None, default=None, min=None, max=None,
          step=None, tooltip=None, width=None):
    """One column of a list row."""
    spec = {"name": name, "kind": kind, "label": label or name}
    if choices is not None:
        spec["choices"] = list(choices)
    if default is not None:
        spec["default"] = default
    for key, value in (("min", min), ("max", max), ("step", step),
                       ("tooltip", tooltip), ("width", width)):
        if value is not None:
            spec[key] = value
    return spec


def list_widget(item, fields, *, default=None, add_label=None, max_rows=0, tooltip=None,
                allow_empty=False, picker=None):
    """An INPUT_TYPES entry for a list input. Returns the ("STRING", {...}) tuple.

    `allow_empty` marks a list whose empty state is a working state — the node passes its
    input through — so a frontend does not flag it as something to fix.
    """
    spec = {"item": item, "fields": list(fields), "add_label": add_label or f"+ Add {item}"}
    if max_rows:
        spec["max_rows"] = int(max_rows)
    if allow_empty:
        spec["allow_empty"] = True
    if picker:
        # Rows are PICKED from this field's file list rather than typed into a table: the
        # frontend shows a searchable list of what is installed, and the row is that file
        # plus its settings. A table of combos is a spreadsheet; this is a picker.
        spec["picker"] = picker
    opts = {
        "default": json.dumps(default or []),
        "multiline": False,
        "funpack_list": spec,
    }
    if tooltip:
        opts["tooltip"] = tooltip
    return ("STRING", opts)


def _coerce(value, kind, default=None):
    if kind == "boolean":
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value) if value is not None else bool(default)
    if kind in ("int", "float"):
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float(default or 0)
        if number != number or number in (float("inf"), float("-inf")):
            number = float(default or 0)
        return int(number) if kind == "int" else number
    if value is None:
        return default
    return value


def parse_rows(value, fields, *, key=None, drop_empty=True):
    """Normalize a list widget's value into a list of row dicts.

    `key` names the field a row is worthless without (the file name, usually); rows whose
    key is missing, empty or "None" are dropped unless `drop_empty` is off. A row may also
    carry "on": False to be skipped without deleting it.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []

    specs = {f["name"]: f for f in fields}
    if key is None and fields:
        key = fields[0]["name"]

    rows = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            continue
        if not _coerce(raw.get("on", True), "boolean", True):
            continue
        row = {"index": index}
        for name, spec in specs.items():
            row[name] = _coerce(raw.get(name), spec["kind"], spec.get("default"))
        if drop_empty and key:
            k = row.get(key)
            if k is None or (isinstance(k, str) and k.strip() in ("", "None")):
                continue
        rows.append(row)
    return rows
