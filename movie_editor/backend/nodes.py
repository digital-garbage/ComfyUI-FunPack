"""Pluggable node slots for the fixed FunPack path.

The generation graph is fixed; only certain slots vary per machine because they
depend on installed models / chosen nodes. The user picks a ComfyUI node for each
slot and the editor exposes that node's widget inputs (rendered from /object_info).

This module: defines the slot roles, filters /object_info to candidate nodes per
role, extracts the user-facing widget inputs, and persists the chosen config.
"""
from __future__ import annotations

import json

from . import config

# Types that are graph CONNECTIONS (wired automatically), not user widgets.
LINK_TYPES = {
    "MODEL", "CLIP", "VAE", "CLIP_VISION", "CONDITIONING", "LATENT", "IMAGE",
    "MASK", "CONTROL_NET", "STYLE_MODEL", "GLIGEN", "AUDIO", "SAMPLER", "SIGMAS",
    "GUIDER", "NOISE", "UPSCALE_MODEL", "PHOTOMAKER", "WEBCAM",
}
WIDGET_PRIMITIVES = {"INT", "FLOAT", "STRING", "BOOLEAN"}
# "COMBO" is the V1/new-style dropdown widget type; it is NOT a graph connection even
# though it is a string.  Old-style combos (spec[0] is a list) are already excluded by
# the isinstance(t, str) guard, but new-style ones need an explicit check.
_WIDGET_TYPES = WIDGET_PRIMITIVES | {"COMBO"}


def _is_combo_type(t) -> bool:
    """Any dropdown widget: 'COMBO' or a V3 dynamic combo (e.g. COMFY_DYNAMICCOMBO_V3).
    These are widgets, never graph connections, even though they're string-typed."""
    return isinstance(t, str) and "COMBO" in t.upper()


def _is_widget_type(t) -> bool:
    return isinstance(t, str) and (t in _WIDGET_TYPES or _is_combo_type(t))


def _combo_choices(opts: dict) -> list:
    """Normalise combo options to plain string/int choices. V3 dynamic combos express
    options as dicts {"key": <value>, "inputs": [...]}; standard combos are list[str]."""
    raw = opts.get("options")
    if raw is None:
        raw = opts.get("choices")
    out = []
    for o in (raw or []):
        if isinstance(o, dict):
            out.append(str(o.get("key", o.get("value", o.get("content", o.get("name", o.get("label", "")))))))
        else:
            out.append(o)
    return out


def _combo_default(opts: dict, choices: list | None = None):
    if choices is None:
        choices = _combo_choices(opts)
    d = opts.get("default")
    if isinstance(d, dict):
        d = d.get("key", d.get("value"))
    if d is not None and d != "":
        return d
    return choices[0] if choices else ""

# ComfyUI V3 dynamic match types that are semantically IMAGE-compatible.
_MATCHTYPE_ALIASES: dict[str, str] = {}  # populated on first lookup — patterns are prefix-matched


def _normalize_type(t: str) -> str:
    """Normalize ComfyUI internal type aliases to canonical names.
    COMFY_MATCHTYPE_V* are dynamic IMAGE-compatible types introduced in ComfyUI V3."""
    if t.startswith("COMFY_MATCHTYPE_"):
        return "IMAGE"
    return t


# role -> {label, category, want_output(s), [want_input]}.
# A node qualifies for a role if it OUTPUTS the wanted type (and, for patchers like
# LoRA / image processors, also INPUTS the relevant type).
ROLES: dict[str, dict] = {
    "unet":          {"label": "Unet / Diffusion Model", "category": "Loaders",  "output": "MODEL"},
    "lora":          {"label": "LoRA",                    "category": "Loaders",  "output": "MODEL", "input": "MODEL"},
    "video_vae":     {"label": "Video VAE",               "category": "Loaders",  "output": "VAE"},
    "audio_vae":     {"label": "Audio VAE",               "category": "Loaders",  "output": "VAE"},
    "clip":          {"label": "CLIP / Text Encoder",     "category": "Loaders",  "output": "CLIP"},
    "clip_vision":   {"label": "CLIP Vision",             "category": "Loaders",  "output": "CLIP_VISION"},
    "image_processing": {"label": "Input Image Processing", "category": "Pipeline", "output": "IMAGE", "input": "IMAGE"},
    "empty_latent":  {"label": "Empty Latent Generator",  "category": "Pipeline", "output": "LATENT"},
    "audio_encoder": {"label": "Audio Encoder",           "category": "Pipeline", "output": "LATENT"},
}

# Pipeline inputs that must be satisfied by configured slot nodes.
# Each entry: id, type, label, required, role_hint (role key to suggest adding), hint.
PIPELINE_REQUIREMENTS = [
    {"id": "model",        "type": "MODEL",  "label": "Diffusion model",  "required": True,
     "role_hint": "unet",       "hint": "Add a Unet / Diffusion Model loader (e.g. LTXVLoader)."},
    {"id": "clip",         "type": "CLIP",   "label": "Text encoder",     "required": True,
     "role_hint": "clip",       "hint": "Add a CLIP / Text Encoder loader (usually bundled with the unet loader)."},
    {"id": "video_vae",    "type": "VAE",    "label": "Video VAE",        "required": True,
     "role_hint": "video_vae",  "hint": "Add a Video VAE loader (e.g. LTXVVideoDecoder or the VAE output from LTXVLoader)."},
    {"id": "audio_vae",    "type": "VAE",    "label": "Audio VAE",        "required": True,
     "role_hint": "audio_vae",  "hint": "Add a separate Audio VAE loader (e.g. LTXVAudioDecoder). This is distinct from the video VAE."},
    {"id": "audio_latent", "type": "LATENT", "label": "Audio latent",     "required": True,
     "role_hint": "audio_encoder", "hint": "Add an Audio Encoder node that converts an audio file into a latent (e.g. LTXVAudioEncoder)."},
    {"id": "source_image", "type": "IMAGE",  "label": "Source image",     "required": False,
     "role_hint": None,         "hint": "Optional — attach an image to a scene for image-to-video generation."},
    {"id": "init_latent",  "type": "LATENT", "label": "Initial video latent", "required": False,
     "role_hint": "empty_latent", "hint": "Optional — override the starting latent. Usually left unset (Studio generates it)."},
]


def _outputs(node_def: dict) -> list[str]:
    out = node_def.get("output") or []
    return [_normalize_type(o) for o in out if isinstance(o, str)]


def _all_input_types(node_def: dict) -> list[str]:
    types = []
    inp = node_def.get("input") or {}
    for group in ("required", "optional"):
        for spec in (inp.get(group) or {}).values():
            t = spec[0] if isinstance(spec, list) and spec else None
            if isinstance(t, str) and t not in _WIDGET_TYPES:
                types.append(_normalize_type(t))
    return types


def _matches_role(node_def: dict, role: dict) -> bool:
    outs = _outputs(node_def)
    ins = _all_input_types(node_def)
    if role.get("output"):
        if role["output"] not in outs:
            return False
        # Pure-source roles (no declared input) must NOT consume their own output type —
        # that signals a patcher/transformer (LoRA, sampler), not a loader/generator.
        if not role.get("input") and role["output"] in ins:
            return False
    if role.get("input") and role["input"] not in ins:
        return False
    return True


def connection_inputs(node_def: dict) -> list[dict]:
    """Typed connection inputs a node accepts (MODEL/CLIP/VAE/LATENT/IMAGE/...),
    i.e. the sockets other nodes can wire INTO. Skips widgets (combo/INT/FLOAT/...)."""
    out = []
    inp = node_def.get("input") or {}
    for group in ("required", "optional"):
        for name, spec in (inp.get(group) or {}).items():
            if not isinstance(spec, list) or not spec:
                continue
            t = spec[0]
            if not isinstance(t, str):
                continue
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if _is_widget_type(t):
                # A widget-typed input is a SOCKET (wireable) only when forceInput is set —
                # e.g. refinement_key_input is ("STRING", {"forceInput": True}). It always
                # has a widget fallback, so it's NEVER auto-required (it must not block
                # generation when left unwired). Plain widgets (incl. V3 dynamic combos
                # like COMFY_DYNAMICCOMBO_V3) are handled by widget_inputs, not here.
                if not opts.get("forceInput"):
                    continue
                out.append({"name": name, "type": _normalize_type(t), "required": False})
                continue
            # V3 nodes can mark a required-group input as optional via the flag.
            is_required = group == "required" and not opts.get("optional", False)
            out.append({"name": name, "type": _normalize_type(t), "required": is_required})
    return out


def node_outputs(node_def: dict) -> list[dict]:
    """Outputs a node produces, as {name, type}, so they can be wired onward."""
    names = node_def.get("output_name")
    if not isinstance(names, list):
        names = []
    out = []
    outputs = node_def.get("output")
    if not isinstance(outputs, list):
        return out
    for i, t in enumerate(outputs):
        if not isinstance(t, str):
            continue
        nm = names[i] if i < len(names) and names[i] else t
        out.append({"name": nm, "type": _normalize_type(t)})
    return out


def widget_inputs(node_def: dict) -> list[dict]:
    """User-facing widgets for a node: combos (with options) and primitive fields.
    Skips graph-connection inputs (MODEL/CLIP/IMAGE/...) and forceInput sockets."""
    out = []
    inp = node_def.get("input") or {}
    for group in ("required", "optional"):
        for name, spec in (inp.get(group) or {}).items():
            if not isinstance(spec, list) or not spec:
                continue
            t = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            # forceInput means the field is always a socket, never a user widget.
            if opts.get("forceInput"):
                continue
            field = {"name": name, "required": group == "required", "options": opts}
            if isinstance(t, list):
                # Old-style combo: t is the list of choices (e.g. folder_paths filenames).
                field["kind"] = "combo"
                field["choices"] = t
                field["default"] = opts.get("default", t[0] if t else None)
            elif _is_combo_type(t):
                # V1 "COMBO" or a V3 dynamic combo (COMFY_DYNAMICCOMBO_V3). Options may be
                # plain strings or {"key": ...} dicts — normalise to the selectable value.
                choices = _combo_choices(opts)
                field["kind"] = "combo"
                field["choices"] = choices
                field["default"] = _combo_default(opts, choices)
            elif t in WIDGET_PRIMITIVES:
                field["kind"] = t.lower()
                field["default"] = opts.get("default")
            else:
                continue  # connection input — wired by the graph builder, not the user
            out.append(field)
    return out


def candidates(object_info: dict, role_key: str) -> list[dict]:
    """Candidate node classes for a role, each with their widget inputs.

    Each node is processed defensively (like ComfyUI's own /object_info): a node
    with an unexpected schema is skipped, never crashing the whole list."""
    role = ROLES.get(role_key)
    if not role:
        return []
    result = []
    for cls, node_def in object_info.items():
        if not isinstance(node_def, dict):
            continue
        try:
            if not _matches_role(node_def, role):
                continue
            result.append({
                "class": cls,
                "display_name": node_def.get("display_name", cls),
                "category": node_def.get("category", ""),
                "inputs": widget_inputs(node_def),
                "outputs": node_outputs(node_def),
                "connection_inputs": connection_inputs(node_def),
            })
        except Exception:  # noqa: BLE001 - skip malformed node, keep the rest
            continue
    result.sort(key=lambda c: str(c["display_name"]).lower())
    return result


def all_nodes(object_info: dict) -> list[dict]:
    """Lightweight list of every registered node for the 'add any node' picker."""
    out = []
    for cls, nd in object_info.items():
        if not isinstance(nd, dict):
            continue
        try:
            out.append({"class": cls, "display_name": nd.get("display_name", cls),
                        "category": nd.get("category", "")})
        except Exception:  # noqa: BLE001
            continue
    out.sort(key=lambda c: (str(c["category"]), str(c["display_name"]).lower()))
    return out


def describe_node(object_info: dict, cls: str) -> dict | None:
    """Full editor spec for one node class (widgets + outputs + connection inputs)."""
    nd = object_info.get(cls)
    if not isinstance(nd, dict):
        return None
    return {
        "class": cls,
        "display_name": nd.get("display_name", cls),
        "category": nd.get("category", ""),
        "inputs": widget_inputs(nd),
        "outputs": node_outputs(nd),
        "connection_inputs": connection_inputs(nd),
    }


# Fixed-core nodes (built by the editor) whose EXTERNAL inputs accept loader/image-proc
# outputs — exposed as wire destinations. Types resolved from object_info at runtime.
CORE_PORT_NODES = [
    ("LTXVConditioning", "LTXV Conditioning"),
    ("LTXVConcatAVLatent", "Concat AV Latent"),
    ("LTXVSeparateAVLatent", "Separate AV Latent"),
    ("LTXVAudioVAEDecode", "Audio VAE Decode"),
    ("NormalizeAudioLoudness", "Normalize Audio"),
    ("VHS_VideoCombine", "Video Combine"),
    ("FunPackSaveRefinementLatent", "Save Refinement Latent"),
]


def ports_from_input_types(label: str, node_key: str, input_types: dict) -> list[dict]:
    """Pipeline connection points derived from a node's INPUT_TYPES (authoritative)."""
    ports = []
    for group in ("required", "optional"):
        for name, spec in (input_types.get(group) or {}).items():
            if not isinstance(spec, (list, tuple)) or not spec:
                continue
            t = spec[0]
            if not isinstance(t, str):
                continue
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            # Expose typed sockets, plus forceInput widget sockets (e.g. the STRING
            # refinement_key_input on Studio/Sampler) so they can be wired manually.
            if _is_widget_type(t) and not opts.get("forceInput"):
                continue
            ports.append({"id": f"{node_key}.{name}", "node": label, "input": name,
                          "type": t, "label": f"{label} · {name}"})
    return ports


def ports_from_object_info(object_info: dict, cls: str, label: str) -> list[dict]:
    nd = object_info.get(cls)
    if not isinstance(nd, dict):
        return []
    ports = []
    for ci in connection_inputs(nd):
        ports.append({"id": f"{cls}.{ci['name']}", "node": label, "input": ci["name"],
                      "type": ci["type"], "label": f"{label} · {ci['name']}"})
    return ports


def pipeline_requirements() -> list[dict]:
    return PIPELINE_REQUIREMENTS


def core_producers() -> list[dict]:
    """Typed outputs from the fixed core that slot nodes can source from."""
    return [
        {"id": "core:frames:0", "type": "INT",   "label": "Project frames (primitive)"},
        {"id": "core:fps:0",    "type": "FLOAT",  "label": "Project FPS (primitive)"},
        {"id": "core:f2i:0",    "type": "INT",    "label": "Project FPS as int (primitive)"},
    ]


def pipeline_ports(object_info: dict | None = None) -> list[dict]:
    """The fixed path's connection points loaders/nodes wire into. FunPack nodes derive from
    their INPUT_TYPES; LTXV core nodes derive from object_info. [] if nothing is loaded."""
    ports = []
    try:
        try:
            from conditioning import FunPackStudio
            from samplers import FunPackLTXAVSceneChainSampler
        except ImportError:
            from ...conditioning import FunPackStudio  # type: ignore
            from ...samplers import FunPackLTXAVSceneChainSampler  # type: ignore
        ports += ports_from_input_types("Studio", "FunPackStudio", FunPackStudio.INPUT_TYPES())
        ports += ports_from_input_types("Chain Sampler", "FunPackLTXAVSceneChainSampler",
                                        FunPackLTXAVSceneChainSampler.INPUT_TYPES())
    except Exception:
        pass
    if object_info:
        for cls, label in CORE_PORT_NODES:
            ports += ports_from_object_info(object_info, cls, label)
    return ports


def roles_payload() -> list[dict]:
    return [{"key": k, **v} for k, v in ROLES.items()]


# ── persistence (global engine config) ────────────────────────────────────────

def _models_path():
    return config.DATA_DIR / "models.json"


def load_models() -> dict:
    config.ensure_dirs()
    p = _models_path()
    if not p.exists():
        return {"slots": []}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {"slots": []}


def save_models(data: dict) -> dict:
    config.ensure_dirs()
    if not isinstance(data, dict):
        data = {"slots": []}
    data.setdefault("slots", [])
    _models_path().write_text(json.dumps(data, indent=2))
    return data
