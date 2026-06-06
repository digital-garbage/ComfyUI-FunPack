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
}


def _outputs(node_def: dict) -> list[str]:
    out = node_def.get("output") or []
    # output entries can be a string or a list-of-options; we only care about strings here
    return [o for o in out if isinstance(o, str)]


def _all_input_types(node_def: dict) -> list[str]:
    types = []
    inp = node_def.get("input") or {}
    for group in ("required", "optional"):
        for spec in (inp.get(group) or {}).values():
            t = spec[0] if isinstance(spec, list) and spec else None
            if isinstance(t, str):
                types.append(t)
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
            if isinstance(t, str) and t not in WIDGET_PRIMITIVES:
                out.append({"name": name, "type": t, "required": group == "required"})
    return out


def node_outputs(node_def: dict) -> list[dict]:
    """Outputs a node produces, as {name, type}, so they can be wired onward."""
    types = _outputs(node_def)
    names = node_def.get("output_name") or []
    out = []
    for i, t in enumerate(node_def.get("output") or []):
        if not isinstance(t, str):
            continue
        nm = names[i] if i < len(names) and names[i] else t
        out.append({"name": nm, "type": t})
    return out


def widget_inputs(node_def: dict) -> list[dict]:
    """User-facing widgets for a node: combos (with options) and primitive fields.
    Skips graph-connection inputs (MODEL/CLIP/IMAGE/...)."""
    out = []
    inp = node_def.get("input") or {}
    for group in ("required", "optional"):
        for name, spec in (inp.get(group) or {}).items():
            if not isinstance(spec, list) or not spec:
                continue
            t = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            field = {"name": name, "required": group == "required", "options": opts}
            if isinstance(t, list):  # combo: t is the list of choices (e.g. files)
                field["kind"] = "combo"
                field["choices"] = t
                field["default"] = opts.get("default", t[0] if t else None)
            elif t in WIDGET_PRIMITIVES:
                field["kind"] = t.lower()
                field["default"] = opts.get("default")
            else:
                continue  # connection input — wired by the graph builder, not the user
            out.append(field)
    return out


def candidates(object_info: dict, role_key: str) -> list[dict]:
    """Candidate node classes for a role, each with their widget inputs."""
    role = ROLES.get(role_key)
    if not role:
        return []
    result = []
    for cls, node_def in object_info.items():
        if not isinstance(node_def, dict):
            continue
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
    result.sort(key=lambda c: c["display_name"].lower())
    return result


def ports_from_input_types(label: str, node_key: str, input_types: dict) -> list[dict]:
    """Pipeline connection points derived from a node's INPUT_TYPES (authoritative)."""
    ports = []
    for group in ("required", "optional"):
        for name, spec in (input_types.get(group) or {}).items():
            if not isinstance(spec, (list, tuple)) or not spec:
                continue
            t = spec[0]
            if isinstance(t, str) and t not in WIDGET_PRIMITIVES:
                ports.append({"id": f"{node_key}.{name}", "node": label, "input": name,
                              "type": t, "label": f"{label} · {name}"})
    return ports


def pipeline_ports() -> list[dict]:
    """The fixed path's connection points loaders/nodes wire into. Derived live from the
    FunPack node classes so they always match the real graph; [] if ComfyUI isn't loaded."""
    try:
        try:
            from conditioning import FunPackStudio
            from samplers import FunPackLTXAVSceneChainSampler
        except ImportError:
            from ...conditioning import FunPackStudio  # type: ignore
            from ...samplers import FunPackLTXAVSceneChainSampler  # type: ignore
    except Exception:
        return []
    ports = []
    ports += ports_from_input_types("Studio", "FunPackStudio", FunPackStudio.INPUT_TYPES())
    ports += ports_from_input_types("Chain Sampler", "FunPackLTXAVSceneChainSampler",
                                    FunPackLTXAVSceneChainSampler.INPUT_TYPES())
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
