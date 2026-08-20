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


def type_parts(t) -> list[str]:
    """Member types of a ComfyUI type string. A V3 MultiType input serializes its members
    comma-joined — "FLOAT,INT" (an int widget that also accepts a float link), "IMAGE,MASK"
    (a socket taking either) — so a single string can name several types."""
    if not isinstance(t, str):
        return []
    return [p.strip() for p in t.split(",") if p.strip()]


def widget_type_of(t, opts: dict | None = None) -> str | None:
    """The widget type an input renders as, or None if it is a pure graph socket.

    A V3 MultiType wrapped around a widget input (io.MultiType.Input(io.Int.Input(...), ...))
    is a WIDGET in ComfyUI's frontend, not a socket: it carries "widgetType" in its options
    and lists the widget's own type first, e.g. LTXVEmptyLatentAudio.frame_rate arriving as
    ("FLOAT,INT", {"widgetType": "INT", "default": 25}). Treating it as a required socket
    would demand a source for a field the user simply types into."""
    opts = opts or {}
    wt = opts.get("widgetType")
    if _is_widget_type(wt):
        return wt
    parts = type_parts(t)
    if parts and all(_is_widget_type(p) for p in parts):
        return parts[0]
    return None


def link_types(t) -> list[str]:
    """The graph-connection member types of a type string (widget members dropped)."""
    return [_normalize_type(p) for p in type_parts(t) if not _is_widget_type(p)]


def type_accepts(input_type: str, output_type: str) -> bool:
    """Whether an output of `output_type` can feed an input of `input_type`. Either side may
    be a union, so they match when any member type is shared."""
    a, b = type_parts(input_type), type_parts(output_type)
    if not a or not b:
        return input_type == output_type
    return bool({_normalize_type(x) for x in a} & {_normalize_type(y) for y in b})


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


def _is_list_widget(t, opts: dict) -> bool:
    """A FunPack list input (see widgets.py): a STRING carrying a `funpack_list` row spec."""
    return t == "STRING" and isinstance(opts.get("funpack_list"), dict)


def _is_autogrow_type(t) -> bool:
    """A V3 autogrow list input (COMFY_AUTOGROW_V3): not a socket itself, but a template
    that ComfyUI expands into one real socket per index (ref_image0, ref_image1, …)."""
    return isinstance(t, str) and "AUTOGROW" in t.upper()


def _autogrow_names(opts: dict) -> list[str]:
    """The socket names an autogrow input expands to. Two template flavours: a prefix with
    a max count ("ref_image" x 10 -> ref_image0..ref_image9) or an explicit name list."""
    tpl = opts.get("template")
    if not isinstance(tpl, dict):
        return []
    names = tpl.get("names")
    if isinstance(names, list):
        return [str(n) for n in names]
    prefix, mx = tpl.get("prefix"), tpl.get("max")
    if isinstance(prefix, str) and isinstance(mx, int) and mx > 0:
        return [f"{prefix}{i}" for i in range(mx)]
    return []


def _autogrow_element_type(opts: dict) -> str | None:
    """The type of ONE element of an autogrow list — the template's single input, e.g. IMAGE
    for ref_images. Widget-typed templates are forced to sockets by ComfyUI, so they count too."""
    tpl = opts.get("template")
    if not isinstance(tpl, dict):
        return None
    inp = tpl.get("input")
    if not isinstance(inp, dict):
        return None
    for group in ("required", "optional"):
        for spec in (inp.get(group) or {}).values():
            if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], str):
                return spec[0]
    return None


def _autogrow_children(name: str, opts: dict) -> list[dict]:
    """Expand an autogrow input into its indexed sockets. Empty when the template can't be
    read — the caller then falls back to the raw (unwireable) input so nothing regresses.

    The socket id ComfyUI expects is the PARENT id, a dot, then the template name
    ("ref_images.ref_image_0") — that dotted path is what it splits to rebuild the node's
    list. Sending the bare name is rejected as a missing input, and any that slips past
    validation reaches execute() as an unexpected keyword argument. `display` carries the
    short name for the UI, which has no reason to show the plumbing.
    """
    t = _autogrow_element_type(opts)
    names = _autogrow_names(opts)
    if not t or not names:
        return []
    return [
        # Always optional: the API graph carries only the indices that are actually wired,
        # and ComfyUI grows the schema to match, so an unwired index must never block.
        {"name": f"{name}.{nm}", "display": nm, "type": _normalize_type(t), "required": False,
         "autogrow": {"parent": name, "index": i}}
        for i, nm in enumerate(names)
    ]


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
    "video_latent":  {"label": "Video Latent Source",     "category": "Pipeline", "output": "LATENT"},
    "audio_encoder": {"label": "Audio Encoder",           "category": "Pipeline", "output": "LATENT"},
}

# Pipeline inputs that must be satisfied by configured slot nodes.
# Each entry: id, type, label, required, role_hint (role key to suggest adding), hint.
# MiniMax H3 replaces two of these: its own EmptyMiniMaxH3LatentAV emits the video AND
# audio streams in one node, so there is no separate audio-latent step and the latent is
# no longer optional — Studio does not generate one for this family, so a node has to. It
# needn't be the Empty node: fl2va / ref2va build their own AV latent alongside their
# conditioning, and satisfy this the same way. The audio VAE stays required — it decodes
# the generated audio — and gains a second, optional job on H3 (encoding audio references).
FAMILY_REQUIREMENTS: dict[str, dict] = {
    "minimax_h3": {
        "drop": ("audio_latent",),
        "replace": {
            "video_vae": {"hint": "Add the MiniMax H3 video VAE loader (24-channel, 16x spatial)."},
            "audio_vae": {"hint": "Add the MiniMax H3 audio VAE loader (32 kHz stereo). Decodes the "
                                  "generated audio, and encodes audio references for ref2va."},
            "clip": {"hint": "Add the MiniMax H3 text encoder (Qwen3-VL-32B, truncated to 50 layers)."},
            "init_latent": {"required": True, "label": "AV latent",
                            "hint": "Add Empty MiniMax H3 AV Latent — it makes the video and audio "
                                    "streams together and feeds the Chain Sampler directly. A "
                                    "MiniMax H3 Image to Video (fl2va) or Reference to Video "
                                    "(ref2va) node covers this too: each builds its own AV latent."},
        },
    },
}

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
            opts = spec[1] if isinstance(spec, list) and len(spec) > 1 and isinstance(spec[1], dict) else {}
            if _is_autogrow_type(t):
                # The node consumes the ELEMENT type (IMAGE), not the list wrapper.
                el_t = _autogrow_element_type(opts)
                if el_t:
                    types.extend(link_types(el_t))
                    continue
            if isinstance(t, str) and not widget_type_of(t, opts):
                types.extend(link_types(t))
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
            if _is_autogrow_type(t):
                # A list input (H3's ref_images/ref_videos/…): the parent name is never wired
                # in the API graph — ComfyUI expands it into ref_image0, ref_image1, … and
                # rebuilds the dict from whichever indices are present. Offer those sockets
                # instead, so they can be sourced like any other input.
                children = _autogrow_children(name, opts)
                if children:
                    out.extend(children)
                    continue
            wt = widget_type_of(t, opts)
            if wt:
                # A widget-typed input is a SOCKET (wireable) only when forceInput is set —
                # e.g. refinement_key_input is ("STRING", {"forceInput": True}). It always
                # has a widget fallback, so it's NEVER auto-required (it must not block
                # generation when left unwired). Plain widgets (incl. V3 dynamic combos
                # like COMFY_DYNAMICCOMBO_V3 and MultiType widgets like ("FLOAT,INT", ...))
                # are handled by widget_inputs, not here.
                if not opts.get("forceInput"):
                    continue
                out.append({"name": name, "type": _normalize_type(wt), "required": False})
                continue
            # V3 nodes can mark a required-group input as optional via the flag.
            is_required = group == "required" and not opts.get("optional", False)
            entry = {"name": name, "type": _normalize_type(t), "required": is_required}
            # An input the node itself calls advanced is one the normal path never wires;
            # the panel folds it away so the normal path stays the obvious one.
            if opts.get("advanced"):
                entry["advanced"] = True
            out.append(entry)
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
            # A MultiType widget ("FLOAT,INT") renders as its widget member, not the union.
            if isinstance(t, str):
                t = widget_type_of(t, opts) or t
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
            elif _is_list_widget(t, opts):
                # A FunPack list input: one STRING holding a JSON array of rows, with the row
                # shape declared alongside it so this renders as rows, not raw JSON.
                field["kind"] = "list"
                field["list"] = opts["funpack_list"]
                field["default"] = opts.get("default", "[]")
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

# H3's core drops LTXVConditioning and Concat AV and decodes audio with core's generic node,
# so offering the LTX-only ports here would invite wires into nodes the graph never emits.
FAMILY_CORE_PORT_NODES: dict[str, list] = {
    "minimax_h3": [
        ("LTXVSeparateAVLatent", "Separate AV Latent"),
        ("VAEDecodeAudio", "VAE Decode Audio"),
        ("NormalizeAudioLoudness", "Normalize Audio"),
        ("VHS_VideoCombine", "Video Combine"),
        ("FunPackSaveRefinementLatent", "Save Refinement Latent"),
    ],
}


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
            if widget_type_of(t, opts) and not opts.get("forceInput"):
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


def pipeline_requirements(family: str = "ltxav") -> list[dict]:
    spec = FAMILY_REQUIREMENTS.get(str(family or "").lower())
    if not spec:
        return PIPELINE_REQUIREMENTS
    dropped = set(spec.get("drop") or ())
    replace = spec.get("replace") or {}
    out = []
    for req in PIPELINE_REQUIREMENTS:
        if req["id"] in dropped:
            continue
        out.append({**req, **replace.get(req["id"], {})})
    return out


def core_producers(object_info: dict | None = None) -> list[dict]:
    """Typed outputs from the fixed core that slot nodes can source from.

    The project primitives are always available. When ``object_info`` is supplied, every
    installed core node's outputs are exposed too — so in Full control a custom node (e.g.
    a replacement sampler) can be fed Studio's model / conditioning / sigmas, or the
    sampler's latent. The slot picker only offers these in Full control (see allowedSources);
    build() resolves the chosen ``core:<id>:<idx>`` via _resolve_source."""
    out = [
        {"id": "core:frames:0", "type": "INT",   "label": "Project frames (primitive)"},
        {"id": "core:fps:0",    "type": "FLOAT",  "label": "Project FPS (primitive)"},
        {"id": "core:f2i:0",    "type": "INT",    "label": "Project FPS as int (primitive)"},
    ]
    if not object_info:
        return out
    from . import builder  # lazy: avoid import cycle at module load
    seen = {p["id"] for p in out}
    for cid, cls in builder.CORE.items():
        nd = object_info.get(cls)
        if not nd:
            continue
        node_label = nd.get("display_name", cls)
        for i, o in enumerate(node_outputs(nd)):
            pid = f"core:{cid}:{i}"
            if pid in seen:
                continue
            seen.add(pid)
            out.append({"id": pid, "type": o["type"], "label": f"{node_label} → {o['name']}"})
    return out


def pipeline_ports(object_info: dict | None = None, family: str = "ltxav") -> list[dict]:
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
        for cls, label in FAMILY_CORE_PORT_NODES.get(str(family or "").lower(), CORE_PORT_NODES):
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
