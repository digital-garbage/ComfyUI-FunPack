"""Build the API-format generation graph from code (no runtime template).

The editor OWNS the fixed FunPack path; only the loader / image-processing /
empty-latent nodes are user-configured (Models menu -> models.json slots). This
module emits the fixed core in code, splices in the configured slot nodes, and
resolves their wiring:

  1. fixed core  — Studio -> LTXVConditioning -> Chain Sampler -> Separate AV ->
     Audio VAE Decode -> Normalize -> VHS Combine, plus Concat AV, SaveRefinement,
     RefinementKeyLoader and the fps/frames primitives. Internal links + widget
     values are derived from the reference workflow (backend/reference/...).
  2. slot nodes  — one per models.json slot, widget values from the slot config.
  3. wiring      — explicit "Wire to" links first (slot output -> core port or
     another slot input), then AUTO-WIRE by type: any still-unbound typed input is
     filled from the unique producer of that type. Ambiguous / unsatisfied inputs
     are reported, not guessed.

Output is a ComfyUI /prompt graph: {node_id: {class_type, inputs}}.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from . import config
from .nodes import WIDGET_PRIMITIVES, connection_inputs, node_outputs, _combo_default

# ── fixed core ────────────────────────────────────────────────────────────────
# logical id -> class_type
CORE: dict[str, str] = {
    "studio": "FunPackStudio",
    "cond": "LTXVConditioning",
    "sampler": "FunPackLTXAVSceneChainSampler",
    "concat": "LTXVConcatAVLatent",
    "separate": "LTXVSeparateAVLatent",
    "audiodec": "LTXVAudioVAEDecode",
    "normaudio": "NormalizeAudioLoudness",
    "vhs": "VHS_VideoCombine",
    "saveref": "FunPackSaveRefinementLatent",
    "keyloader": "FunPackRefinementKeyLoader",
    "pos": "PrimitiveStringMultiline",
    "neg": "PrimitiveStringMultiline",
    "frames": "PrimitiveInt",
    "fps": "PrimitiveFloat",
    "f2i": "LTXFloatToInt",
}

# reference-workflow node id per core node (to seed widget values from the user's
# real settings). Missing -> bare object_info / param defaults.
REF_ID: dict[str, int] = {
    "studio": 5299, "cond": 5147, "sampler": 5333, "concat": 4528, "separate": 4845,
    "audiodec": 5021, "normaudio": 5314, "vhs": 5019, "saveref": 5328, "keyloader": 5303,
    "pos": 5012, "neg": 5256, "frames": 4988, "fps": 4989, "f2i": 5145,
}

# core internal links: core_id -> {input_name: (src_core_id, output_index)}
CORE_LINKS: dict[str, dict[str, tuple[str, int]]] = {
    "studio":   {"positive_prompt": ("pos", 0), "negative_prompt": ("neg", 0),
                 "refinement_key_input": ("keyloader", 0)},
    "cond":     {"positive": ("studio", 1), "negative": ("studio", 2), "frame_rate": ("fps", 0)},
    "sampler":  {"model": ("studio", 0), "positive": ("cond", 0), "negative": ("cond", 1),
                 "sampler": ("studio", 4), "sigmas": ("studio", 5),
                 "latent_template": ("concat", 0), "refinement_key_input": ("keyloader", 0),
                 "num_frames_per_scene": ("frames", 0)},
    "concat":   {"video_latent": ("studio", 12)},
    "separate": {"av_latent": ("sampler", 0)},
    "audiodec": {"samples": ("separate", 1)},
    "normaudio": {"audio": ("audiodec", 0)},
    "vhs":      {"images": ("sampler", 1), "audio": ("normaudio", 0), "frame_rate": ("fps", 0)},
    "saveref":  {"latent": ("separate", 0), "refinement_key": ("keyloader", 0)},
    "f2i":      {"a": ("fps", 0)},
}

# core inputs fed by a user slot (the old group-node outputs). `required` controls
# whether an unsatisfied/ambiguous one blocks generation — source_image and latent
# are optional on FunPackStudio, so a text-to-video montage is valid without them.
OPEN_PORTS: list[tuple[str, str, str, bool]] = [   # (core_id, input, type, required)
    ("studio", "model", "MODEL", True),
    ("studio", "clip", "CLIP", True),
    ("studio", "source_image", "IMAGE", False),
    ("studio", "latent", "LATENT", False),
    ("sampler", "vae", "VAE", True),
    ("concat", "audio_latent", "LATENT", True),
    ("audiodec", "audio_vae", "VAE", True),
]

# core outputs offered as auto-wire producers for slot inputs (image-proc etc.).
CORE_PRODUCERS: list[tuple[str, int, str]] = [  # (core_id, output_index, type)
    ("frames", 0, "INT"),
    ("fps", 0, "FLOAT"),
    ("f2i", 0, "INT"),
]

CONTROL_VALUES = {"fixed", "randomize", "increment", "decrement"}
# Type-appropriate empty values for widgets that declare no default.
_WIDGET_EMPTY = {"STRING": "", "INT": 0, "FLOAT": 0.0, "BOOLEAN": False}
_VHS_NON_INPUT = {"videopreview"}


class BuildError(RuntimeError):
    pass


# ── widget extraction (UI widgets_values -> named inputs) ─────────────────────

def _ordered_widget_names(node_def: dict) -> list[str]:
    names = []
    inp = node_def.get("input") or {}
    for group in ("required", "optional"):
        for name, spec in (inp.get(group) or {}).items():
            if not isinstance(spec, (list, tuple)) or not spec:
                continue
            t = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if opts.get("forceInput"):
                continue  # forceInput STRING is a socket, never a widget value
            if isinstance(t, list) or t in WIDGET_PRIMITIVES:
                names.append(name)
    return names


def extract_widgets(node_def: Optional[dict], widgets_values: Any) -> dict:
    """Map a UI node's widgets_values onto named inputs using INPUT_TYPES order.

    Handles the control_after_generate companion value that follows numeric
    widgets (seed/value), and VHS-style dict widget stores.
    """
    if isinstance(widgets_values, dict):
        return {k: v for k, v in widgets_values.items() if k not in _VHS_NON_INPUT}
    if not isinstance(widgets_values, list) or not node_def:
        return {}
    names = _ordered_widget_names(node_def)
    out: dict[str, Any] = {}
    vi = 0
    for nm in names:
        if vi >= len(widgets_values):
            break
        val = widgets_values[vi]; vi += 1
        out[nm] = val
        if (vi < len(widgets_values) and isinstance(val, (int, float))
                and isinstance(widgets_values[vi], str) and widgets_values[vi] in CONTROL_VALUES):
            vi += 1  # skip control_after_generate
    return out


def _widget_defaults(node_def: Optional[dict]) -> dict:
    out = {}
    if not node_def:
        return out
    inp = node_def.get("input") or {}
    for group in ("required", "optional"):
        for name, spec in (inp.get(group) or {}).items():
            if not isinstance(spec, (list, tuple)) or not spec:
                continue
            t = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if opts.get("forceInput"):
                continue
            if isinstance(t, list):
                out[name] = opts.get("default", t[0] if t else None)
            elif isinstance(t, str) and "COMBO" in t.upper():
                # "COMBO" or a V3 dynamic combo (COMFY_DYNAMICCOMBO_V3): emit the selected
                # key string (options may be {"key": ...} dicts), not the option object.
                out[name] = _combo_default(opts)
            elif t in WIDGET_PRIMITIVES:
                # Always emit a value for every widget — ComfyUI's frontend does, and a
                # required widget with no declared default (e.g. ImageTransform's bboxes)
                # otherwise goes missing and ComfyUI rejects the prompt. Fall back to a
                # type-appropriate empty so generation isn't blocked.
                out[name] = opts.get("default", _WIDGET_EMPTY.get(t, ""))
    return out


# ── reference workflow ────────────────────────────────────────────────────────

def load_reference() -> dict:
    try:
        return json.loads((config.REFERENCE_DIR / "montage.workflow.json").read_text())
    except Exception:
        return {}


def _ref_widgets(ref: dict) -> dict[int, Any]:
    return {n.get("id"): n.get("widgets_values") for n in (ref.get("nodes") or [])}


# ── graph assembly ────────────────────────────────────────────────────────────

def _output_index(node_def: Optional[dict], out_name: str) -> int:
    for i, o in enumerate(node_outputs(node_def or {})):
        if o["name"] == out_name:
            return i
    return 0


def build(object_info: dict, models_config: dict, params: dict, media: dict | None = None) -> tuple[dict, dict]:
    """Return (graph, report). `params`: prompt, negative_prompt, seed,
    num_frames_per_scene, frame_rate, width, height. `models_config`: {"slots":[...],
    "links":[...]}. `media`: optional {"filename": <comfy-input file>, "target": wire}."""
    object_info = object_info or {}
    ref_wv = _ref_widgets(load_reference())
    graph: dict[str, dict] = {}
    # `blocking` is the subset of problems that should stop generation (required inputs).
    report: dict[str, list] = {"wired": [], "auto_wired": [], "ambiguous": [], "unsatisfied": [], "blocking": []}

    missing = [c for c in CORE if c not in ("",) and CORE[c] not in object_info]
    if missing:
        msg = "Missing core node classes in ComfyUI: " + ", ".join(sorted(CORE[c] for c in missing))
        report["unsatisfied"].append(msg); report["blocking"].append(msg)

    # 1. core nodes: widget defaults <- reference values <- core links.
    for cid, cls in CORE.items():
        nd = object_info.get(cls)
        inputs = _widget_defaults(nd)
        inputs.update(extract_widgets(nd, ref_wv.get(REF_ID.get(cid))))
        for inp, (src, idx) in CORE_LINKS.get(cid, {}).items():
            inputs[inp] = [src, idx]
        graph[cid] = {"class_type": cls, "inputs": inputs}

    # 2. param overrides on the primitives + sampler seed.
    graph["pos"]["inputs"]["value"] = params.get("prompt", "")
    if params.get("negative_prompt") is not None:
        graph["neg"]["inputs"]["value"] = params["negative_prompt"]
    if params.get("num_frames_per_scene") is not None:
        graph["frames"]["inputs"]["value"] = params["num_frames_per_scene"]
    if params.get("frame_rate") is not None:
        graph["fps"]["inputs"]["value"] = params["frame_rate"]
    if params.get("seed") is not None:
        graph["sampler"]["inputs"]["seed"] = params["seed"]
    if params.get("max_scenes") is not None:
        graph["sampler"]["inputs"]["max_scenes"] = params["max_scenes"]
    # UI renders are EPHEMERAL: write to ComfyUI's temp dir (cleared on restart), not
    # the output dir. Persisting a result is the user's job via Export (Save dialog).
    # save_output=False routes VHS_VideoCombine to temp; history reports type="temp",
    # which the result proxy + final-render concat resolve against the temp directory.
    graph["vhs"]["inputs"]["save_output"] = False

    # 2b. project-level widget overrides for the built-in FunPack nodes.
    for k, v in (params.get("studio_inputs") or {}).items():
        if k not in graph["studio"]["inputs"] or not isinstance(graph["studio"]["inputs"][k], list):
            graph["studio"]["inputs"][k] = v

    # split_by_transitions is NOT a top-level Studio input — Studio reads it from
    # studio_settings.refiner.split_by_transitions (default False = single-scene mode).
    # The Movie Editor always builds multi-scene combined prompts, so force it ON inside
    # the settings JSON (after the overrides above so it can't be turned off).
    _ss = graph["studio"]["inputs"].get("studio_settings")
    try:
        _ss = json.loads(str(_ss or "{}"))
    except Exception:
        _ss = {}
    if not isinstance(_ss, dict):
        _ss = {}
    _rf = _ss.get("refiner") if isinstance(_ss.get("refiner"), dict) else {}
    _rf["split_by_transitions"] = True
    # reset_session also lives in studio_settings.refiner — armed per-run by the editor
    # (first run after the user clicks "Reset Studio session"); explicit so it's never
    # left on from a previous run.
    _rf["reset_session"] = bool(params.get("reset_session"))
    _ss["refiner"] = _rf
    graph["studio"]["inputs"]["studio_settings"] = json.dumps(_ss)
    for k, v in (params.get("sampler_inputs") or {}).items():
        if k not in graph["sampler"]["inputs"] or not isinstance(graph["sampler"]["inputs"][k], list):
            graph["sampler"]["inputs"][k] = v

    # 3. slot nodes.
    slots = (models_config or {}).get("slots") or []
    slot_node_id = {}
    slot_def = {}
    for s in slots:
        sid = "slot_" + str(s.get("id"))
        cls = s.get("node_class")
        nd = object_info.get(cls)
        slot_node_id[s["id"]] = sid
        slot_def[s["id"]] = nd
        inputs = _widget_defaults(nd)
        defaults = dict(inputs)
        inputs.update(s.get("inputs") or {})
        # Coerce stale/invalid combo values (e.g. a saved "[object Object]" or a dict from
        # an earlier bug) back to the node's default key.
        for wname, wval in list(inputs.items()):
            if isinstance(wval, dict) or wval == "[object Object]":
                inputs[wname] = defaults.get(wname, "")
        graph[sid] = {"class_type": cls, "inputs": inputs}
        if cls not in object_info:
            msg = f"Slot node '{cls}' is not installed in ComfyUI."
            report["unsatisfied"].append(msg); report["blocking"].append(msg)

    # 3b. explicit input sources: user-chosen source for a slot's connection input.
    # Runs before auto-wire so these pre-empt the uniqueness heuristic.
    # source values: "" / "auto" → skip; "timeline" → scene image (media_load);
    #   "out:<slotId>:<outName>" → another slot's named output;
    #   "core:<coreId>:<outIdx>" → a core primitive output.
    for s in slots:
        sid = slot_node_id[s["id"]]
        nd_s = slot_def[s["id"]]
        for ci_name, source in (s.get("input_sources") or {}).items():
            if not source or source == "auto":
                continue
            if source == "timeline":
                if media and media.get("filename"):
                    graph.setdefault("media_load", {
                        "class_type": "LoadImage", "inputs": {"image": media["filename"]}})
                    graph[sid]["inputs"][ci_name] = ["media_load", 0]
                    report["wired"].append(f"timeline image -> {sid}.{ci_name}")
                continue
            src = _resolve_source(source, slot_node_id, slot_def, object_info)
            if src:
                graph[sid]["inputs"][ci_name] = list(src)
                report["wired"].append(f"{source} -> {sid}.{ci_name}")
            else:
                report["unsatisfied"].append(
                    f"{s.get('node_class')}.{ci_name}: input source '{source}' could not be resolved.")

    # 3d. linked inputs: one shared value drives several node inputs (e.g. width/height).
    # An "editor"-sourced link pulls its value from a project setting (params) instead.
    for link in (models_config or {}).get("links") or []:
        if link.get("source") == "editor":
            val = params.get(link.get("editor_key"))
        else:
            val = link.get("value")
        if val is None:
            continue
        for m in link.get("members") or []:
            sid = slot_node_id.get(m.get("slotId"))
            if sid and sid in graph:
                graph[sid]["inputs"][m.get("input")] = val

    port_to_core = _port_index(object_info)

    # 3e. media: inject a LoadImage for the chosen asset, wired to the chosen input(s).
    # scene.source.target provides a legacy explicit target; "timeline" input_sources
    # (step 3b) may have already created media_load — this step adds any extra targets.
    if media and media.get("filename"):
        # Always materialise the scene image as a LoadImage so it can feed IMAGE inputs
        # (it's the timeline's image — there's no node that "outputs" it otherwise).
        graph.setdefault("media_load", {
            "class_type": "LoadImage", "inputs": {"image": media["filename"]}})
        if media.get("target"):
            dst = _resolve_target(media["target"], port_to_core, slot_node_id)
            if dst and dst[0] in graph:
                graph[dst[0]]["inputs"][dst[1]] = ["media_load", 0]
                report["wired"].append(f"media -> {dst[0]}.{dst[1]}")
            else:
                report["unsatisfied"].append(
                    f"media target '{media['target']}' could not be resolved.")

    # 4. explicit wires (slot OUTPUT -> port:<id> | node:<slotId>:<input>).
    # target may be a string (legacy single) or a list of strings (multi-wire).
    for s in slots:
        sid = slot_node_id[s["id"]]
        nd = slot_def[s["id"]]
        for out_name, target in (s.get("wires") or {}).items():
            if not target:
                continue
            targets = target if isinstance(target, list) else [target]
            oidx = _output_index(nd, out_name)
            for t in targets:
                if not t:
                    continue
                dst = _resolve_target(t, port_to_core, slot_node_id)
                if not dst:
                    report["unsatisfied"].append(f"{s.get('node_class')}.{out_name}: wire target '{t}' could not be resolved.")
                    continue
                dnode, dinput = dst
                graph[dnode]["inputs"][dinput] = [sid, oidx]
                report["wired"].append(f"{s.get('node_class')}.{out_name} -> {dnode}.{dinput}")

    # 4b. core input overrides: redirect a built-in core node input to a chosen source
    # ("core:<coreId>:<idx>" or "out:<slotId>:<outName>"). Overrides the default wiring.
    for cid, ovs in ((models_config or {}).get("core_overrides") or {}).items():
        if cid not in graph:
            continue
        for inp, source in (ovs or {}).items():
            if not source:
                continue
            src = _resolve_source(source, slot_node_id, slot_def, object_info)
            if src:
                graph[cid]["inputs"][inp] = list(src)
                report["wired"].append(f"{source} -> {cid}.{inp} (core override)")
            else:
                report["unsatisfied"].append(f"core override {cid}.{inp}: '{source}' could not be resolved.")

    # 5. auto-wire remaining unbound typed inputs by unique producer.
    producers = _producers(graph, slots, slot_node_id, slot_def, object_info)
    # The timeline scene image (LoadImage) is an IMAGE producer too, so a node's IMAGE
    # input (e.g. an ImageTransform) auto-wires to the scene image instead of reporting
    # 'no node outputs IMAGE'.
    if "media_load" in graph:
        producers.setdefault("IMAGE", []).append(("media_load", 0))
    # A slot only matters if it's wired into the pipeline (an output wire, or referenced
    # as a source by a core override). Inert/unused slots are NOT validated or auto-wired,
    # so an unused node's required input never blocks generation.
    active_slots = set()
    for s in slots:
        if any(t for tg in (s.get("wires") or {}).values() for t in (tg if isinstance(tg, list) else [tg]) if t):
            active_slots.add(s["id"])
    for ovs in ((models_config or {}).get("core_overrides") or {}).values():
        for src in (ovs or {}).values():
            if isinstance(src, str) and src.startswith("out:"):
                active_slots.add(src.split(":", 2)[1])
    _autowire(graph, slots, slot_node_id, slot_def, object_info, producers, report, active_slots)

    return graph, report


def core_graph(object_info: dict, models_config: dict | None = None) -> list[dict]:
    """Describe the built-in core pipeline for the editor: each node, its inputs (with the
    current source + candidate sources so they can be re-wired) and outputs (with internal
    destinations). Inputs default to the built-in wiring; `core_overrides` in models_config
    can redirect any of them (applied by build())."""
    object_info = object_info or {}
    overrides = (models_config or {}).get("core_overrides") or {}
    slots = (models_config or {}).get("slots", []) or []

    rev: dict[str, list] = {}
    for cid, links in CORE_LINKS.items():
        for inp, (src, idx) in links.items():
            rev.setdefault(src, []).append((idx, cid, inp))

    def _outs(core_id):
        return node_outputs(object_info.get(CORE.get(core_id)) or {})

    def _out_name(core_id, idx):
        outs = _outs(core_id)
        return outs[idx]["name"] if 0 <= idx < len(outs) else f"out{idx}"

    # typed producers (core outputs + slot outputs) for the source pickers
    producers: dict[str, list] = {}
    for cid2, cls2 in CORE.items():
        for i, o in enumerate(node_outputs(object_info.get(cls2) or {})):
            producers.setdefault(o["type"], []).append((f"core:{cid2}:{i}", f"{cid2} · {o['name']}"))
    slot_label = {}
    for s in slots:
        sid = s.get("id"); lbl = s.get("label") or s.get("node_class") or sid
        slot_label[sid] = lbl
        for o in node_outputs(object_info.get(s.get("node_class")) or {}):
            producers.setdefault(o["type"], []).append((f"out:{sid}:{o['name']}", f"{lbl} · {o['name']}"))

    open_by_node: dict[str, dict] = {}
    for (cid, inp, t, req) in OPEN_PORTS:
        open_by_node.setdefault(cid, {})[inp] = (t, req)

    def _input_type(cid, inp):
        for ci in connection_inputs(object_info.get(CORE.get(cid)) or {}):
            if ci["name"] == inp:
                return ci["type"]
        return None

    def _options(t, builtin_label, self_cid):
        opts = [{"value": "", "label": f"built-in: {builtin_label}"}]
        for val, lbl in producers.get(t, []):
            if val.startswith(f"core:{self_cid}:"):
                continue
            opts.append({"value": val, "label": lbl})
        return opts

    nodes = []
    for cid, cls in CORE.items():
        nd = object_info.get(cls)
        ov = overrides.get(cid) or {}
        inputs = []
        for inp, (src, idx) in CORE_LINKS.get(cid, {}).items():
            t = _input_type(cid, inp) or "*"
            builtin = f"{src} · {_out_name(src, idx)}"
            inputs.append({"name": inp, "type": t, "from": "internal", "detail": builtin,
                           "value": ov.get(inp, ""), "options": _options(t, builtin, cid)})
        for inp, (t, req) in open_by_node.get(cid, {}).items():
            builtin = f"a {t} from your loaders"
            inputs.append({"name": inp, "type": t, "from": "loader", "required": req,
                           "detail": builtin + ("" if req else " (optional)"),
                           "value": ov.get(inp, ""), "options": _options(t, "(auto-wire from loaders)", cid)})
        outputs = []
        for i, o in enumerate(node_outputs(nd or {})):
            dests = [f"{d} · {di}" for (oi, d, di) in rev.get(cid, []) if oi == i]
            outputs.append({"name": o["name"], "type": o["type"], "to": dests})
        nodes.append({"id": cid, "class": cls,
                      "display_name": (nd or {}).get("display_name", cls),
                      "installed": cls in object_info,
                      "inputs": inputs, "outputs": outputs})
    return nodes


def _port_index(object_info: dict) -> dict[str, tuple[str, str]]:
    """Map pipeline-port id ('Class.input' / 'FunPackStudio.input') -> (core_id, input)."""
    cls_to_core = {v: k for k, v in CORE.items()}
    idx: dict[str, tuple[str, str]] = {}
    for cid, cls in CORE.items():
        nd = object_info.get(cls)
        for ci in connection_inputs(nd or {}):
            idx[f"{cls}.{ci['name']}"] = (cid, ci["name"])
    # FunPack ports use the class name directly too (already covered above).
    return idx


def _resolve_source(source: str, slot_node_id: dict, slot_def: dict, object_info: dict) -> Optional[tuple[str, int]]:
    """Resolve an input-source id to (node_id, output_index).
    Formats: "out:<slotId>:<outName>" | "core:<coreId>:<outIdx>" | "timeline" handled by caller.
    """
    if source.startswith("out:"):
        _, sid, out_name = source.split(":", 2)
        nid = slot_node_id.get(sid)
        if not nid:
            return None
        nd = slot_def.get(sid)
        return (nid, _output_index(nd, out_name))
    if source.startswith("core:"):
        _, cid, oidx = source.split(":", 2)
        return (cid, int(oidx))
    return None


def _resolve_target(target: str, port_to_core, slot_node_id) -> Optional[tuple[str, str]]:
    if target.startswith("port:"):
        return port_to_core.get(target[5:])
    if target.startswith("node:"):
        _, sid, inp = target.split(":", 2)
        node = slot_node_id.get(sid)
        return (node, inp) if node else None
    return None


def _producers(graph, slots, slot_node_id, slot_def, object_info):
    """type -> list of (node_id, output_index). Slots + core producers."""
    out: dict[str, list] = {}
    for cid, oidx, t in CORE_PRODUCERS:
        out.setdefault(t, []).append((cid, oidx))
    for s in slots:
        nd = slot_def[s["id"]]
        for i, o in enumerate(node_outputs(nd or {})):
            out.setdefault(o["type"], []).append((slot_node_id[s["id"]], i))
    return out


def _node_labels(slots, slot_node_id, object_info):
    """node_id -> human label ('Role / Name [NodeClass]') for report messages, so users
    don't see opaque ids like 'slot_5u09gm5'."""
    label = {}
    for cid, cls in CORE.items():
        label[cid] = f"{(object_info.get(cls) or {}).get('display_name', cls)} [{cls}]"
    for s in slots:
        nid = slot_node_id.get(s["id"])
        if not nid:
            continue
        cls = s.get("node_class") or ""
        nm = s.get("label") or s.get("role_label") or cls or nid
        label[nid] = f"{nm} [{cls}]" if cls and nm != cls else (nm or nid)
    return label


def _autowire(graph, slots, slot_node_id, slot_def, object_info, producers, report, active_slots=None):
    label = _node_labels(slots, slot_node_id, object_info)
    L = lambda nid: label.get(nid, nid)

    targets = list(OPEN_PORTS)  # (core_id, input, type, required)
    for s in slots:  # slot connection inputs (e.g. image-proc vae/image/length)
        if active_slots is not None and s["id"] not in active_slots:
            continue  # inert slot (feeds nothing) — don't auto-wire or block on its inputs
        nd = slot_def[s["id"]]
        for ci in connection_inputs(nd or {}):
            targets.append((slot_node_id[s["id"]], ci["name"], ci["type"], ci.get("required", False)))

    for node_id, inp, t, required in targets:
        node = graph.get(node_id)
        if not node:
            continue
        if isinstance(node["inputs"].get(inp), list):
            continue  # already wired (explicit/core)
        cands = [p for p in producers.get(t, []) if p[0] != node_id]
        if len(cands) == 1:
            node["inputs"][inp] = [cands[0][0], cands[0][1]]
            report["auto_wired"].append(f"{L(node_id)}.{inp} <- {L(cands[0][0])} ({t})")
        elif len(cands) > 1:
            msg = f"{L(node_id)}.{inp} ({t}): {len(cands)} possible sources — wire it explicitly."
            report["ambiguous"].append(msg)
            if required:
                report["blocking"].append(msg)
        else:
            msg = f"{L(node_id)}.{inp} ({t}): no node outputs {t} — add one in Models, or this input may not be needed."
            report["unsatisfied"].append(msg)
            if required:
                report["blocking"].append(msg)
