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
from .nodes import WIDGET_PRIMITIVES, connection_inputs, node_outputs

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
            elif t in WIDGET_PRIMITIVES:
                if "default" in opts:
                    out[name] = opts["default"]
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


def build(object_info: dict, models_config: dict, params: dict) -> tuple[dict, dict]:
    """Return (graph, report). `params`: prompt, negative_prompt, seed,
    num_frames_per_scene, frame_rate. `models_config`: {"slots": [...]}."""
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
    # the result must be written to the output dir so the editor can fetch it back
    # (the reference graph used preview-only save_output=False).
    graph["vhs"]["inputs"]["save_output"] = True

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
        inputs.update(s.get("inputs") or {})
        graph[sid] = {"class_type": cls, "inputs": inputs}
        if cls not in object_info:
            msg = f"Slot node '{cls}' is not installed in ComfyUI."
            report["unsatisfied"].append(msg); report["blocking"].append(msg)

    # 3b. linked inputs: one shared value drives several node inputs (e.g. width/height).
    for link in (models_config or {}).get("links") or []:
        val = link.get("value")
        if val is None:
            continue
        for m in link.get("members") or []:
            sid = slot_node_id.get(m.get("slotId"))
            if sid and sid in graph:
                graph[sid]["inputs"][m.get("input")] = val

    # 4. explicit wires (slot OUTPUT -> port:<id> | node:<slotId>:<input>).
    port_to_core = _port_index(object_info)
    for s in slots:
        sid = slot_node_id[s["id"]]
        nd = slot_def[s["id"]]
        for out_name, target in (s.get("wires") or {}).items():
            if not target:
                continue
            oidx = _output_index(nd, out_name)
            dst = _resolve_target(target, port_to_core, slot_node_id)
            if not dst:
                report["unsatisfied"].append(f"{s.get('node_class')}.{out_name}: wire target '{target}' could not be resolved.")
                continue
            dnode, dinput = dst
            graph[dnode]["inputs"][dinput] = [sid, oidx]
            report["wired"].append(f"{s.get('node_class')}.{out_name} -> {dnode}.{dinput}")

    # 5. auto-wire remaining unbound typed inputs by unique producer.
    producers = _producers(graph, slots, slot_node_id, slot_def, object_info)
    _autowire(graph, slots, slot_node_id, slot_def, object_info, producers, report)

    return graph, report


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


def _autowire(graph, slots, slot_node_id, slot_def, object_info, producers, report):
    targets = list(OPEN_PORTS)  # (core_id, input, type, required)
    for s in slots:  # slot connection inputs (e.g. image-proc vae/image/length)
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
            report["auto_wired"].append(f"{node_id}.{inp} <- {cands[0][0]} ({t})")
        elif len(cands) > 1:
            msg = f"{node_id}.{inp} ({t}): {len(cands)} possible sources — wire it explicitly."
            report["ambiguous"].append(msg)
            if required:
                report["blocking"].append(msg)
        else:
            msg = f"{node_id}.{inp} ({t}): no source available."
            report["unsatisfied"].append(msg)
            if required:
                report["blocking"].append(msg)
