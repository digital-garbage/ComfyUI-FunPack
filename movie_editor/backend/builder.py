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
import uuid
from typing import Any, Optional

from . import config
from . import pipeline_wiring
from .nodes import WIDGET_PRIMITIVES, connection_inputs, node_outputs, _combo_default, _combo_choices

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


def _widget_choices(node_def: Optional[dict]) -> dict:
    """{name: [choices]} for combo widgets, per the LIVE object_info (e.g. installed
    LoRA files). Used to coerce a saved slot value that no longer exists (renamed/
    removed file) back to a current choice instead of being sent to ComfyUI as-is."""
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
                if t:
                    out[name] = t
            elif isinstance(t, str) and "COMBO" in t.upper():
                choices = _combo_choices(opts)
                if choices:
                    out[name] = choices
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


def _infer_slot_output_type(node_def: Optional[dict], out_name: str) -> Optional[str]:
    for o in node_outputs(node_def or {}):
        if o.get("name") == out_name:
            return o.get("type")
    return None


def build(object_info: dict, models_config: dict, params: dict, media: dict | None = None) -> tuple[dict, dict]:
    """Return (graph, report). `params`: prompt, negative_prompt, seed,
    num_frames_per_scene, frame_rate, width, height. `models_config`: {"slots":[...],
    "links":[...]}. `media`: optional {"filename": <comfy-input file>, "target": wire}."""
    object_info = object_info or {}
    ref_wv = _ref_widgets(load_reference())
    graph: dict[str, dict] = {}
    # `blocking` is the subset of problems that should stop generation (required inputs).
    report: dict[str, list] = {"wired": [], "auto_wired": [], "ambiguous": [], "unsatisfied": [], "blocking": []}
    # (node_id, input) edges the cycle-breaker must never drop: fixed core-internal links and
    # explicit user/override wires. Auto-wires and role-default wires are left droppable.
    protected_edges: set[tuple[str, str]] = set()

    # "Disable built-in pipeline": skip the whole fixed FunPack core graph and run only the
    # user-wired nodes. The final result then comes from whatever is wired to the global
    # video/audio outputs (or any output node the user includes).
    disable_core = bool((models_config or {}).get("disable_core"))

    if not disable_core:
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
                protected_edges.add((cid, inp))
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
        # Per-run unique filename_prefix so each run's output is its OWN temp file. Without
        # this, every run shares the default prefix and a later run can reuse/overwrite an
        # earlier run's filename — so a previously rendered scene stops playing once the next
        # scene generates, even though its temp file is only really gone on ComfyUI restart.
        # Each scene records the exact filename ComfyUI returns, so distinct prefixes keep
        # every scene tied to its own file for the whole session.
        graph["vhs"]["inputs"]["filename_prefix"] = f"funpack_preview_{uuid.uuid4().hex[:12]}"

        # Refinement key for this run. FunPackRefinementKeyLoader resolves target = selected
        # combo OR typed key_name; force the combo to "-None-" so the typed project key wins
        # (an existing combo value seeded from the reference workflow would otherwise override).
        _rkey = str(params.get("refinement_key") or "default").strip() or "default"
        if "keyloader" in graph:
            graph["keyloader"]["inputs"]["key_name"] = _rkey
            graph["keyloader"]["inputs"]["refinement_key"] = "-None-"

        # 2b. project-level widget overrides for the built-in FunPack nodes.
        _me_scene_ratings = None
        for k, v in (params.get("studio_inputs") or {}).items():
            if str(k).startswith("_"):
                if k == "_movie_editor_scene_ratings":
                    _me_scene_ratings = v
                continue
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
        # Explicit scene boundaries from the editor (no `scene N` markers in the prompt).
        # Studio (split_scenes_from_segments) uses this list directly to split scenes.
        _seg = params.get("scene_segments")
        if isinstance(_seg, dict) and _seg.get("scenes"):
            _rf["scenes"] = _seg
        # reset_session also lives in studio_settings.refiner — armed per-run by the editor
        # (first run after the user clicks "Reset Studio session"); explicit so it's never
        # left on from a previous run.
        _rf["reset_session"] = bool(params.get("reset_session"))
        # Project `$name` variables — Studio resolves them dead-last (after split), per scene.
        _vars = params.get("variables")
        if isinstance(_vars, (list, dict)) and _vars:
            _rf["variables"] = _vars
        if _me_scene_ratings:
            _rf["movie_editor_scene_ratings"] = _me_scene_ratings
        _ss["refiner"] = _rf
        graph["studio"]["inputs"]["studio_settings"] = json.dumps(_ss)
        for k, v in (params.get("sampler_inputs") or {}).items():
            if k not in graph["sampler"]["inputs"] or not isinstance(graph["sampler"]["inputs"][k], list):
                graph["sampler"]["inputs"][k] = v

    # The global editor outputs: a VHS_VideoCombine synthesized on demand when a slot output
    # is wired to global:video / global:audio. Lazily created so it only exists when used.
    def _ensure_global_out() -> str:
        if "global_out" in graph:
            return "global_out"
        vhs_cls = CORE.get("vhs", "VHS_VideoCombine")
        nd = object_info.get(vhs_cls)
        g_inputs = _widget_defaults(nd)
        g_inputs["save_output"] = False  # ephemeral, like the core combine
        # Unique per-build prefix so this whole-timeline preview never collides with a
        # per-run scene output (or a prior build) in the shared temp dir.
        g_inputs["filename_prefix"] = f"funpack_global_{uuid.uuid4().hex[:12]}"
        if params.get("frame_rate") is not None and "frame_rate" in g_inputs:
            g_inputs["frame_rate"] = params["frame_rate"]
        graph["global_out"] = {"class_type": vhs_cls, "inputs": g_inputs}
        return "global_out"

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
        choices = _widget_choices(nd)
        inputs.update(s.get("inputs") or {})
        # Coerce stale/invalid combo values back to the node's current default key:
        # a saved "[object Object]" or dict from an earlier bug, or a value (e.g. a
        # LoRA filename) that was renamed/removed and no longer appears in the live
        # choices list — the Models menu shows the live choices, so the saved value
        # should fall back to match what the selector actually displays.
        for wname, wval in list(inputs.items()):
            if isinstance(wval, dict) or wval == "[object Object]":
                inputs[wname] = defaults.get(wname, "")
            elif wname in choices and wval not in choices[wname]:
                inputs[wname] = defaults.get(wname, choices[wname][0])
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
                    protected_edges.add((sid, ci_name))
                    report["wired"].append(f"timeline image -> {sid}.{ci_name}")
                continue
            src = _resolve_source(source, slot_node_id, slot_def, object_info)
            if src:
                graph[sid]["inputs"][ci_name] = list(src)
                protected_edges.add((sid, ci_name))
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
        role_defaults = pipeline_wiring.DEFAULT_WIRES_BY_ROLE.get(s.get("role") or "", {})
        for out_name, target in (s.get("wires") or {}).items():
            if not target:
                continue
            targets = target if isinstance(target, list) else [target]
            oidx = _output_index(nd, out_name)
            out_type = _infer_slot_output_type(nd, out_name)
            # A wire the user explicitly made is protected from the cycle-breaker; one that merely
            # matches this role's default wiring is droppable (it's what the editor auto-added).
            is_default = role_defaults.get(out_type) == target
            for t in targets:
                if not t:
                    continue
                # Global editor outputs: feed the synthesized combine node. global:video ->
                # its IMAGE frames, global:audio -> its AUDIO (muxed into the same video+audio
                # file). Either is optional; wire both for a combined video with sound.
                if t in ("global:video", "global:audio"):
                    gid = _ensure_global_out()
                    dinput = "images" if t == "global:video" else "audio"
                    graph[gid]["inputs"][dinput] = [sid, oidx]
                    if not is_default:
                        protected_edges.add((gid, dinput))
                    report["wired"].append(f"{s.get('node_class')}.{out_name} -> Global {'video' if t == 'global:video' else 'audio'} output")
                    continue
                dst = _resolve_target(t, port_to_core, slot_node_id)
                if not dst:
                    report["unsatisfied"].append(f"{s.get('node_class')}.{out_name}: wire target '{t}' could not be resolved.")
                    continue
                dnode, dinput = dst
                if dnode not in graph:  # e.g. a core-port target while the built-in pipeline is disabled
                    report["unsatisfied"].append(f"{s.get('node_class')}.{out_name}: target '{t}' node is not in the graph (built-in pipeline disabled?).")
                    continue
                graph[dnode]["inputs"][dinput] = [sid, oidx]
                if not is_default:
                    protected_edges.add((dnode, dinput))
                report["wired"].append(f"{s.get('node_class')}.{out_name} -> {dnode}.{dinput}")

    # 4b. core input overrides. In guided mode only loader-facing open ports may be
    # overridden (e.g. Studio · latent from a custom node); internal core links stay fixed.
    for cid, ovs in ((models_config or {}).get("core_overrides") or {}).items():
        if cid not in graph:
            continue
        locked = pipeline_wiring.wiring_locked(models_config)
        for inp, source in (ovs or {}).items():
            if not source:
                continue
            if locked and (cid, inp) not in pipeline_wiring.OPEN_CORE_INPUTS:
                continue
            src = _resolve_source(source, slot_node_id, slot_def, object_info)
            if src:
                graph[cid]["inputs"][inp] = list(src)
                protected_edges.add((cid, inp))
                report["wired"].append(f"{source} -> {cid}.{inp} (core override)")
            else:
                report["unsatisfied"].append(
                    f"core override {cid}.{inp}: '{source}' could not be resolved.")

    # 5. auto-wire remaining unbound typed inputs by unique producer.
    producers = _producers(graph, slots, slot_node_id, slot_def, object_info)
    # Drop producers whose node isn't in the graph (core producers vanish when the built-in
    # pipeline is disabled) so auto-wire never points at a non-existent node.
    for t in list(producers):
        producers[t] = [(nid, oi) for (nid, oi) in producers[t] if nid in graph]
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

    # 5b. bypass: drop a slot's node from the graph and rewire its consumers straight to
    # whatever fed its matching-type input, so the node's effect is skipped without losing
    # its saved configuration (vs. removing the node or zeroing a strength widget every time).
    _apply_bypass(graph, slots, slot_node_id, slot_def, report)

    # Cycle-guard: a default/auto wire that loops (e.g. an upscale node whose IMAGE both feeds the
    # export AND defaults back to Studio · source_image while it consumes the decoded frames) is
    # dropped here; an all-explicit loop is reported precisely. Never hand ComfyUI a cyclic graph.
    _break_cycles(graph, report, protected_edges, _node_labels(slots, slot_node_id, object_info))

    for msg in pipeline_wiring.validate_models_wiring(models_config):
        report["blocking"].append(msg)
        report["unsatisfied"].append(msg)

    # With the built-in pipeline disabled, the result must come from the global outputs (or a
    # save node the user wired). If NO output node exists at all this is BLOCKING: ComfyUI
    # would reject the queue with a raw "Prompt has no outputs" — surface the real cause and
    # the fix instead. If some output-capable node exists, keep it a non-blocking warning
    # (the editor may still not see the result, but the run itself is valid).
    if disable_core and "global_out" not in graph:
        has_output_node = any(
            (object_info.get(n.get("class_type")) or {}).get("output_node")
            for n in graph.values()
        )
        msg = (
            "Built-in pipeline is disabled for this project and nothing is wired to the "
            "🌐 Global video output. Wire a final IMAGE output to it, or re-enable the "
            "built-in pipeline (Models → Enable built-in pipeline)."
        )
        if not has_output_node:
            report["blocking"].append(msg)
        report["unsatisfied"].append(msg)

    return graph, report


def core_graph(object_info: dict, models_config: dict | None = None) -> list[dict]:
    """Describe the built-in core pipeline for the editor: each node, its inputs (with the
    current source + candidate sources so they can be re-wired) and outputs (with internal
    destinations). Inputs default to the built-in wiring; `core_overrides` in models_config
    can redirect any of them (applied by build())."""
    object_info = object_info or {}
    overrides = (models_config or {}).get("core_overrides") or {}
    slots = (models_config or {}).get("slots", []) or []
    locked = pipeline_wiring.wiring_locked(models_config)

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

    # Slot inputs that pull from a core output (input_sources "core:<cid>:<idx>") so each
    # core output can show where it's tapped — e.g. Studio · cond → MyCustomSampler · positive.
    slot_consumers: dict[tuple[str, int], list[str]] = {}
    for s in slots:
        lbl = slot_label.get(s.get("id"), s.get("id"))
        for inp, src in (s.get("input_sources") or {}).items():
            if isinstance(src, str) and src.startswith("core:"):
                try:
                    _, ccid, cidx = src.split(":", 2)
                    slot_consumers.setdefault((ccid, int(cidx)), []).append(f"{lbl} · {inp}")
                except ValueError:
                    pass

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
                           "value": ov.get(inp, ""), "options": _options(t, builtin, cid),
                           "locked": locked})
        for inp, (t, req) in open_by_node.get(cid, {}).items():
            builtin = f"a {t} from your loaders"
            inputs.append({"name": inp, "type": t, "from": "loader", "required": req,
                           "detail": builtin + ("" if req else " (optional)"),
                           "value": ov.get(inp, ""), "options": _options(t, "(auto-wire from loaders)", cid),
                           "locked": False})
        outputs = []
        for i, o in enumerate(node_outputs(nd or {})):
            dests = [f"{d} · {di}" for (oi, d, di) in rev.get(cid, []) if oi == i]
            dests += slot_consumers.get((cid, i), [])
            outputs.append({"name": o["name"], "type": o["type"], "to": dests,
                            "source_id": f"core:{cid}:{i}"})
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


def _reaches_upstream(graph, start, goal):
    """True if `goal` is upstream of `start` — i.e. `start` transitively consumes `goal`'s
    output. Used to decide whether wiring `start`'s output into a node would close a loop."""
    seen = set()
    stack = [start]
    while stack:
        n = stack.pop()
        if n == goal:
            return True
        if n in seen:
            continue
        seen.add(n)
        for v in (graph.get(n, {}).get("inputs", {}) or {}).values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in graph:
                stack.append(v[0])
    return False


def _edge_input(graph, dst, up):
    """The input name on `dst` that consumes `up`'s output (or None)."""
    for inp, val in (graph.get(dst, {}).get("inputs", {}) or {}).items():
        if isinstance(val, list) and len(val) == 2 and val[0] == up:
            return inp
    return None


def _find_cycle(graph):
    """Return a node-id path [a, …, a] forming a dependency cycle, or None. Edges follow the
    'consumes' direction (a node depends on every node feeding its inputs)."""
    color: dict[str, int] = {}          # 0/absent=white, 1=gray (on stack), 2=black
    path: list[str] = []

    def dfs(u):
        color[u] = 1
        path.append(u)
        for val in (graph.get(u, {}).get("inputs", {}) or {}).values():
            if not (isinstance(val, list) and len(val) == 2 and isinstance(val[0], str)):
                continue
            v = val[0]
            if v not in graph:
                continue
            if color.get(v, 0) == 1:                 # back-edge → cycle
                return path[path.index(v):] + [v]
            if color.get(v, 0) == 0:
                hit = dfs(v)
                if hit:
                    return hit
        path.pop()
        color[u] = 2
        return None

    for n in list(graph):
        if color.get(n, 0) == 0:
            hit = dfs(n)
            if hit:
                return hit
    return None


def _break_cycles(graph, report, protected, label):
    """Guarantee the assembled graph is acyclic. A cycle's first droppable edge (an auto-wire or
    a role-default wire — NOT a core-internal link or an explicit user/override wire) is removed
    with a clear note. An all-explicit cycle can't be auto-resolved, so it's reported as blocking
    with the exact loop — far better than ComfyUI's opaque 'Dependency cycle detected'."""
    L = lambda nid: label.get(nid, nid)
    for _ in range(128):                             # bounded: each pass removes >=1 edge
        cyc = _find_cycle(graph)
        if not cyc:
            return
        dropped = False
        for i in range(len(cyc) - 1):
            dst, up = cyc[i], cyc[i + 1]
            inp = _edge_input(graph, dst, up)
            if inp is None or (dst, inp) in protected:
                continue
            del graph[dst]["inputs"][inp]
            report["auto_wired"].append(
                f"Dropped {L(dst)}.{inp} ← {L(up)} to avoid a dependency cycle "
                f"(auto/default wire). Wire it explicitly if you really want it.")
            dropped = True
            break
        if not dropped:
            loop = " → ".join(L(n) for n in cyc)
            msg = (f"Wiring forms a dependency cycle: {loop}. Remove one of these links — "
                   f"ComfyUI can't run a graph that feeds into itself.")
            report["unsatisfied"].append(msg)
            report["blocking"].append(msg)
            return


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
        raw = [p for p in producers.get(t, []) if p[0] != node_id]
        # Never auto-wire a source that already depends on this node — that closes a loop and
        # ComfyUI rejects the whole graph. Drop such candidates so auto-wire stays acyclic.
        cands = [p for p in raw if not _reaches_upstream(graph, p[0], node_id)]
        if len(cands) == 1:
            node["inputs"][inp] = [cands[0][0], cands[0][1]]
            report["auto_wired"].append(f"{L(node_id)}.{inp} <- {L(cands[0][0])} ({t})")
        elif len(cands) > 1:
            msg = f"{L(node_id)}.{inp} ({t}): {len(cands)} possible sources — wire it explicitly."
            report["ambiguous"].append(msg)
            if required:
                report["blocking"].append(msg)
        elif raw:
            msg = (f"{L(node_id)}.{inp} ({t}): the only available source would form a dependency "
                   f"cycle — wire it explicitly to a node upstream of it.")
            report["ambiguous"].append(msg)
            if required:
                report["blocking"].append(msg)
        else:
            msg = f"{L(node_id)}.{inp} ({t}): no node outputs {t} — add one in Models, or this input may not be needed."
            report["unsatisfied"].append(msg)
            if required:
                report["blocking"].append(msg)


def _apply_bypass(graph, slots, slot_node_id, slot_def, report):
    """Drop each bypassed slot's node, rewiring its consumers to whatever already feeds its
    matching-type input (same idea as ComfyUI's node bypass) — only when that mapping is
    unambiguous (exactly one connection_input per output type) AND that input actually has
    something wired to pass through. Otherwise blocks generation with a clear reason rather
    than silently leaving the node active or dropping a consumer's input — a bypass a user
    explicitly asked for must never be silently ignored. Runs after auto-wire so the
    passthrough source is already resolved to a concrete value/link.
    """
    for s in slots:
        if not s.get("bypassed"):
            continue
        sid = slot_node_id.get(s["id"])
        if not sid or sid not in graph:
            continue
        nd = slot_def.get(s["id"]) or {}
        outs = node_outputs(nd)
        by_type: dict[str, list[str]] = {}
        for ci in connection_inputs(nd):
            by_type.setdefault(ci["type"], []).append(ci["name"])
        passthrough = {}
        ok = True
        for i, o in enumerate(outs):
            names = by_type.get(o["type"])
            if not names or len(names) != 1:
                ok = False
                break
            passthrough[i] = graph[sid]["inputs"].get(names[0])
        if not ok:
            # Silently leaving the node active would mean a user who explicitly bypassed it
            # (e.g. to skip an i2v preprocessing node) gets generation output as if they
            # hadn't — with no visible sign why. Block instead: bypass either does what was
            # asked or the run stops with a clear reason, never a silent no-op.
            msg = (
                f"{s.get('node_class')}: bypass needs exactly one input matching each output's "
                f"type to pass through — this node doesn't have one, so it can't be safely "
                f"bypassed. Remove it or rewire it with a single matching input per output type.")
            report["unsatisfied"].append(msg)
            report["blocking"].append(msg)
            continue
        for nid, ndata in graph.items():
            if nid == sid:
                continue
            for inp_name, val in list((ndata.get("inputs") or {}).items()):
                if isinstance(val, list) and len(val) == 2 and val[0] == sid:
                    replacement = passthrough.get(val[1])
                    if replacement is not None:
                        ndata["inputs"][inp_name] = replacement
                    else:
                        # The bypassed node's own matching input was never wired — passing
                        # nothing through would silently drop {nid}.{inp_name} instead of
                        # restoring the original source. Block rather than guess.
                        msg = (
                            f"{s.get('node_class')}: bypass can't pass a value through to "
                            f"{nid}.{inp_name} — its own matching input isn't wired to anything.")
                        report["unsatisfied"].append(msg)
                        report["blocking"].append(msg)
                        del ndata["inputs"][inp_name]
        del graph[sid]
        report["wired"].append(f"{s.get('node_class')} bypassed (pass-through)")
