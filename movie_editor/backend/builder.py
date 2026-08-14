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
from .nodes import (WIDGET_PRIMITIVES, connection_inputs, node_outputs, type_accepts,
                    widget_type_of, _combo_default, _combo_choices)

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


# core internal links: core_id -> {input_name: (src_core_id, output_index)}
CORE_LINKS: dict[str, dict[str, tuple[str, int]]] = {
    "studio":   {"positive_prompt": ("pos", 0), "negative_prompt": ("neg", 0),
                 "refinement_key_input": ("keyloader", 0)},
    "cond":     {"positive": ("studio", 1), "negative": ("studio", 2), "frame_rate": ("fps", 0)},
    "sampler":  {"model": ("studio", 0), "positive": ("cond", 0), "negative": ("cond", 1),
                 "sampler": ("studio", 4), "sigmas": ("studio", 5),
                 # Studio has always emitted a second schedule (low_pass_sigmas, output 7);
                 # the Editor just never wired it. second_pass reads it as pass 2's schedule
                 # and ignores it when empty, so this link costs nothing when the feature is
                 # off. Only the SIGMAS is taken — pass 2 reuses the main sampler object, so
                 # the low pass's sampler-type settings deliberately do not apply.
                 "second_pass_sigmas": ("studio", 7),
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

# A required port another input can stand in for: {(core_id, input): (alternative inputs…)}.
# Studio encodes the prompt through `clip`, but a pre-encoded CONDITIONING wired into
# positive_conditioning replaces that path entirely (_v2_conditioning_source takes whichever
# is present, CLIP first). So a project that brings its own conditioning must not be blocked
# for leaving CLIP unwired — or, with several text encoders installed, for its CLIP being
# ambiguous. The port stays in OPEN_PORTS: it still auto-wires when there IS a single
# obvious encoder, it just no longer holds up generation when the alternative is satisfied.
PORT_ALTERNATIVES: dict[tuple[str, str], tuple[str, ...]] = {
    ("studio", "clip"): ("positive_conditioning",),
}

# core outputs offered as auto-wire producers for slot inputs (image-proc etc.).
CORE_PRODUCERS: list[tuple[str, int, str]] = [  # (core_id, output_index, type)
    ("frames", 0, "INT"),
    ("fps", 0, "FLOAT"),
    ("f2i", 0, "INT"),
]

# ── model families ────────────────────────────────────────────────────────────
# LTXAV above is the baseline. MiniMax H3 keeps the same SHAPE (Studio -> sampler ->
# separate AV -> audio decode -> combine) but three nodes in it do not apply:
#
#   * LTXVConditioning stamps LTX's frame_rate onto the conditioning. H3 has no such
#     conditioning field — its 24 fps is fixed in the model — so Studio wires straight
#     to the sampler.
#   * LTXVConcatAVLatent merges two separately-created empty latents. H3's own
#     EmptyMiniMaxH3LatentAV emits BOTH streams already nested, so there is nothing to
#     concat and the latent slot feeds the sampler directly.
#   * LTXVAudioVAEDecode reads `audio_vae.first_stage_model.output_sample_rate`, which
#     H3's audio VAE does not define — it would raise. Core's VAEDecodeAudio is generic.
#   * LTXFloatToInt only exists to offer an INT copy of the project's frame rate to user
#     slots. H3's frame rate is fixed at 24 by the model, so there is nothing to convert —
#     and keeping it made an LTX node pack a hard requirement of a non-LTX pipeline.
#
# Anything not listed here is shared, so a fix to the core graph reaches both families.
DEFAULT_FAMILY = "ltxav"

FAMILIES: dict[str, dict] = {
    "ltxav": {"label": "LTX2 / LTX2.3 / LTX2.5"},
    "minimax_h3": {
        "label": "MiniMax H3 (Hailuo)",
        "drop": ("cond", "concat", "f2i"),
        "core": {"audiodec": "VAEDecodeAudio"},
        "links": {
            # positive/negative come straight from Studio, with no LTXVConditioning between
            "sampler": {"positive": ("studio", 1), "negative": ("studio", 2)},
            # Audio decodes from the sampler's own AV latent, NOT from a pre-separated audio
            # tensor: ComfyUI's official H3 templates (video_minimax_h3_r2v / _i2v) feed the
            # raw sampler latent to both VAEDecode and VAEDecodeAudio and let each VAE take
            # its own stream, and vae_decode_audio hands `samples["samples"]` straight to
            # vae.decode(). Unbinding first gives that VAE a different object than the
            # reference graph does. `separate` stays in the graph for saveref's video latent.
            "audiodec": {"samples": ("sampler", 0)},
        },
        # (core_id, input, type, required) replacements for the ports the dropped nodes owned
        "open_ports": {
            "drop": (("concat", "audio_latent"), ("audiodec", "audio_vae"), ("studio", "latent")),
            "add": (
                # one node makes the whole AV latent, so it feeds the sampler directly
                ("sampler", "latent_template", "LATENT", True),
                # VAEDecodeAudio names its VAE input `vae`, not `audio_vae`
                ("audiodec", "vae", "VAE", True),
                # optional: only needed to encode AUDIO ref2va references
                ("sampler", "audio_vae", "VAE", False),
                # optional: a MiniMax H3 Image to Video node emits its first/last frame pins
                # on its CONDITIONING output, which this pipeline otherwise drops (the
                # sampler's positive comes from Studio). Wiring it here keeps the pins.
                ("sampler", "h3_keyframes", "CONDITIONING", False),
            ),
        },
    },
}


# Pixel-frame grid and generation frame rate per family. LTX: 8k+1 at the project's fps.
# MiniMax H3: 17k+5 at a fixed 24 fps (both are properties of the model, not settings).
# The frontends snap to the same numbers (PipelineCaps.frameGrid); this is the authority,
# so a project built by an older UI — or by the API directly — still gets a valid graph.
FAMILY_FRAME_GRID: dict[str, dict] = {
    "ltxav": {"step": 8, "base": 1, "fps": None},
    "minimax_h3": {"step": 17, "base": 5, "fps": 24},
}


def family_frames(family: str, frames) -> int:
    """`frames` snapped UP to `family`'s pixel-frame grid (never below one whole step)."""
    grid = FAMILY_FRAME_GRID.get(family) or FAMILY_FRAME_GRID[DEFAULT_FAMILY]
    try:
        n = int(frames)
    except (TypeError, ValueError):
        return grid["step"] + grid["base"]
    n = max(grid["step"] + grid["base"], n)
    while (n - grid["base"]) % grid["step"] != 0:
        n += 1
    return n


def family_frame_rate(family: str, frame_rate):
    """The frame rate the graph must use — the model's own when it has a fixed one."""
    grid = FAMILY_FRAME_GRID.get(family) or FAMILY_FRAME_GRID[DEFAULT_FAMILY]
    return grid["fps"] if grid["fps"] is not None else frame_rate


def family_of(models_config: Optional[dict]) -> str:
    """Which model family this project's graph is built for.

    Explicit, never guessed from a checkpoint filename: the two families need different
    node classes, and a wrong guess produces a graph that fails deep inside ComfyUI
    instead of in the Models panel where it can be fixed.
    """
    fam = str((models_config or {}).get("model_family") or DEFAULT_FAMILY).strip().lower()
    return fam if fam in FAMILIES else DEFAULT_FAMILY


def family_core(family: str) -> tuple[dict, dict, list]:
    """(CORE, CORE_LINKS, OPEN_PORTS) for `family`."""
    spec = FAMILIES.get(family) or {}
    dropped = set(spec.get("drop") or ())

    core = {k: v for k, v in CORE.items() if k not in dropped}
    core.update(spec.get("core") or {})

    links: dict[str, dict] = {}
    for cid, mapping in CORE_LINKS.items():
        if cid in dropped:
            continue
        kept = {inp: src for inp, src in mapping.items() if src[0] not in dropped}
        links[cid] = kept
    for cid, mapping in (spec.get("links") or {}).items():
        links.setdefault(cid, {}).update(mapping)

    ports_spec = spec.get("open_ports") or {}
    drop_ports = set(ports_spec.get("drop") or ())
    ports = [p for p in OPEN_PORTS if (p[0], p[1]) not in drop_ports and p[0] not in dropped]
    ports.extend(ports_spec.get("add") or ())
    return core, links, ports


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
            if isinstance(t, list) or widget_type_of(t, opts) in WIDGET_PRIMITIVES:
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
            wt = widget_type_of(t, opts) if isinstance(t, str) else None
            if isinstance(t, list):
                out[name] = opts.get("default", t[0] if t else None)
            elif wt and "COMBO" in wt.upper():
                # "COMBO" or a V3 dynamic combo (COMFY_DYNAMICCOMBO_V3): emit the selected
                # key string (options may be {"key": ...} dicts), not the option object.
                out[name] = _combo_default(opts)
            elif wt in WIDGET_PRIMITIVES:
                # Always emit a value for every widget — ComfyUI's frontend does, and a
                # required widget with no declared default (e.g. ImageTransform's bboxes)
                # otherwise goes missing and ComfyUI rejects the prompt. Fall back to a
                # type-appropriate empty so generation isn't blocked.
                out[name] = opts.get("default", _WIDGET_EMPTY.get(wt, ""))
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
            elif isinstance(t, str) and "COMBO" in (widget_type_of(t, opts) or "").upper():
                choices = _combo_choices(opts)
                if choices:
                    out[name] = choices
    return out


# ── core widget baseline ──────────────────────────────────────────────────────

def load_reference() -> dict:
    """Baseline widget values for the fixed core nodes, keyed by core id — the tuned
    settings (sigma schedules, sampler config, container format) the editor starts from
    before per-run params are applied.

    Values only: no prompt text, and nothing user-typed. Every text field the run needs
    arrives from the project at generate time, so none of it belongs in a checked-in file.
    A missing/unreadable file just means each core node falls back to its object_info
    defaults.
    """
    try:
        data = json.loads((config.REFERENCE_DIR / "core_widgets.json").read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


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
    # Which model family's core graph to emit. Shadows the module tables for this whole
    # build (including the closures below), so every reference downstream is family-aware.
    family = family_of(models_config)
    CORE, CORE_LINKS, OPEN_PORTS = family_core(family)
    ref_wv = load_reference()
    graph: dict[str, dict] = {}
    # `blocking` is the subset of problems that should stop generation (required inputs).
    # "ignored" is its own bucket, not part of "unsatisfied": it means a value the user SET
    # did not reach the graph, so the run is valid but is not the one they configured. That
    # is the only class worth interrupting them about, and it is surfaced in the editor.
    report: dict[str, list] = {"wired": [], "auto_wired": [], "ambiguous": [], "unsatisfied": [],
                               "ignored": [], "blocking": []}
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
            inputs.update(extract_widgets(nd, ref_wv.get(cid)))
            for inp, (src, idx) in CORE_LINKS.get(cid, {}).items():
                inputs[inp] = [src, idx]
                protected_edges.add((cid, inp))
            graph[cid] = {"class_type": cls, "inputs": inputs}

        # 2. param overrides on the primitives + sampler seed.
        graph["pos"]["inputs"]["value"] = params.get("prompt", "")
        if params.get("negative_prompt") is not None:
            graph["neg"]["inputs"]["value"] = params["negative_prompt"]
        if params.get("num_frames_per_scene") is not None:
            graph["frames"]["inputs"]["value"] = family_frames(family, params["num_frames_per_scene"])
        if params.get("frame_rate") is not None:
            # H3 generates at a fixed 24 fps. The project's rate still drives the container
            # the clip is muxed into, so anything else plays the generated frames at the
            # wrong speed — pin it here rather than render something that looks fast/slow.
            graph["fps"]["inputs"]["value"] = family_frame_rate(family, params["frame_rate"])
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
        # A saved combo selection is checked against the LIVE list ComfyUI validates with:
        # a projector / LoRA / model file that isn't on THIS machine makes ComfyUI reject
        # the whole prompt ("Value not in list"), so nothing queues at all — even when the
        # feature reading it is switched off. Fall back to the node's default and say so,
        # rather than stalling the run on a setting that isn't in play.
        def _live_value(cid: str, name: str, value):
            nd_c = object_info.get((graph.get(cid) or {}).get("class_type"))
            choices = _widget_choices(nd_c).get(name)
            if not choices or isinstance(value, list) or value in choices:
                return value
            fallback = _widget_defaults(nd_c).get(name)
            if fallback not in choices:
                fallback = choices[0]
            report["unsatisfied"].append(
                f"{graph[cid]['class_type']}.{name}: '{value}' is not installed on this "
                f"machine — using '{fallback}' instead.")
            return fallback

        _me_scene_ratings = None
        for k, v in (params.get("studio_inputs") or {}).items():
            if str(k).startswith("_"):
                if k == "_movie_editor_scene_ratings":
                    _me_scene_ratings = v
                continue
            if k not in graph["studio"]["inputs"] or not isinstance(graph["studio"]["inputs"][k], list):
                graph["studio"]["inputs"][k] = _live_value("studio", k, v)

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
                graph["sampler"]["inputs"][k] = _live_value("sampler", k, v)

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

    # Media marked "R" in the bin, by id — one loader node per (media, loader class) so two
    # sockets fed from the same reference share it instead of decoding it twice.
    references = {str(r.get("id")): r for r in (params.get("references") or []) if r.get("id")}

    def _reference_link(ref_id: str, want_type: Optional[str], where: str):
        ref = references.get(ref_id)
        if not ref or not ref.get("filename"):
            report["unsatisfied"].append(
                f"{where}: reference media '{ref_id}' is no longer in the media bin.")
            return None
        found = _reference_loader(object_info, ref.get("kind") or "image", want_type)
        if not found:
            report["unsatisfied"].append(
                f"{where}: no installed node can load a {ref.get('kind')} reference "
                f"into a {want_type} input.")
            return None
        cls, oidx, file_input = found
        nid = f"ref_load_{ref_id}_{cls}"
        graph.setdefault(nid, {"class_type": cls,
                               "inputs": {**_widget_defaults(object_info.get(cls)),
                                          file_input: ref["filename"]}})
        return [nid, oidx]

    # 3b. explicit input sources: user-chosen source for a slot's connection input.
    # Runs before auto-wire so these pre-empt the uniqueness heuristic.
    # source values: "" / "auto" → skip; "timeline" → scene image (media_load);
    #   "ref:<mediaId>" → media marked R in the bin;
    #   "out:<slotId>:<outName>" → another slot's named output;
    #   "core:<coreId>:<outIdx>" → a core primitive output.
    for s in slots:
        sid = slot_node_id[s["id"]]
        nd_s = slot_def[s["id"]]
        for ci_name, source in (s.get("input_sources") or {}).items():
            ci_name = _canonical_input(nd_s, ci_name)
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
            if source.startswith("ref:"):
                want = next((ci["type"] for ci in connection_inputs(nd_s or {})
                             if ci["name"] == ci_name), None)
                link = _reference_link(source[4:], want, f"{s.get('node_class')}.{ci_name}")
                if link:
                    graph[sid]["inputs"][ci_name] = link
                    protected_edges.add((sid, ci_name))
                    report["wired"].append(f"reference -> {sid}.{ci_name}")
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
            key = link.get("editor_key")
            # Text keys come pre-expanded (shortcuts + $variables) in params["expanded"]:
            # the node on the other end of a link encodes the string as-is, so a raw
            # `/trigger` or `$name` would be encoded literally. Studio's own ports are not
            # linked inputs — they still receive the raw text and expand it themselves.
            expanded = params.get("expanded") or {}
            val = expanded[key] if key in expanded else params.get(key)
            # A latent node driven by "Project · Frames" has to receive the SAME number the
            # sampler is given, or the two disagree about the scene length and the run dies
            # on the mismatch. Same for a fixed-rate family's fps.
            if val is not None and key == "num_frames_per_scene":
                val = family_frames(family, val)
            elif val is not None and key == "frame_rate":
                val = family_frame_rate(family, val)
        else:
            val = link.get("value")
        # Every way a link can fail to fire is reported below. It used to fail in silence —
        # the node kept its own widget value, so a project setting the user had just changed
        # simply did not reach the graph, and nothing anywhere said which value won.
        label = str(link.get("name") or link.get("editor_key") or "link")
        if val is None:
            report["ignored"].append(
                f"Linked input '{label}' had no value to send"
                + (f" — the project has no '{link.get('editor_key')}'." if link.get("source") == "editor"
                   else " — set one in Models ▸ Linked inputs.")
                + " Every input it drives kept its own value.")
            continue
        applied = []
        for m in link.get("members") or []:
            slot_id, inp = m.get("slotId"), m.get("input")
            sid = slot_node_id.get(slot_id)
            if not sid or sid not in graph:
                report["ignored"].append(
                    f"Linked input '{label}' drives a node that is no longer in the pipeline, "
                    f"so its '{inp}' kept its own value. Re-pick it in Models ▸ Linked inputs.")
                continue
            # A member naming a widget the node does not have writes a key ComfyUI ignores —
            # the node keeps its default and the link looks like it fired. Renaming an input
            # upstream is enough to cause it.
            nd = slot_def.get(slot_id)
            if nd is not None and inp not in _widget_defaults(nd):
                report["ignored"].append(
                    f"Linked input '{label}': {graph[sid].get('class_type')} has no widget called "
                    f"'{inp}' — nothing was set. Re-pick it in Models ▸ Linked inputs.")
                continue
            graph[sid]["inputs"][inp] = val
            applied.append(f"{graph[sid].get('class_type')}.{inp}")
        if applied:
            report["wired"].append(f"linked '{label}' = {val} -> " + ", ".join(applied))

    port_to_core = _port_index(object_info, core=CORE)

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
        role_defaults = pipeline_wiring._default_wires(family).get(s.get("role") or "", {})
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
                # A wire authored into an autogrow entry names it the same way its own page
                # does; canonicalise so both directions land on the dotted socket id.
                dinput = _canonical_input(object_info.get((graph.get(dnode) or {}).get("class_type")), dinput)
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
        # Inputs the class at this core id actually declares. An override is saved per core
        # ID, not per class, so switching model family leaves the OLD node's input names
        # behind on the NEW node — audiodec's audio_vae (LTXVAudioVAEDecode) survives onto
        # VAEDecodeAudio, whose VAE input is called `vae`. Nothing shows it: the Models panel
        # lists the new node's inputs, so the stale key is invisible right up until ComfyUI
        # is handed a kwarg the node never declared and the run dies inside it
        # ("VAEDecodeAudio.execute() got an unexpected keyword argument 'audio_vae'").
        # Guided mode happened to filter these out; full control applied them.
        core_inputs = {ci["name"] for ci in connection_inputs(object_info.get(CORE.get(cid)) or {})}
        for inp, source in (ovs or {}).items():
            if not source:
                continue
            if core_inputs and inp not in core_inputs:
                report["unsatisfied"].append(
                    f"core override {cid}.{inp}: {CORE.get(cid)} has no input '{inp}' "
                    f"— left over from another model family, ignored.")
                continue
            if locked and (cid, inp) not in pipeline_wiring.open_core_inputs(family):
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
    _autowire(graph, slots, slot_node_id, slot_def, object_info, producers, report, active_slots,
              open_ports=OPEN_PORTS, core=CORE)

    # 5b. bypass: drop a slot's node from the graph and rewire its consumers straight to
    # whatever fed its matching-type input, so the node's effect is skipped without losing
    # its saved configuration (vs. removing the node or zeroing a strength widget every time).
    _apply_bypass(graph, slots, slot_node_id, slot_def, report)

    # Cycle-guard: a default/auto wire that loops (e.g. an upscale node whose IMAGE both feeds the
    # export AND defaults back to Studio · source_image while it consumes the decoded frames) is
    # dropped here; an all-explicit loop is reported precisely. Never hand ComfyUI a cyclic graph.
    _break_cycles(graph, report, protected_edges,
                  _node_labels(slots, slot_node_id, object_info, core=CORE))

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
    CORE, CORE_LINKS, OPEN_PORTS = family_core(family_of(models_config))
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
        for val, lbl in _matching_producers(producers, t):
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


def _port_index(object_info: dict, core: Optional[dict] = None) -> dict[str, tuple[str, str]]:
    """Map pipeline-port id ('Class.input' / 'FunPackStudio.input') -> (core_id, input)."""
    core = core if core is not None else CORE
    idx: dict[str, tuple[str, str]] = {}
    for cid, cls in core.items():
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


# Loaders that turn a marked ("R") media-bin file into a graph output, per media kind.
# Ordered by preference; the first one INSTALLED whose output type the destination socket
# accepts wins, so a video reference can feed either a VIDEO socket or a frames (IMAGE) one
# depending on what the node actually asks for.
REFERENCE_LOADERS: dict[str, list[tuple[str, str, str]]] = {
    "image": [("LoadImage", "IMAGE", "image")],
    "audio": [("LoadAudio", "AUDIO", "audio")],
    "video": [("LoadVideo", "VIDEO", "file"), ("VHS_LoadVideo", "IMAGE", "video")],
}


def _reference_loader(object_info: dict, kind: str, want_type: Optional[str]):
    """(class, output_index, filename_input) for loading a reference of `kind` into a socket
    of `want_type`. None when nothing installed can bridge the two."""
    for cls, out_type, file_input in REFERENCE_LOADERS.get(kind, []):
        nd = object_info.get(cls)
        if not nd:
            continue
        outs = node_outputs(nd)
        for i, o in enumerate(outs):
            if want_type and not type_accepts(want_type, o["type"]):
                continue
            if not want_type and o["type"] != out_type:
                continue
            return cls, i, file_input
    return None


def _canonical_input(node_def: Optional[dict], name: str) -> str:
    """The socket id ComfyUI expects for `name` on this node.

    Autogrow entries are addressed by their dotted path ("ref_images.ref_image_0"). An
    editor config saved before that was known holds the bare template name, and sending
    that reaches the node as an unexpected keyword argument — so map it back rather than
    letting an old config keep failing the run.
    """
    if not node_def or "." in name:
        return name
    for ci in connection_inputs(node_def):
        if ci.get("autogrow") and ci["name"].rsplit(".", 1)[-1] == name:
            return ci["name"]
    return name


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
    # Only what this family's core actually built — offering a producer the graph dropped
    # would auto-wire a slot to a node that is not there.
    for cid, oidx, t in CORE_PRODUCERS:
        if cid not in graph:
            continue
        out.setdefault(t, []).append((cid, oidx))
    for s in slots:
        nd = slot_def[s["id"]]
        for i, o in enumerate(node_outputs(nd or {})):
            out.setdefault(o["type"], []).append((slot_node_id[s["id"]], i))
    return out


def _matching_producers(producers: dict, t: str) -> list:
    """Producers that can feed an input of type `t`. A union type ("IMAGE,MASK") is fed by
    any of its members, so collect across them instead of matching the string exactly."""
    if t in producers:
        return producers[t]
    out = []
    for pt, lst in producers.items():
        if type_accepts(t, pt):
            out.extend(lst)
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


def _node_labels(slots, slot_node_id, object_info, core=None):
    """node_id -> human label ('Role / Name [NodeClass]') for report messages, so users
    don't see opaque ids like 'slot_5u09gm5'."""
    label = {}
    for cid, cls in (core if core is not None else CORE).items():
        label[cid] = f"{(object_info.get(cls) or {}).get('display_name', cls)} [{cls}]"
    for s in slots:
        nid = slot_node_id.get(s["id"])
        if not nid:
            continue
        cls = s.get("node_class") or ""
        nm = s.get("label") or s.get("role_label") or cls or nid
        label[nid] = f"{nm} [{cls}]" if cls and nm != cls else (nm or nid)
    return label


def _autowire(graph, slots, slot_node_id, slot_def, object_info, producers, report,
              active_slots=None, open_ports=None, core=None):
    label = _node_labels(slots, slot_node_id, object_info, core=core)
    L = lambda nid: label.get(nid, nid)

    targets = list(open_ports if open_ports is not None else OPEN_PORTS)  # (core_id, input, type, required)
    for s in slots:  # slot connection inputs (e.g. image-proc vae/image/length)
        if active_slots is not None and s["id"] not in active_slots:
            continue  # inert slot (feeds nothing) — don't auto-wire or block on its inputs
        nd = slot_def[s["id"]]
        for ci in connection_inputs(nd or {}):
            # Autogrow list sockets (ref_image0, ref_image1, …) are explicit-only: a single
            # IMAGE producer would otherwise be auto-wired into EVERY index, silently
            # duplicating one reference ten times. An unwired index is simply absent.
            if ci.get("autogrow"):
                continue
            targets.append((slot_node_id[s["id"]], ci["name"], ci["type"], ci.get("required", False)))

    for node_id, inp, t, required in targets:
        node = graph.get(node_id)
        if not node:
            continue
        if isinstance(node["inputs"].get(inp), list):
            continue  # already wired (explicit/core)
        alt_wired = next((alt for alt in PORT_ALTERNATIVES.get((node_id, inp), ())
                          if isinstance(node["inputs"].get(alt), list)), None)
        if alt_wired:
            # Left deliberately unwired: the alternative the user wired IS the source now, and
            # auto-wiring this port would take precedence over it and silently win.
            report["auto_wired"].append(
                f"{L(node_id)}.{inp} ({t}) left unwired — {L(node_id)}.{alt_wired} is wired and "
                f"takes over. Not required.")
            continue
        raw = [p for p in _matching_producers(producers, t) if p[0] != node_id]
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
    # Every node going away, known up front. A node that only feeds OTHER bypassed nodes
    # needs no pass-through for that output, because once they are all gone nothing reads
    # it. Judging each one against a graph that still holds its doomed siblings refused
    # perfectly sound bypasses — and refused every group bypass of a connected run of nodes.
    leaving = {slot_node_id.get(s["id"]) for s in slots if s.get("bypassed")}
    leaving.discard(None)

    for s in slots:
        if not s.get("bypassed"):
            continue
        sid = slot_node_id.get(s["id"])
        if not sid or sid not in graph:
            continue
        nd = slot_def.get(s["id"]) or {}
        outs = node_outputs(nd)
        cis = connection_inputs(nd)
        # Only the outputs something in the graph actually CONSUMES need a passthrough. A
        # node can emit an output nothing here reads — LTXICLoRALoaderModelOnly returns a
        # FLOAT (latent_downscale_factor) alongside its MODEL, and the editor's graph wires
        # only the MODEL — and demanding a matching input for an output that feeds nothing
        # would refuse a bypass that is completely unambiguous for every link that exists.
        consumed = set()
        for nid, ndata in graph.items():
            if nid == sid or nid in leaving:
                continue
            for val in (ndata.get("inputs") or {}).values():
                if isinstance(val, list) and len(val) == 2 and val[0] == sid:
                    consumed.add(val[1])
        passthrough = {}
        blocked = None
        for i, o in enumerate(outs):
            # A union-typed input ("IMAGE,MASK") can carry any of its members through.
            names = [ci["name"] for ci in cis if type_accepts(ci["type"], o["type"])]
            if i in consumed and len(names) != 1:
                blocked = (o, names)
                break
            # Resolved even for outputs only OTHER leaving nodes read, so a chain of
            # bypasses collapses to the surviving producer instead of to a deleted one.
            if len(names) == 1:
                passthrough[i] = graph[sid]["inputs"].get(names[0])
        if blocked is not None:
            # Silently leaving the node active would mean a user who explicitly bypassed it
            # (e.g. to skip an i2v preprocessing node) gets generation output as if they
            # hadn't — with no visible sign why. Block instead: bypass either does what was
            # asked or the run stops with a clear reason, never a silent no-op.
            _o, _names = blocked
            msg = (
                f"{s.get('node_class')}: bypass needs exactly one input matching each output's "
                f"type to pass through, and its '{_o['name']}' output ({_o['type']}) has "
                f"{'no matching input' if not _names else 'more than one'} — so it can't be "
                f"safely bypassed. Remove it, give it a single matching input for that type, "
                f"or leave that output unconnected.")
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
                    elif nid in leaving:
                        # The consumer is being bypassed too, so there is nothing to repair
                        # and nothing to warn about — it will not be in the graph either.
                        del ndata["inputs"][inp_name]
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
