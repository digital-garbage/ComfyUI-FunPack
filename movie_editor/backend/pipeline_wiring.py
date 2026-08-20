"""Built-in pipeline wiring rules: guided (locked) vs full-control (manual) mode.

When the built-in FunPack core is enabled and ``full_control`` is false, loader outputs
may only wire to their designated core ports. Internal core links (Studio -> Conditioning
-> Sampler -> decode) are never user-overridable. Toggle ``full_control`` to restore the
legacy free rewiring behaviour.
"""
from __future__ import annotations

import json

from typing import Any, Optional

# port id (NodeClass.input) allowed for each model role + output type.
# output_name: None = any output socket of that type on the slot node.
ROLE_WIRE_TARGETS: dict[str, list[tuple[str, Optional[str], str]]] = {
    "unet": [("MODEL", None, "FunPackStudio.model")],
    "clip": [("CLIP", None, "FunPackStudio.clip")],
    "lora": [("MODEL", None, "FunPackStudio.model"), ("CLIP", None, "FunPackStudio.clip")],
    "video_vae": [("VAE", None, "FunPackLTXAVSceneChainSampler.vae")],
    "audio_vae": [("VAE", None, "LTXVAudioVAEDecode.audio_vae")],
    "audio_encoder": [("LATENT", None, "LTXVConcatAVLatent.audio_latent")],
    "empty_latent": [("LATENT", None, "FunPackStudio.latent")],
    "video_latent": [("LATENT", None, "FunPackStudio.latent")],
    "image_processing": [
        ("IMAGE", None, "FunPackStudio.source_image"),
        ("LATENT", None, "FunPackStudio.latent"),
    ],
}

# MODEL / CLIP / LATENT / IMAGE may pass through patcher or encode nodes via node:* wires;
# these are the built-in ports they may terminate on in guided mode.
TYPE_CHAIN_TERMINALS: dict[str, list[str]] = {
    "MODEL": ["FunPackStudio.model"],
    "CLIP": ["FunPackStudio.clip"],
    "LATENT": ["FunPackStudio.latent"],
    "IMAGE": ["FunPackStudio.source_image"],
}

# Core-internal ports: shown in the full port list but not user-wirable in guided mode
# (another core node feeds them automatically).
GUIDED_HIDDEN_PORTS: frozenset[str] = frozenset({
    "LTXVConcatAVLatent.video_latent",  # Studio output 12 -> concat (CORE_LINKS)
})

DEFAULT_WIRES_BY_ROLE: dict[str, dict[str, str]] = {
    "unet": {"MODEL": "port:FunPackStudio.model"},
    "clip": {"CLIP": "port:FunPackStudio.clip"},
    "video_vae": {"VAE": "port:FunPackLTXAVSceneChainSampler.vae"},
    "audio_vae": {"VAE": "port:LTXVAudioVAEDecode.audio_vae"},
    "audio_encoder": {"LATENT": "port:LTXVConcatAVLatent.audio_latent"},
    "empty_latent": {"LATENT": "port:FunPackStudio.latent"},
    "video_latent": {"LATENT": "port:FunPackStudio.latent"},
    "image_processing": {"IMAGE": "port:FunPackStudio.source_image"},
}

DEFAULT_INPUT_SOURCES_BY_ROLE: dict[str, dict[str, str]] = {
    "image_processing": {"image": "timeline"},
}

# Human labels for error messages / UI hints.
PORT_LABELS: dict[str, str] = {
    "FunPackStudio.model": "Studio · model",
    "FunPackStudio.clip": "Studio · clip",
    "FunPackStudio.latent": "Studio · latent (forwards to Concat AV · video_latent)",
    "FunPackStudio.source_image": "Studio · source_image (Img2Video anchor)",
    "FunPackLTXAVSceneChainSampler.vae": "Chain Sampler · vae",
    "LTXVConcatAVLatent.audio_latent": "Concat AV Latent · audio_latent",
    "LTXVAudioVAEDecode.audio_vae": "Audio VAE Decode · audio_vae",
}

# Built-in core inputs fed by user loaders (not fixed internal core links).
# core_overrides for these apply even in guided mode; internal links stay fixed.
OPEN_CORE_INPUTS: frozenset[tuple[str, str]] = frozenset({
    ("studio", "model"),
    ("studio", "clip"),
    ("studio", "source_image"),
    ("studio", "latent"),
    ("sampler", "vae"),
    ("concat", "audio_latent"),
    ("audiodec", "audio_vae"),
})

PORT_TO_OPEN_CORE: dict[str, tuple[str, str]] = {
    "FunPackStudio.model": ("studio", "model"),
    "FunPackStudio.clip": ("studio", "clip"),
    "FunPackStudio.source_image": ("studio", "source_image"),
    "FunPackStudio.latent": ("studio", "latent"),
    "FunPackLTXAVSceneChainSampler.vae": ("sampler", "vae"),
    "LTXVConcatAVLatent.audio_latent": ("concat", "audio_latent"),
    "LTXVAudioVAEDecode.audio_vae": ("audiodec", "audio_vae"),
}


# ── model families ────────────────────────────────────────────────────────────
# The tables above describe the LTXAV core. MiniMax H3 drops LTXVConditioning and
# LTXVConcatAVLatent and swaps the audio decode (see builder.FAMILIES), which moves
# three wirable ports onto different nodes:
#
#   * the empty latent no longer goes through Studio into Concat — H3's own
#     EmptyMiniMaxH3LatentAV emits both streams, so it feeds the sampler directly;
#   * the audio VAE lands on core's VAEDecodeAudio.vae, and may ALSO feed the sampler
#     to encode audio ref2va references;
#   * there is no audio_encoder step at all.
#
# Everything not listed is inherited, so a rule added for LTXAV reaches H3 too.
_H3_LATENT_PORT = "FunPackLTXAVSceneChainSampler.latent_template"
# A MiniMax H3 Image to Video node in the latent role also emits CONDITIONING carrying its
# first_frame / last_frame pins. Studio owns the sampler's positive, so that conditioning has
# nowhere to go and the image is silently lost; this port is where the pins are salvaged.
_H3_KEYFRAME_PORT = "FunPackLTXAVSceneChainSampler.h3_keyframes"

FAMILY_WIRING: dict[str, dict] = {
    "ltxav": {},
    "minimax_h3": {
        "role_targets": {
            "audio_vae": [("VAE", None, "VAEDecodeAudio.vae"),
                          ("VAE", None, "FunPackLTXAVSceneChainSampler.audio_vae")],
            "audio_encoder": [],
            "empty_latent": [("LATENT", None, _H3_LATENT_PORT),
                             ("CONDITIONING", None, _H3_KEYFRAME_PORT)],
            "video_latent": [("LATENT", None, _H3_LATENT_PORT),
                             ("CONDITIONING", None, _H3_KEYFRAME_PORT)],
            "image_processing": [("IMAGE", None, "FunPackStudio.source_image")],
        },
        "type_chain_terminals": {"LATENT": [_H3_LATENT_PORT],
                                 "CONDITIONING": [_H3_KEYFRAME_PORT]},
        "default_wires": {
            "audio_vae": {"VAE": "port:VAEDecodeAudio.vae"},
            "audio_encoder": {},
            # both outputs of the H3 latent node are wired by default: the AV latent, and the
            # keyframe pins that come with it when the node has a first/last frame image
            "empty_latent": {"LATENT": "port:" + _H3_LATENT_PORT,
                             "CONDITIONING": "port:" + _H3_KEYFRAME_PORT},
            "video_latent": {"LATENT": "port:" + _H3_LATENT_PORT,
                             "CONDITIONING": "port:" + _H3_KEYFRAME_PORT},
        },
        "port_labels": {
            _H3_LATENT_PORT: "Chain Sampler · latent_template (H3 AV latent)",
            _H3_KEYFRAME_PORT: "Chain Sampler · h3_keyframes (first/last frame pins)",
            "VAEDecodeAudio.vae": "VAE Decode Audio · vae",
            "FunPackLTXAVSceneChainSampler.audio_vae": "Chain Sampler · audio_vae (ref2va audio)",
        },
        "guided_hidden": (),   # nothing to hide: Concat is not in this family's core
        "open_core": {
            "FunPackStudio.model": ("studio", "model"),
            "FunPackStudio.clip": ("studio", "clip"),
            "FunPackStudio.source_image": ("studio", "source_image"),
            "FunPackLTXAVSceneChainSampler.vae": ("sampler", "vae"),
            _H3_LATENT_PORT: ("sampler", "latent_template"),
            _H3_KEYFRAME_PORT: ("sampler", "h3_keyframes"),
            "FunPackLTXAVSceneChainSampler.audio_vae": ("sampler", "audio_vae"),
            "VAEDecodeAudio.vae": ("audiodec", "vae"),
        },
    },
}

DEFAULT_FAMILY = "ltxav"


def family_of(models: Any) -> str:
    fam = str(models_dict(models).get("model_family") or DEFAULT_FAMILY).strip().lower()
    return fam if fam in FAMILY_WIRING else DEFAULT_FAMILY


def _family_core_classes(family: str) -> frozenset:
    """The node classes this family's fixed core actually contains.

    Imported lazily: builder imports THIS module, so a module-level import would be a cycle.
    """
    try:
        from .builder import family_core
        core, _links, _ports = family_core(family)
        return frozenset(core.values())
    except Exception:
        return frozenset()


def _port_exists_in_core(port: str, core_classes: frozenset) -> bool:
    """Is `NodeClass.input` a port on a node this family's core has? Unknown -> allow."""
    if not core_classes or "." not in str(port):
        return True
    return str(port).split(".", 1)[0] in core_classes


def _role_targets(family: str) -> dict:
    out = {k: list(v) for k, v in ROLE_WIRE_TARGETS.items()}
    out.update({k: list(v) for k, v in (FAMILY_WIRING.get(family, {}).get("role_targets") or {}).items()})
    # Drop targets on nodes this family's core does not build. The inherited LTXAV rules
    # point audio_vae at LTXVAudioVAEDecode, which H3 drops entirely — so a project whose
    # family was read wrongly (or a family that simply forgets to override a role) offered a
    # wire to a node that is not in the graph. Filtering here makes that impossible rather
    # than merely fixed: a role can only ever target a port that exists.
    core_classes = _family_core_classes(family)
    return {role: [t for t in targets if _port_exists_in_core(t[2], core_classes)]
            for role, targets in out.items()}


def _chain_terminals(family: str) -> dict:
    out = {k: list(v) for k, v in TYPE_CHAIN_TERMINALS.items()}
    out.update({k: list(v) for k, v in (FAMILY_WIRING.get(family, {}).get("type_chain_terminals") or {}).items()})
    return out


def _type_fallback_ports(family: str) -> dict[str, list[str]]:
    """Ports an output of a given type may reach when its own role has no rule for that type.

    Guided wiring exists to keep the core's INTERNAL links fixed, not to decide which node is
    allowed to fill an open socket. Restricting to role rules did the latter: a node added
    through "Any node…" has role ``custom``, which no table mentions, and the chain terminals
    cover only MODEL/CLIP/LATENT/IMAGE — so a custom VAE loader could not be wired into the
    pipeline at all, and an already-saved wire read back as "(not allowed)".

    So the fallback is every port some role can reach with this type, minus the hidden
    internal ones. Roles that DO have a rule keep it: audio_encoder's LATENT still has to
    land on Concat · audio_latent and nowhere else.
    """
    out: dict[str, list[str]] = {}

    def add(t, port):
        lst = out.setdefault(t, [])
        if port not in lst:
            lst.append(port)

    for rules in _role_targets(family).values():
        for t, _name, port in rules:
            add(t, port)
    for t, ports in _chain_terminals(family).items():
        for port in ports:
            add(t, port)
    hidden = _hidden_ports(family)
    return {t: [p for p in ports if p not in hidden] for t, ports in out.items()}


def _default_wires(family: str) -> dict:
    out = {k: dict(v) for k, v in DEFAULT_WIRES_BY_ROLE.items()}
    out.update({k: dict(v) for k, v in (FAMILY_WIRING.get(family, {}).get("default_wires") or {}).items()})
    # Same guard as _role_targets: never auto-wire a new loader to a port on a node this
    # family does not build. A default wire is applied without the user asking, so a stale
    # one is the most likely way a phantom port reaches a saved project.
    core_classes = _family_core_classes(family)
    return {
        role: {t: w for t, w in wires.items()
               if not (isinstance(w, str) and w.startswith("port:"))
               or _port_exists_in_core(w[len("port:"):], core_classes)}
        for role, wires in out.items()
    }


def _port_labels(family: str) -> dict:
    out = dict(PORT_LABELS)
    out.update(FAMILY_WIRING.get(family, {}).get("port_labels") or {})
    return out


def _hidden_ports(family: str) -> frozenset:
    spec = FAMILY_WIRING.get(family, {})
    return frozenset(spec["guided_hidden"]) if "guided_hidden" in spec else GUIDED_HIDDEN_PORTS


def _open_core(family: str) -> dict:
    spec = FAMILY_WIRING.get(family, {}).get("open_core")
    return dict(spec) if spec else dict(PORT_TO_OPEN_CORE)


def open_core_inputs(family: str) -> frozenset:
    return frozenset(_open_core(family).values())


def models_dict(models: Any) -> dict:
    return models if isinstance(models, dict) else {}


def uses_builtin_core(models: Any) -> bool:
    return not bool(models_dict(models).get("disable_core"))


def wiring_locked(models: Any) -> bool:
    """Guided wiring: built-in core on and full_control off."""
    m = models_dict(models)
    return uses_builtin_core(m) and not bool(m.get("full_control"))


def allowed_port_ids(role: str, out_type: str, out_name: Optional[str] = None,
                     family: str = DEFAULT_FAMILY) -> list[str]:
    role_targets = _role_targets(family)
    out: list[str] = []
    role_rules = role_targets.get(role or "", [])
    for t, name, port in role_rules:
        if t != out_type:
            continue
        if name is not None and out_name is not None and name != out_name:
            continue
        out.append(port)
    # Only when the role has no rule for this type at all (an unknown or custom role, or a
    # role that simply says nothing about this output). A role that DOES have one keeps it:
    # audio_encoder LATENT stays on Concat · audio_latent, not Studio · latent.
    has_explicit = any(t == out_type for t, _, _ in role_rules)
    if not has_explicit:
        for port in _type_fallback_ports(family).get(out_type, []):
            if port not in out:
                out.append(port)
    return out


def port_label(port_id: str, family: str = DEFAULT_FAMILY) -> str:
    return _port_labels(family).get(port_id, port_id.replace(".", " · "))


def validate_port_wire(
    *,
    role: str,
    out_type: str,
    out_name: str,
    target: str,
    models: Any,
) -> Optional[str]:
    """Return an error string if the wire is forbidden in guided mode."""
    if not wiring_locked(models):
        return None
    if not target or target in ("global:video", "global:audio"):
        return None
    if target.startswith("node:"):
        return None  # slot-to-slot wiring stays free
    if not target.startswith("port:"):
        return f"Unknown wire target '{target}'."
    family = family_of(models)
    port_id = target[5:]
    if port_id in _hidden_ports(family):
        if port_id == "LTXVConcatAVLatent.video_latent":
            return (
                f"{out_name} ({out_type}) cannot wire directly to Concat AV Latent · video_latent "
                f"in guided mode — wire to Studio · latent instead; the built-in pipeline "
                f"forwards Studio output to Concat automatically."
            )
        return (
            f"{out_name} ({out_type}) cannot wire to {port_label(port_id, family)} in guided mode "
            f"(internal core link). Enable Full control to override."
        )
    # allowed_port_ids falls back to every port of this type for a role with no rule, which
    # is what a custom or legacy slot has — no special case needed here any more.
    allowed = allowed_port_ids(role, out_type, out_name, family=family)
    if not allowed:
        return (
            f"{out_name or out_type} from role '{role}' cannot wire into the built-in pipeline "
            f"in guided mode — enable Full control in Models to wire it manually."
        )
    if port_id not in allowed:
        labels = ", ".join(port_label(p, family) for p in allowed)
        return (
            f"{out_name} ({out_type}) may only wire to {labels} in guided mode "
            f"(not {port_label(port_id, family)}). Enable Full control to wire freely."
        )
    return None


def validate_models_wiring(models: Any) -> list[str]:
    """Collect blocking wiring errors for all slots."""
    m = models_dict(models)
    if not wiring_locked(m):
        return []
    fam = family_of(m)
    errors: list[str] = []
    port_owners: dict[str, str] = {}
    for slot in m.get("slots") or []:
        role = slot.get("role") or "custom"
        sid = slot.get("id") or "?"
        label = slot.get("label") or slot.get("node_class") or sid
        wires = slot.get("wires") or {}
        for out_name, raw in wires.items():
            targets = raw if isinstance(raw, list) else ([raw] if raw else [])
            out_type = _infer_output_type(slot, out_name)
            for t in targets:
                if not t:
                    continue
                err = validate_port_wire(role=role, out_type=out_type, out_name=out_name,
                                         target=t, models=m)
                if err:
                    errors.append(f"{label}: {err}")
                    continue
                if t.startswith("port:"):
                    prev = port_owners.get(t[5:])
                    if prev:
                        errors.append(
                            f"{port_label(t[5:], fam)} is already wired from {prev} — "
                            f"only one source per built-in input in guided mode.")
                    else:
                        port_owners[t[5:]] = label
    return errors




# ── the default pipeline ──────────────────────────────────────────────────────
# A fresh project starts with FunPack's own loaders, already wired to the core ports it
# needs, so setting up a model is choosing files and nothing else. Every one of these is an
# ordinary slot afterwards: rename it, swap the node class, rewire it, delete it.
#
# (role, node class, label, input sources). Outputs are read from the live schema and wired
# through DEFAULT_WIRES_BY_ROLE, the same table used when a node is added by hand.
FUNPACK_DEFAULT_LOADERS: list[tuple[str, str, str, dict]] = [
    ("unet",      "FunPackDiffusionModelLoader", "Diffusion model", {}),
    # Seeded empty, and empty means pass-through: the one node every LoRA needs is already
    # in the chain, so using one is picking a file rather than adding and rewiring a node.
    ("lora",      "FunPackLoraLoader",           "LoRAs",           {"model": "out:fp_unet:MODEL"}),
    ("clip",      "FunPackCLIPLoader",           "Text encoder",    {}),
    ("video_vae", "FunPackVAELoader",            "Video VAE",       {}),
    ("audio_vae", "FunPackVAELoader",            "Audio VAE",       {}),
]

# Wires for a seeded slot whose role has no default, or wants less than the role allows.
# The LoRA loader hands CLIP straight back untouched, so wiring it there would only add a
# hop — and a second source for the port the CLIP loader already feeds.
DEFAULT_SEED_WIRES: dict[str, dict[str, str]] = {
    "lora": {"MODEL": "port:FunPackStudio.model"},
}

# The rest of what a family needs before it can generate. Not loaders and not FunPack's, but
# leaving them out would mean "the pipeline sets itself up" still ended in a required slot
# the user has to find, add and wire by hand — which is the thing this is here to remove.
# Sources name seeded slots by their fixed ids, and the project's own frames / fps
# primitives, so the node follows the project instead of holding a second copy of its length.
FAMILY_DEFAULT_EXTRAS: dict[str, list[tuple[str, str, str, dict]]] = {
    "ltxav": [
        ("audio_encoder", "LTXVEmptyLatentAudio", "Audio latent",
         {"audio_vae": "out:fp_audio_vae:VAE",
          "frames_number": "core:frames:0",
          "frame_rate": "core:fps:0"}),
    ],
    "minimax_h3": [
        # H3 emits both streams from one node, so this IS the latent the sampler starts from.
        ("empty_latent", "EmptyMiniMaxH3LatentAV", "AV latent", {"length": "core:frames:0"}),
    ],
}


def declared_widget_defaults(node_def: Optional[dict]) -> dict:
    """A node's DECLARED widget defaults — no first-choice fallback.

    `nodes.widget_inputs` fills a combo's default with its first option, which is right when
    someone adds a node by hand and wrong here: it would pre-select an arbitrary model file
    and make an unconfigured loader look configured. Only what the node actually declares is
    seeded, so the file pickers stay empty and say so.
    """
    out = {}
    inp = (node_def or {}).get("input") or {}
    for group in ("required", "optional"):
        for name, spec in (inp.get(group) or {}).items():
            if not isinstance(spec, (list, tuple)) or not spec:
                continue
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if opts.get("forceInput") or "default" not in opts:
                continue
            out[name] = opts["default"]
    return out


def default_pipeline_slots(family: str = DEFAULT_FAMILY,
                           object_info: Optional[dict] = None) -> list[dict]:
    """Slot recipes for a project that has none yet, wired for `family`.

    A loader with nothing to feed in this family is skipped rather than seeded as a node
    wired to nowhere.
    """
    from .nodes import node_outputs  # lazy: nodes imports this module's siblings

    family = family if family in FAMILY_WIRING else DEFAULT_FAMILY
    wires_by_role = _default_wires(family)
    out = []
    for role, cls, label, sources in (FUNPACK_DEFAULT_LOADERS
                                      + FAMILY_DEFAULT_EXTRAS.get(family, [])):
        node_def = (object_info or {}).get(cls)
        if node_def is None:
            continue        # this ComfyUI does not have the node
        role_wires = DEFAULT_SEED_WIRES.get(role) or wires_by_role.get(role) or {}
        # By output NAME first, then by type — the same rule the add-a-node path uses, and
        # necessary because the tables are keyed by type while a node names its own outputs.
        wires = {}
        for o in node_outputs(node_def):
            target = role_wires.get(o["name"]) or role_wires.get(o["type"])
            if target:
                wires[o["name"]] = [target]
        if not wires:
            continue        # nothing to feed in this family: not a node, just clutter
        out.append({"id": f"fp_{role}", "role": role, "role_label": label,
                    "node_class": cls, "inputs": declared_widget_defaults(node_def),
                    "input_sources": dict(sources), "wires": wires,
                    "outputs": {o["name"]: o["type"] for o in node_outputs(node_def)}})
    for slot in out:
        _claim_upstream_ports(slot, out)
    for slot in out:
        slot.pop("outputs", None)
    return out


def _claim_upstream_ports(slot: dict, slots: list[dict]) -> None:
    """Move a port wire onto a seeded pass-through that sits in front of it.

    A slot that consumes type T from another seeded slot AND emits T is a link in that
    chain, not a second branch off it: the producer must feed the pass-through, or both
    would arrive at the same core port. Slots that merely consume T (an audio latent taking
    a VAE) emit no T and so claim nothing.
    """
    emits = set((slot.get("outputs") or {}).values())
    by_id = {s["id"]: s for s in slots}
    for input_name, source in (slot.get("input_sources") or {}).items():
        parts = str(source).split(":")
        if len(parts) != 3 or parts[0] != "out":
            continue
        producer = by_id.get(parts[1])
        if producer is None or producer is slot:
            continue
        out_type = (producer.get("outputs") or {}).get(parts[2])
        if out_type is None or out_type not in emits:
            continue
        producer["wires"][parts[2]] = [f"node:{slot['id']}:{input_name}"]


def seed_default_pipeline(models: dict, object_info: Optional[dict] = None) -> dict:
    """Give a pipeline config that has never been set up FunPack's own loaders.

    Recorded with `defaults_seeded` so it happens exactly once: a pipeline someone emptied
    on purpose stays empty, and so does one built around an imported workflow.
    """
    if not isinstance(models, dict) or models.get("defaults_seeded"):
        return models
    if models.get("slots") or models.get("workflow_import") or models.get("disable_core"):
        models["defaults_seeded"] = True     # nothing to seed, and never seed it later
        return models
    if not isinstance(object_info, dict):
        # No live schema means no way to tell whether FunPack's loaders are even installed,
        # and no widget defaults to seed them with. Leave it unmarked and try again later.
        return models
    models["slots"] = default_pipeline_slots(family_of(models), object_info)
    models["defaults_seeded"] = True
    return models


# Where a seeded loader's file lives, and which widget names on ANOTHER pipeline's node
# hold the same file. An imported workflow uses whatever loader its author picked, so the
# only reliable bridge between the two is the file name itself.
SEEDED_FILE_INPUTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "FunPackDiffusionModelLoader": ("model_name", ("model_name", "unet_name", "ckpt_name")),
    "FunPackVAELoader": ("vae_name", ("vae_name",)),
}


def _combo_choices(node_def: Optional[dict], input_name: str) -> Optional[list]:
    inp = (node_def or {}).get("input") or {}
    for group in ("required", "optional"):
        spec = (inp.get(group) or {}).get(input_name)
        if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], list):
            return spec[0]
    return None


def is_workflow_import(models: Optional[dict]) -> bool:
    """A pipeline built from someone's exported graph rather than from FunPack's loaders."""
    models = models if isinstance(models, dict) else {}
    return bool(models.get("workflow_import") or models.get("disable_core"))


def _file_input_names(cls: str) -> tuple[str, ...]:
    if cls == "FunPackCLIPLoader":
        return ("clip_name",)          # matched by prefix: clip_name, clip_name1, clip_name2…
    return SEEDED_FILE_INPUTS.get(cls, (None, ()))[1]


def _slot_picks(slot: dict, cls: str) -> list[str]:
    """The file names a slot has picked, in widget-name order."""
    names = _file_input_names(cls)
    out = []
    for key in sorted((slot.get("inputs") or {}).keys()):
        value = (slot.get("inputs") or {})[key]
        if not isinstance(value, str) or not value:
            continue
        if key.startswith(names[0]) if cls == "FunPackCLIPLoader" else key in names:
            out.append(value)
    return out


def _donor_for(slot: dict, source_slots: list[dict]) -> Optional[dict]:
    """The old slot that held the same file as this seeded loader.

    Role first — but an imported workflow records every node as `custom`, so the fallbacks
    matter more than the happy path: a node wired to the same core port is the same link in
    the pipeline, and failing that, a file input only one node in the whole graph has can
    only be that one. Anything ambiguous is left for the user rather than guessed at.
    """
    cls = slot.get("node_class")
    role = slot.get("role")
    candidates = [d for d in source_slots if _slot_picks(d, cls)]
    if not candidates:
        return None
    by_role = [d for d in candidates if role and d.get("role") == role]
    if len(by_role) == 1:
        return by_role[0]
    targets = {t for wires in (slot.get("wires") or {}).values() for t in (wires or [])}
    if targets:
        by_port = [d for d in candidates
                   if targets & {t for w in (d.get("wires") or {}).values() for t in (w or [])}]
        if len(by_port) == 1:
            return by_port[0]
    return candidates[0] if len(candidates) == 1 else None


def carry_over_model_files(slots: list[dict], source: Optional[dict],
                           object_info: Optional[dict] = None) -> int:
    """Move the FILES an old pipeline had picked onto freshly seeded loaders.

    The point of a new project inheriting anything at all is not having to find the same
    files again. A name the installed node does not offer is skipped rather than written,
    so a stale pick cannot make an empty loader look configured.
    """
    source_slots = (source or {}).get("slots") or []
    carried = 0
    for slot in slots:
        cls = slot.get("node_class")
        if cls != "FunPackCLIPLoader" and cls not in SEEDED_FILE_INPUTS:
            continue
        donor = _donor_for(slot, source_slots)
        if donor is None:
            continue
        picks = _slot_picks(donor, cls)
        node_def = (object_info or {}).get(cls)
        if cls == "FunPackCLIPLoader":
            choices = _combo_choices(node_def, "clip_list") or []
            rows = [{"clip_name": n} for n in picks if not choices or n in choices]
            if rows:
                slot.setdefault("inputs", {})["clip_list"] = json.dumps(rows)
                carried += len(rows)
            continue
        target = SEEDED_FILE_INPUTS[cls][0]
        choices = _combo_choices(node_def, target)
        pick = next((n for n in picks if not choices or n in choices), None)
        if pick:
            slot.setdefault("inputs", {})[target] = pick
            carried += 1
    return carried


def is_seeded_pipeline(models: Optional[dict]) -> bool:
    """True when the pipeline is still exactly what seeding produced — no node the user
    added, so rebuilding it for another family throws away nothing of theirs."""
    slots = (models or {}).get("slots") or []
    seeded_ids = {f"fp_{role}" for role, _c, _l, _s in FUNPACK_DEFAULT_LOADERS}
    for extras in FAMILY_DEFAULT_EXTRAS.values():
        seeded_ids |= {f"fp_{role}" for role, _c, _l, _s in extras}
    return bool(slots) and all(s.get("id") in seeded_ids for s in slots)


def reseed_for_family(models: dict, object_info: Optional[dict] = None) -> bool:
    """Rebuild an untouched seeded pipeline for the family it now says it is.

    Answering "MiniMax H3" and being handed LTX's nodes is the setup answering back with
    something the user did not choose. Only ever runs on a pipeline nobody has edited.
    """
    if not isinstance(object_info, dict) or not is_seeded_pipeline(models):
        return False
    slots = default_pipeline_slots(family_of(models), object_info)
    if not slots:
        return False
    carry_over_model_files(slots, models, object_info)
    models["slots"] = slots
    return True


def new_project_models(glob: Optional[dict], object_info: Optional[dict] = None) -> dict:
    """The pipeline a NEW project starts with.

    A configured pipeline IS the template — picking the same files in every project is the
    thing the global default exists to prevent. An IMPORTED WORKFLOW is not: it is one
    project's graph, and copying it hands every later project a pile of third-party loaders
    to understand instead of FunPack's four. The files it picked come along; the graph does
    not.
    """
    glob = glob if isinstance(glob, dict) else {}
    if not is_workflow_import(glob):
        return dict(glob)
    carried = {k: glob[k] for k in ("model_family", "full_control") if glob.get(k)}
    carried["slots"] = []
    seed_default_pipeline(carried, object_info)
    if carried.get("slots"):
        carry_over_model_files(carried["slots"], glob, object_info)
    return carried


def wiring_rules_payload(family: str = DEFAULT_FAMILY) -> dict[str, Any]:
    """Static rules for the Models UI, for one model family."""
    family = family if family in FAMILY_WIRING else DEFAULT_FAMILY
    labels = _port_labels(family)
    role_targets = {}
    for role, rules in _role_targets(family).items():
        role_targets[role] = [
            {"type": t, "output_name": n, "port": p,
             "label": labels.get(p, p.replace(".", " · "))}
            for t, n, p in rules
        ]
    return {
        "family": family,
        "role_targets": role_targets,
        "type_chain_terminals": _chain_terminals(family),
        # What a role with no rule for this output type may reach. The panel has to filter
        # destinations exactly as validate_port_wire does, or it hides a wire the builder
        # would have accepted — which is how a saved wire came back "(not allowed)".
        "type_fallback_ports": _type_fallback_ports(family),
        "guided_hidden_ports": sorted(_hidden_ports(family)),
        "default_wires": _default_wires(family),
        "default_input_sources": DEFAULT_INPUT_SOURCES_BY_ROLE,
        "port_labels": labels,
        "open_core_ports": [
            {"core_id": cid, "input": inp, "port": port}
            for port, (cid, inp) in _open_core(family).items()
        ],
    }


def _infer_output_type(slot: dict, out_name: str) -> str:
    """Best-effort type lookup; frontend validates with full spec."""
    role = slot.get("role") or ""
    for t, name, _ in ROLE_WIRE_TARGETS.get(role, []):  # type only; identical across families
        if name is None or name == out_name:
            return t
    # Common Comfy output names
    upper = (out_name or "").upper()
    if upper in ("MODEL", "CLIP", "VAE", "LATENT", "IMAGE", "AUDIO"):
        return upper
    return "LATENT" if "latent" in (out_name or "").lower() else upper or "*"
