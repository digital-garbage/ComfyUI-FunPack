"""Built-in pipeline wiring rules: guided (locked) vs full-control (manual) mode.

When the built-in FunPack core is enabled and ``full_control`` is false, loader outputs
may only wire to their designated core ports. Internal core links (Studio -> Conditioning
-> Sampler -> decode) are never user-overridable. Toggle ``full_control`` to restore the
legacy free rewiring behaviour.
"""
from __future__ import annotations

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

FAMILY_WIRING: dict[str, dict] = {
    "ltxav": {},
    "minimax_h3": {
        "role_targets": {
            "audio_vae": [("VAE", None, "VAEDecodeAudio.vae"),
                          ("VAE", None, "FunPackLTXAVSceneChainSampler.audio_vae")],
            "audio_encoder": [],
            "empty_latent": [("LATENT", None, _H3_LATENT_PORT)],
            "video_latent": [("LATENT", None, _H3_LATENT_PORT)],
            "image_processing": [("IMAGE", None, "FunPackStudio.source_image")],
        },
        "type_chain_terminals": {"LATENT": [_H3_LATENT_PORT]},
        "default_wires": {
            "audio_vae": {"VAE": "port:VAEDecodeAudio.vae"},
            "audio_encoder": {},
            "empty_latent": {"LATENT": "port:" + _H3_LATENT_PORT},
            "video_latent": {"LATENT": "port:" + _H3_LATENT_PORT},
        },
        "port_labels": {
            _H3_LATENT_PORT: "Chain Sampler · latent_template (H3 AV latent)",
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
            "FunPackLTXAVSceneChainSampler.audio_vae": ("sampler", "audio_vae"),
            "VAEDecodeAudio.vae": ("audiodec", "vae"),
        },
    },
}

DEFAULT_FAMILY = "ltxav"


def family_of(models: Any) -> str:
    fam = str(models_dict(models).get("model_family") or DEFAULT_FAMILY).strip().lower()
    return fam if fam in FAMILY_WIRING else DEFAULT_FAMILY


def _role_targets(family: str) -> dict:
    out = {k: list(v) for k, v in ROLE_WIRE_TARGETS.items()}
    out.update({k: list(v) for k, v in (FAMILY_WIRING.get(family, {}).get("role_targets") or {}).items()})
    return out


def _chain_terminals(family: str) -> dict:
    out = {k: list(v) for k, v in TYPE_CHAIN_TERMINALS.items()}
    out.update({k: list(v) for k, v in (FAMILY_WIRING.get(family, {}).get("type_chain_terminals") or {}).items()})
    return out


def _default_wires(family: str) -> dict:
    out = {k: dict(v) for k, v in DEFAULT_WIRES_BY_ROLE.items()}
    out.update({k: dict(v) for k, v in (FAMILY_WIRING.get(family, {}).get("default_wires") or {}).items()})
    return out


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
    terminals = _chain_terminals(family)
    out: list[str] = []
    role_rules = role_targets.get(role or "", [])
    for t, name, port in role_rules:
        if t != out_type:
            continue
        if name is not None and out_name is not None and name != out_name:
            continue
        out.append(port)
    # Generic chain terminals only when the role has no explicit rule for this type
    # (e.g. audio_encoder LATENT must stay on Concat · audio_latent, not Studio · latent).
    has_explicit = any(t == out_type for t, _, _ in role_rules)
    if not has_explicit:
        for port in terminals.get(out_type, []):
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
    allowed = allowed_port_ids(role, out_type, out_name, family=family)
    if not allowed and role in ("custom", "", None):
        # Legacy slots without a role: allow canonical built-in ports for this output type.
        for rules in _role_targets(family).values():
            for t, name, port in rules:
                if t != out_type:
                    continue
                if name is not None and name != out_name:
                    continue
                if port not in allowed:
                    allowed.append(port)
        for port in _chain_terminals(family).get(out_type, []):
            if port not in allowed:
                allowed.append(port)
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
