"""Timeline data model + Studio-format prompt assembly.

V1 drives the EXISTING uniform chain: one combined prompt (anchor + scene texts
joined by transition markers), one num_frames_per_scene, one frame_rate. The model
already carries per-scene `source` and per-scene length/fps so later phases can act
on them without a migration.

The combined prompt MUST match what FunPack Studio's split-by-transitions expects.
Pipeline order in Studio is: shortcuts -> transitions -> split (see memory
project-director-vision). We only assemble text here; ComfyUI does the rest.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# Studio default: one i2v guide from scene 1's template, applied at pixel frame 0.
STUDIO_DEFAULT_GUIDE: dict[str, Any] = {
    "enabled": True,
    "source": "template",
    "frame_idx": 0,
    "apply_at": 0,
    "strength": 0.35,
}


def normalize_guide_settings(raw: Optional[dict]) -> dict[str, bool]:
    """Guide stack toggles — all off unless explicitly enabled."""
    raw = raw or {}
    return {
        "stack_enabled": bool(raw.get("stack_enabled", False)),
        "accumulate_prior": bool(raw.get("accumulate_prior", False)),
    }


def effective_anchor(project: Project) -> str:
    """The project's manual anchor text (prepended to every scene by Studio)."""
    return (project.anchor or "").strip()


def effective_negative_prompt(project: Project) -> str:
    """The project's manual negative prompt."""
    return (project.negative_prompt or "").strip()


def effective_postfix(project: Project) -> str:
    """The project's manual postfix text (appended to every scene by Studio), or "" when
    the toggle is off. Symmetric to the anchor, which is prepended; the postfix tails each
    scene's text. Never part of the verbatim global prompt — a separate project setting."""
    if not getattr(project, "postfix_enabled", True):
        return ""
    return (project.postfix or "").strip()


def resolve_scene_identity_pin(cs: dict[str, Any]) -> Optional[str]:
    """Project-level identity pin reference for continuity guides, if set."""
    return cs.get("identity_pin_ref")


def continuity_settings_for_run(project: Project) -> dict[str, Any]:
    """Continuity knobs (per-scene identity pins are resolved in auto guides)."""
    return normalize_continuity_settings(project.continuity_settings)


def normalize_continuity_settings(raw: Optional[dict]) -> dict[str, Any]:
    """Automated cross-scene stability — guides, mid-scene anchor, identity pin.

    When auto_enabled (default), the Movie Editor builds sampler guide JSON and
    continuity knobs per run (carry / mixed / image / empty). Manual guide_settings
    stack_enabled overrides auto guide lists but mid-scene guide can still apply."""
    raw = raw or {}
    return {
        "auto_enabled": bool(raw.get("auto_enabled", True)),
        "identity_pin_ref": raw.get("identity_pin_ref") or None,
        "identity_pin_strength": float(raw.get("identity_pin_strength", 0.35)),
        "prior_scene_guides": bool(raw.get("prior_scene_guides", True)),
        "prior_scene_strength": float(raw.get("prior_scene_strength", 0.35)),
        "mid_scene_guide": bool(raw.get("mid_scene_guide", True)),
        "mid_scene_guide_strength": float(raw.get("mid_scene_guide_strength", 0.3)),
        "guide_decay": float(raw.get("guide_decay", 0.85)),
        "solo_scene_guides": bool(raw.get("solo_scene_guides", True)),
    }


def _anchor_media_ref(scene: Scene) -> Optional[str]:
    src = scene.source
    if not src:
        return None
    ref = getattr(src, "media_ref", None)
    stype = src.type or "carry"
    if ref and stype in ("image", "mixed", "generated_frame", "anchor_guide"):
        return ref
    return None


def _identity_pin_guide(media_ref: str, strength: float, *, is_pin: bool = True) -> dict:
    entry = {
        "enabled": True,
        "source": "image",
        "media_ref": media_ref,
        "frame_idx": 0,
        "apply_at": 0,
        "strength": max(0.0, min(1.0, float(strength))),
    }
    if is_pin:
        # Marks this entry (not prior-scene/mid-scene/template guides) as the one eligible
        # for the sampler's identity_transfer source-phase RoPE tag — see samplers.py.
        entry["identity_pin"] = True
    return entry


def _prior_scene_guide(prior: Scene, strength: float) -> Optional[dict]:
    ref = _anchor_media_ref(prior)
    if ref:
        # Reuses the same image-guide shape as the project identity pin, but this is the
        # PRIOR SCENE's own anchor (motion/layout continuity) — never the identity_pin tag.
        return _identity_pin_guide(ref, strength, is_pin=False)
    return {
        "enabled": True,
        "source": "template",
        "frame_idx": 0,
        "apply_at": 0,
        "strength": max(0.0, min(1.0, float(strength))),
    }


def _decayed_strength(base: float, scene_index: int, decay: float) -> float:
    if scene_index <= 0 or decay >= 0.999:
        return base
    return max(0.0, min(1.0, base * (decay ** scene_index)))


def build_auto_continuity_guides(full: Project, target: Project) -> Optional[dict]:
    """Build per-run funpack_scene_guides when auto continuity is on."""
    cs = continuity_settings_for_run(full)
    if not cs["auto_enabled"] or normalize_guide_settings(full.guide_settings)["stack_enabled"]:
        return None

    active_full = [s for s in full.scenes if not s.excluded]
    active_target = [s for s in target.scenes if not s.excluded]
    if not active_target:
        return None

    decay = cs["guide_decay"]
    project_pin = resolve_scene_identity_pin(cs)

    def _pin_for(_sc: Scene) -> Optional[str]:
        return project_pin

    def _entries_for_index(full_idx: int, chain_idx: int, scene: Scene) -> list[dict]:
        entries: list[dict] = []
        pin = _pin_for(scene)
        if pin:
            entries.append(_identity_pin_guide(pin, cs["identity_pin_strength"]))
        if full_idx > 0 and cs["prior_scene_guides"]:
            prior = active_full[full_idx - 1]
            strength = _decayed_strength(cs["prior_scene_strength"], chain_idx, decay)
            entries.append(_prior_scene_guide(prior, strength))
        elif chain_idx > 0:
            strength = _decayed_strength(cs["prior_scene_strength"], chain_idx, decay)
            entries.append({**STUDIO_DEFAULT_GUIDE, "strength": strength})
        return entries

    # Solo run (mixed, image, empty, generated_frame — one scene per queue request).
    if len(active_target) == 1:
        sc = active_target[0]
        full_idx = next((i for i, s in enumerate(active_full) if s.id == sc.id), 0)
        pin = _pin_for(sc)
        if full_idx <= 0 and not pin:
            return None
        wants_prior = cs["solo_scene_guides"] and solo_run_wants_prior_guides(sc)
        if full_idx > 0 and not wants_prior and not pin:
            return None
        entries = _entries_for_index(full_idx, 0, sc) if (full_idx > 0 and wants_prior) else []
        if pin and not any(e.get("media_ref") == pin for e in entries):
            entries.insert(0, _identity_pin_guide(pin, cs["identity_pin_strength"]))
        if not entries:
            return None
        return {"stack_enabled": True, "accumulate_prior": False, "scenes": [entries]}

    # Multi-scene carry chain.
    per_scene: list[Optional[list[dict]]] = []
    any_pin = False
    for chain_idx, sc in enumerate(active_target):
        full_idx = next((i for i, s in enumerate(active_full) if s.id == sc.id), chain_idx)
        pin = _pin_for(sc)
        if pin:
            any_pin = True
        if chain_idx == 0:
            entries = [_identity_pin_guide(pin, cs["identity_pin_strength"])] if pin else []
            per_scene.append(entries if entries else None)
            continue
        entries = _entries_for_index(full_idx, chain_idx, sc)
        per_scene.append(entries)

    if not any_pin and all(e is None or not e for e in per_scene[1:]):
        return None
    return {"stack_enabled": True, "accumulate_prior": False, "scenes": per_scene}


def continuity_media_refs(full: Project, target: Project) -> list[str]:
    """Media-bin ids that must exist for the active continuity/guide path."""
    cs = continuity_settings_for_run(full)
    refs: list[str] = []

    if cs["auto_enabled"]:
        if cs["identity_pin_ref"]:
            refs.append(cs["identity_pin_ref"])
        payload = build_auto_continuity_guides(full, target)
        if payload:
            for scene_list in payload.get("scenes") or []:
                for g in scene_list or []:
                    if g.get("source") == "image" and g.get("media_ref"):
                        refs.append(g["media_ref"])

    gs = normalize_guide_settings(full.guide_settings)
    if gs["stack_enabled"]:
        for sc in full.scenes:
            if sc.excluded:
                continue
            for raw in (sc.guides or []):
                g = GuideEntry.from_dict(raw if isinstance(raw, dict) else {})
                if g.enabled and g.source == "image" and g.media_ref:
                    refs.append(g.media_ref)

    return list(dict.fromkeys(refs))


@dataclass
class GuideEntry:
    """Optional per-scene guide (only used when project guide_settings.stack_enabled)."""
    enabled: bool = True
    source: str = "template"   # "template" | "scene" | "image"
    scene_id: Optional[str] = None
    media_ref: Optional[str] = None
    frame_idx: int = 0         # source frame; negative counts from end
    apply_at: int = 0          # target pixel frame in chunk; negative from end
    strength: float = 0.35

    @staticmethod
    def from_dict(d: Optional[dict]) -> "GuideEntry":
        d = d or {}
        return GuideEntry(
            enabled=bool(d.get("enabled", True)),
            source=str(d.get("source", "template")),
            scene_id=d.get("scene_id"),
            media_ref=d.get("media_ref"),
            frame_idx=int(d.get("frame_idx", 0)),
            apply_at=int(d.get("apply_at", 0)),
            strength=float(d.get("strength", 0.35)),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def scene_anchor_media_refs(target: Project) -> list[str]:
    """Media-bin ids required as i2v anchors on generative scenes."""
    refs: list[str] = []
    for sc in target.scenes:
        if sc.excluded or is_video_clip(sc):
            continue
        src = sc.source
        if not src:
            continue
        stype = source_type(sc)
        ref = src.media_ref
        if ref and stype in ("image", "generated_frame", "mixed", "anchor_guide"):
            refs.append(ref)
    return list(dict.fromkeys(refs))


def is_mixed_source(scene: Scene) -> bool:
    return (scene.source.type or "carry") == "mixed"


def is_anchor_guide(scene: Scene) -> bool:
    return source_type(scene) == "anchor_guide"


def anchor_guide_strength(scene: Scene) -> float:
    """The guide pull for an anchor_guide scene, clamped to a sane 0..1 range."""
    raw = getattr(scene.source, "guide_strength", None) if scene.source else None
    val = STUDIO_DEFAULT_GUIDE["strength"] if raw is None else float(raw)
    return max(0.0, min(1.0, val))


def _self_image_guide(media_ref: str, strength: float) -> dict:
    """A frame-0 guide from the scene's OWN image (anchor_guide mode). Same full 0..1
    range as every other guide strength — the measured audio-safe band is 0.25-0.35,
    but that is advice, not a clamp."""
    return {
        "enabled": True,
        "source": "image",
        "media_ref": media_ref,
        "frame_idx": 0,
        "apply_at": 0,
        "strength": max(0.0, min(1.0, float(strength))),
    }


def build_self_image_guides(target: Project) -> Optional[dict]:
    """Per-scene frame-0 guide from a scene's OWN image, for modes that want the input
    image to steer via guide attention. Merged onto continuity/manual stacks by the caller:

      - anchor_guide: the image is ONLY a guide (latent stays empty); per-scene strength.
      - mixed: the image is also the i2v anchor; this adds a reinforcing frame-0 guide so
        even the first scene (no prior to carry) gets guide attention from its own image.
    """
    active = [s for s in target.scenes if not s.excluded]
    per_scene: list[Optional[list[dict]]] = []
    any_guide = False
    for sc in active:
        ref = sc.source.media_ref if sc.source else None
        if ref and is_anchor_guide(sc):
            per_scene.append([_self_image_guide(ref, anchor_guide_strength(sc))])
            any_guide = True
        elif ref and is_mixed_source(sc):
            per_scene.append([_self_image_guide(ref, STUDIO_DEFAULT_GUIDE["strength"])])
            any_guide = True
        else:
            per_scene.append(None)
    if not any_guide:
        return None
    return {"stack_enabled": True, "accumulate_prior": False, "scenes": per_scene}


def merge_scene_guide_payloads(base: Optional[dict], extra: Optional[dict]) -> Optional[dict]:
    """Overlay two funpack_scene_guides payloads, combining per-scene entry lists by index."""
    if not base:
        return extra
    if not extra:
        return base
    a = base.get("scenes") or []
    b = extra.get("scenes") or []
    merged: list[Optional[list[dict]]] = []
    for i in range(max(len(a), len(b))):
        ea = a[i] if i < len(a) else None
        eb = b[i] if i < len(b) else None
        if ea and eb:
            merged.append(list(ea) + list(eb))
        else:
            merged.append(ea or eb)
    return {
        "stack_enabled": True,
        "accumulate_prior": bool(base.get("accumulate_prior") or extra.get("accumulate_prior")),
        "scenes": merged,
    }


def source_type(scene: Scene) -> str:
    return (scene.source.type if scene.source else None) or "carry"


def is_video_clip(scene: Scene) -> bool:
    """Locked timeline video — plays as-is, excluded from generation and global prompt."""
    return source_type(scene) == "video"


def is_generative_scene(scene: Scene) -> bool:
    return not is_video_clip(scene)


def solo_run_wants_prior_guides(scene: Scene) -> bool:
    """Solo i2v runs: only mixed borrows prior-scene guides; image/empty/generated_frame are anchor-only."""
    return is_mixed_source(scene)


def scene_accepts_stacked_guides(scene: Scene) -> bool:
    """Manual guide stack applies to carry/mixed/empty — not pure i2v anchor modes."""
    return source_type(scene) not in ("image", "generated_frame")


def build_mixed_solo_guides_payload(project: Project, mixed_scene: Scene) -> Optional[dict]:
    """Backward-compatible alias — delegates to build_auto_continuity_guides."""
    from dataclasses import replace
    segment = Project.from_dict(project.to_dict())
    segment.scenes = [mixed_scene]
    return build_auto_continuity_guides(project, segment)


def build_scene_anchors_payload(project: Project) -> Optional[dict]:
    """Legacy multi-scene chain anchors (unused by Movie Editor mixed runs).

    Mixed timeline scenes are generated solo via the graph-level Img2Video path."""
    active = [s for s in project.scenes if not s.excluded]
    anchors: dict[str, dict] = {}
    for i, sc in enumerate(active):
        if i == 0 or not is_mixed_source(sc) or not sc.source.media_ref:
            continue
        anchors[str(i)] = {
            "scene_id": sc.id,
            "media_ref": sc.source.media_ref,
            "strength": 1.0,
        }
    return anchors if anchors else None


@dataclass
class SceneSource:
    """How a scene's latent is born. V1 ignores this (uniform chain); Phase 2 maps
    it onto EmptyLTXVLatent (empty/t2v) or LTXV Image to Video (image/i2v).

    ``video`` = locked NLE clip (imported or converted from a scene); never generated.
    ``v2v`` = generative scene using a video file as the source (video-to-video).
    ``anchor_guide`` = image steers via a guide at frame 0, but the latent stays EMPTY
    (t2v). The image is never the i2v anchor; ``guide_strength`` sets its pull (0..1)."""
    type: str = "carry"  # carry | empty | image | generated_frame | mixed | video | v2v | anchor_guide
    media_ref: Optional[str] = None              # asset id — image/mixed anchor (Img2Video); anchor_guide guide image
    frame_ref: Optional[dict[str, Any]] = None   # {scene_id, frame_idx}, for "generated_frame"
    target: Optional[str] = None                 # wire dest for the image: "port:<id>" | "node:<slotId>:<input>"
    guide_strength: Optional[float] = None       # anchor_guide: guide pull (0..1), None -> default 0.35

    @staticmethod
    def from_dict(d: Optional[dict]) -> "SceneSource":
        d = d or {}
        gs = d.get("guide_strength")
        return SceneSource(
            type=d.get("type", "carry"),
            media_ref=d.get("media_ref"),
            frame_ref=d.get("frame_ref"),
            target=d.get("target"),
            guide_strength=float(gs) if gs is not None else None,
        )


@dataclass
class Scene:
    id: str = field(default_factory=_new_id)
    text: str = ""
    # Transition marker applied at the seam AFTER this scene (e.g. "cut", "blur").
    # Empty string = no explicit transition. Library values come from /funpack/transitions.
    transition_to_next: str = ""
    # Crossfade / transition length at that seam, in frames (post-decode pixel op only;
    # never re-encodes to latent). None/0 = hard cut. Editor-set via the timeline.
    transition_frames: Optional[int] = None
    # Post-decode VIDEO transition at the seam AFTER this scene (pure pixel op; never
    # re-encodes to latent): "" = hard cut | "crossfade" | "fadeblack" | "wipeleft" |
    # "wiperight" | "dissolve". Length comes from transition_frames. Distinct from
    # transition_to_next (a prompt trigger that shapes generation) — this only affects the
    # rendered/previewed pixels at the seam.
    video_transition: str = ""
    # Per-scene post-decode video effects applied to this clip (pixel ops), keys:
    #   {blur: 0..1, fade_in: sec, fade_out: sec, zoom: "none"|"in"|"out",
    #    zoom_ratio: 0..0.5, zoom_frames: int, zoom_start_frame: int}.
    effects: dict = field(default_factory=dict)
    # Gain applied to this clip's ORIGINAL (LTXAV) audio at render. 1.0 = unchanged, 0 = mute.
    audio_volume: float = 1.0
    # When True the embedded audio is muted on the clip; the same audio lives on a linked
    # project.audio_tracks entry (kind=separated) so video and audio edit independently.
    audio_separated: bool = False
    # Forward-compat per-scene knobs (uniform values still win in V1).
    frames: Optional[int] = None
    fps: Optional[int] = None
    # How `frames`/`fps` are resolved:
    #   "project"  → ignore the per-scene value, use the project global.
    #   "timeline" → use the per-scene value; the timeline trim handle writes it live.
    #   "custom"   → use the per-scene value; the timeline trim handle is locked.
    # Defaults: length and fps follow project until trimmed (timeline) or set in inspector (custom).
    frames_mode: str = "project"
    fps_mode: str = "project"
    width: Optional[int] = None
    height: Optional[int] = None
    source: SceneSource = field(default_factory=SceneSource)
    # FunPack Studio RLHF rating of this scene's last render (a V2_RATING_LABELS value).
    # Fed into Studio at the next generation of this scene's run so it refines. "" = unrated.
    rating: str = ""
    # Selective generation (UI-functional only in Phase 3 — see plan). `excluded`
    # scenes are skipped by a full Generate/Render; "generate this scene only" is a
    # transient request handled at the API/route level, not stored here.
    excluded: bool = False
    # Editorial timeline cuts of one generative scene share `gen_unit_id` (the root
    # scene's id). Only the root (cut_offset_frames == 0) owns prompt/rating/source
    # for generation; subclips are NLE trims of the same output.
    gen_unit_id: Optional[str] = None
    cut_offset_frames: int = 0
    # Per-scene guide overrides (only when project.guide_settings.stack_enabled).
    guides: list = field(default_factory=list)
    # Source trim inside generated media (slip edit): in-point seconds and optional duration.
    source_in: float = 0.0
    source_dur: Optional[float] = None
    # Saved generative state when this clip is converted to a locked ``video`` clip.
    # Restored on Convert to scene (scene → video → scene round-trip).
    scene_archive: Optional[dict] = None
    # Editorial pause (black/silent) after this clip before the next timeline segment.
    gap_after_sec: float = 0.0
    # Removed from the PLAN but its generated clip stays on the TIMELINE. Also `excluded`,
    # so generation/prompt skip it everywhere; export keeps it because it has a render.
    removed_from_plan: bool = False

    @staticmethod
    def from_dict(d: dict) -> "Scene":
        return Scene(
            id=d.get("id") or _new_id(),
            text=str(d.get("text", "")),
            transition_to_next=str(d.get("transition_to_next", "")),
            transition_frames=d.get("transition_frames"),
            video_transition=str(d.get("video_transition", "")),
            effects=dict(d.get("effects") or {}),
            audio_volume=float(d.get("audio_volume", 1.0)),
            audio_separated=bool(d.get("audio_separated", False)),
            frames=d.get("frames"),
            fps=d.get("fps"),
            frames_mode=str(d.get("frames_mode") or "project"),
            fps_mode=str(d.get("fps_mode") or "project"),
            width=d.get("width"),
            height=d.get("height"),
            source=SceneSource.from_dict(d.get("source")),
            rating=str(d.get("rating", "")),
            excluded=bool(d.get("excluded", False)),
            gen_unit_id=d.get("gen_unit_id"),
            cut_offset_frames=int(d.get("cut_offset_frames", 0) or 0),
            guides=list(d.get("guides") or []),
            source_in=float(d.get("source_in") or 0),
            source_dur=d.get("source_dur"),
            scene_archive=d.get("scene_archive"),
            gap_after_sec=float(d.get("gap_after_sec") or 0),
            removed_from_plan=bool(d.get("removed_from_plan", False)),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def eff_frames(self, project: "Project") -> int:
        # Honor frames_mode: only "timeline"/"custom" use the per-scene value. In "project" mode
        # (the inherit default) a leftover `frames` must be IGNORED so the scene tracks the project
        # length — otherwise a stale per-scene value makes the scene ignore project length changes.
        if self.frames_mode in ("timeline", "custom") and self.frames is not None:
            return self.frames
        return project.num_frames_per_scene

    def eff_fps(self, project: "Project") -> int:
        if self.fps_mode in ("timeline", "custom") and self.fps is not None:
            return self.fps
        return project.frame_rate


#: Sampler knobs that belong to the REFINEMENT KEY, not to the project.
#:
#: These are learned from ratings and live in the key's own state. A copy in the project file
#: was a second source of truth that outlived the key it came from: deleting every key left
#: the old values still applying, with no way to clear them but typing the neutral value back
#: in by hand. Stripped on load, so an existing project cleans itself the first time it is
#: opened and deleting the keys really does reset the behaviour.
#:
#: The node's own widgets are untouched — a raw ComfyUI graph with no Refiner in it still
#: drives them by hand, which is what `h3_gain_mode: manual` is for.
KEY_SCOPED_SAMPLER_INPUTS = frozenset({
    "h3_gain_mode", "h3_gain_video", "h3_gain_prompt", "h3_gain_audio",
    "h3_prompt_scale", "h3_taste_bias",
})


def _without_key_scoped(raw) -> dict:
    return {k: v for k, v in dict(raw or {}).items() if k not in KEY_SCOPED_SAMPLER_INPUTS}


@dataclass



class Project:
    id: str = field(default_factory=_new_id)
    name: str = "Untitled"
    anchor: str = ""               # extra anchor text prepended to every scene
    # Optional master prompt in Studio combined syntax. When the user edits it and
    # hits Apply, the editor reparses it into anchor + scenes + transitions (normal
    # Studio behaviour). Stored verbatim so the field round-trips; not used at build.
    global_prompt: str = ""
    negative_prompt: str = ""      # passed to the neg primitive node in the graph
    # Shared postfix text appended to EVERY scene (symmetric to anchor, which is prepended).
    # A standalone project setting — NOT part of the verbatim global prompt. postfix_enabled
    # toggles it without losing the text.
    postfix: str = ""
    postfix_enabled: bool = True
    # Marker separating the anchor from the first scene (Studio needs a trigger to
    # close segments[0]=anchor). Defaults resolved at assembly from the library.
    intro_transition: str = ""
    scenes: list[Scene] = field(default_factory=list)
    # Uniform chain settings (V1). `seed` is legacy — use sampler_inputs.seed for a fixed seed.
    seed: int = 1
    num_frames_per_scene: int = 97
    frame_rate: int = 25
    width: int = 768
    height: int = 512
    max_scenes: int = 8
    # Pluggable role slots: "funpack" = built-in Studio/ChainSampler; any other value
    # is a slot id from models.json. Stored now, full builder wiring in a future phase.
    conditioning_slot: str = "funpack"
    sampler_slot: str = "funpack"
    # "i2v" (default) or "t2v". t2v means shots start from the prompt, so nothing checks
    # or reports a missing anchor image. Images stay wireable either way — the mode only
    # decides what is EXPECTED, never what is allowed.
    generation_mode: str = "i2v"
    # Widget-input overrides for the built-in FunPack nodes (only used when the
    # corresponding slot == "funpack"). Keys match ComfyUI widget/input names exactly.
    studio_inputs: dict = field(default_factory=dict)
    sampler_inputs: dict = field(default_factory=dict)
    # Prompt `$name` variables — a project-scoped find/replace layer resolved at generation AFTER
    # shortcut-expand and the transition split (so they never affect scene cuts). Ordered list of
    # {"name": str, "value": str}. Shortcuts may reference variables; resolution is recursive.
    variables: list = field(default_factory=list)
    # MiniMax H3 ref2va reference media, in the order the user listed it — that order is
    # load-bearing: Studio bakes "<Picture i>" / "<Audio j>" / "<Video k>" labels from this
    # list, and the Chain Sampler encodes the same list into packed blocks. Entries are
    # {"kind": "image"|"audio"|"video", "filename": <media-bin file>, "audio": <optional
    # soundtrack file, video only>}. Ignored by every other model family.
    h3_references: list = field(default_factory=list)
    # Media marked "R" in the Media Bin / Easy Gen gallery: an ORDERED list of media-bin ids
    # (mark order — R1, R2, R3 — which is what the badge shows). Each becomes a wireable
    # source in Models & Pipeline, so any node input that takes an image / video / audio can
    # be fed from marked media. Order is the only thing distinguishing one from another, so
    # it is preserved on write and never re-sorted.
    references: list = field(default_factory=list)
    # Saved global-prompt templates: [{"name": str, "prompt": str, "variables": [...]}]. Selecting
    # one in the Composer applies its prompt + variables; loaded with the project (no Load button).
    prompt_templates: list = field(default_factory=list)
    # Which saved template the global prompt currently came from, so the Composer can show
    # it as selected and offer rename / update / delete for it. "" = none applied; the
    # prompt is whatever the user typed.
    active_prompt_template: str = ""
    # Refinement key for this project's runs. Feeds the FunPackRefinementKeyLoader (Studio /
    # Chain Sampler / SaveRefinementLatent). "default" = the keyless/default store. Shortcuts
    # bound to a non-default key layer their own per-scene training on top of this.
    refinement_key: str = "default"
    # Audio editing (render-time mix). keep_original_audio=False drops the per-clip LTXAV
    # audio entirely. audio_tracks = lanes mixed over the montage:
    #   overlay — {id, kind:"overlay", media_ref, start_sec, source_in_sec?, source_dur?, volume, label}
    #   separated — {id, kind:"separated", scene_id, start_sec, source_in_sec, source_dur,
    #     pinned_media, pinned_bin_ref, pinned_in_sec, pinned_dur, volume, label} — pinned_* freeze audio
    #     from the render at separation time so video-only regen keeps the old audio.
    # Overlay tracks use absolute timeline start_sec and persist across scene edits.
    keep_original_audio: bool = True
    audio_tracks: list = field(default_factory=list)
    # Graphics overlay lanes (bottom → top). Each lane is one timeline row; higher lanes draw on top.
    #   lane — {id, label}
    #   clip — {id, lane_id, kind:"image"|"text", start_sec, duration_sec, x, y, …}
    overlay_lanes: list = field(default_factory=list)
    overlay_tracks: list = field(default_factory=list)
    # Per-project pipeline config: the configured loader/node slots + linked inputs
    # (same shape as the global models.json). Empty {"slots": []} falls back to the
    # global default at build/read time; the editor seeds new projects from it.
    models: dict = field(default_factory=lambda: {"slots": []})
    # Guide stack toggles — empty / stack_enabled=false keeps Studio carry behaviour.
    guide_settings: dict = field(default_factory=dict)
    continuity_settings: dict = field(default_factory=dict)
    # Editor preferences carried WITH the project (autocomplete, ideas, anchor, i2v bypass,
    # and a snapshot of the shortcut-revolver mode). These are otherwise per-machine —
    # browser localStorage and a server-side sidecar — so on a fresh rented instance every
    # one of them is back at its default. Riding along in the project file is what makes
    # them survive the move. Never read by generation; the editor applies them on open.
    # An ABSENT key means "leave whatever this browser already has" — opening a project
    # saved before this existed must not reset anything.
    editor_settings: dict = field(default_factory=dict)
    # Last queued generation prompt fingerprint — used to auto-reset Studio session when
    # the timeline text changes (avoids stale repair memory overriding new actions).
    generation_meta: dict = field(default_factory=dict)
    # Persisted editor session: generated clip refs + preview ghosts (survive reload).
    scene_renders: dict = field(default_factory=dict)
    scene_ghosts: list = field(default_factory=list)
    # Cut order: scene ids in TIMELINE (result) order, independent of plan (scenes) order.
    # Empty = follow plan order (back-compat). Reordering the plan never touches this.
    timeline_order: list = field(default_factory=list)
    # True once the user explicitly reordered a timeline clip (◀ ▶). Until then the cut order
    # tracks the plan, so editing the global prompt re-derives it cleanly instead of letting a
    # stale id sequence scramble the clips against the plan badges.
    timeline_manually_ordered: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @staticmethod
    def from_dict(d: dict) -> "Project":
        return Project(
            id=d.get("id") or _new_id(),
            name=str(d.get("name", "Untitled")),
            anchor=str(d.get("anchor", "")),
            global_prompt=str(d.get("global_prompt", "")),
            negative_prompt=str(d.get("negative_prompt", "")),
            postfix=str(d.get("postfix", "")),
            postfix_enabled=bool(d.get("postfix_enabled", True)),
            intro_transition=str(d.get("intro_transition", "")),
            scenes=[Scene.from_dict(s) for s in d.get("scenes", [])],
            seed=int(d.get("seed", 1)),
            num_frames_per_scene=int(d.get("num_frames_per_scene", 97)),
            frame_rate=int(d.get("frame_rate", 25)),
            width=int(d.get("width", 768)),
            height=int(d.get("height", 512)),
            max_scenes=int(d.get("max_scenes", 8)),
            conditioning_slot=str(d.get("conditioning_slot", "funpack")),
            sampler_slot=str(d.get("sampler_slot", "funpack")),
            generation_mode=("t2v" if str(d.get("generation_mode", "")).lower() == "t2v" else "i2v"),
            studio_inputs=dict(d.get("studio_inputs") or {}),
            sampler_inputs=_without_key_scoped(d.get("sampler_inputs")),
            variables=list(d.get("variables") or []),
            h3_references=list(d.get("h3_references") or []),
            references=[str(r) for r in (d.get("references") or []) if r],
            prompt_templates=list(d.get("prompt_templates") or []),
            active_prompt_template=str(d.get("active_prompt_template") or ""),
            refinement_key=str(d.get("refinement_key") or "default"),
            keep_original_audio=bool(d.get("keep_original_audio", True)),
            audio_tracks=list(d.get("audio_tracks") or []),
            overlay_lanes=list(d.get("overlay_lanes") or []),
            overlay_tracks=list(d.get("overlay_tracks") or []),
            models=dict(d.get("models") or {"slots": []}),
            guide_settings=dict(d.get("guide_settings") or {}),
            continuity_settings=dict(d.get("continuity_settings") or {}),
            editor_settings=dict(d.get("editor_settings") or {}),
            generation_meta=dict(d.get("generation_meta") or {}),
            scene_renders=dict(d.get("scene_renders") or {}),
            scene_ghosts=list(d.get("scene_ghosts") or []),
            timeline_order=list(d.get("timeline_order") or []),
            timeline_manually_ordered=bool(d.get("timeline_manually_ordered", False)),
            created_at=float(d.get("created_at", time.time())),
            updated_at=float(d.get("updated_at", time.time())),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def gen_unit_id(scene: Scene) -> str:
    return scene.gen_unit_id or scene.id


def group_generative_units(scenes: list[Scene]) -> list[tuple[str, list[Scene]]]:
    """Consecutive timeline scenes that share a gen_unit_id are one generative unit."""
    units: list[tuple[str, list[Scene]]] = []
    for scene in scenes:
        uid = gen_unit_id(scene)
        if units and units[-1][0] == uid:
            units[-1][1].append(scene)
        else:
            units.append((uid, [scene]))
    return units


def gen_unit_root(group: list[Scene]) -> Scene:
    return min(group, key=lambda s: (int(s.cut_offset_frames or 0), s.id))


def collapse_generative_units(project: Project) -> Project:
    """Merge editorial subclips into one Scene per generative unit for ComfyUI runs."""
    clone = Project.from_dict(project.to_dict())
    active = [s for s in clone.scenes if not s.excluded]
    collapsed: list[Scene] = []
    for uid, group in group_generative_units(active):
        root = gen_unit_root(group)
        total_frames = sum(s.eff_frames(clone) for s in group)
        merged = Scene.from_dict(root.to_dict())
        merged.id = uid
        merged.gen_unit_id = None
        merged.cut_offset_frames = 0
        merged.frames = total_frames
        merged.frames_mode = "timeline"
        collapsed.append(merged)
    clone.scenes = collapsed
    return clone






def build_combined_prompt(project: Project, include_excluded: bool = False,
                          for_generation: bool = False) -> str:
    """The master prompt = anchor + scene texts, joined VERBATIM.

    Transition triggers/shortcuts stay in-text exactly as typed, so it round-trips with the
    global prompt. NOTHING is injected — not even for generation. Scene boundaries for
    generation are passed STRUCTURALLY via build_generation_scene_segments() (Studio uses that
    list directly), so we never have to smuggle a `scene N` delimiter into the prompt. The
    `for_generation` flag is kept for call-site compatibility but no longer changes the output.
    """
    scenes = [
        s for s in project.scenes
        if (include_excluded or not s.excluded) and is_generative_scene(s)
    ]
    parts: list[str] = []
    anchor = effective_anchor(project)
    if anchor:
        parts.append(anchor)
    for _uid, group in group_generative_units(scenes):
        text = (gen_unit_root(group).text or "").strip()
        if text:
            parts.append(text)
    return " ".join(p for p in parts if p).strip()


def build_generation_scene_segments(project: Project, include_excluded: bool = False) -> dict:
    """The run's scene boundaries, handed to Studio structurally so no `scene N` marker is ever
    injected into the prompt. The editor already knows every boundary (each generative unit is a
    scene), so we just list them.

    Returns {"anchor": <shared prefix text>, "scenes": [<raw unit text>, ...]} where each unit's
    text is prefixed with its REAL leading transition (intro_transition for the first unit, the
    previous unit's transition_to_next otherwise) when one exists — so Studio's per-scene split
    still picks up the transition's visual effect. A `carry` unit with no transition is just its
    text (effect = none). Studio (split_scenes_from_segments) expands shortcuts + extracts keys
    per item; one unit = exactly one scene.
    """
    scenes = [
        s for s in project.scenes
        if (include_excluded or not s.excluded) and is_generative_scene(s)
    ]
    units = group_generative_units(scenes)
    out: list[str] = []
    for i, (_uid, group) in enumerate(units):
        text = (gen_unit_root(group).text or "").strip()
        prev_root = gen_unit_root(units[i - 1][1]) if i > 0 else None
        trans = ((project.intro_transition or "").strip() if i == 0
                 else ((prev_root.transition_to_next if prev_root else "") or "").strip())
        out.append((trans + " " + text).strip() if trans else text)
    return {"anchor": effective_anchor(project), "scenes": out,
            "postfix": effective_postfix(project)}


def _scene_run_fingerprint(sc: Scene) -> str:
    """Source mode + media anchor — affects auto guides and i2v paths per run."""
    ref = (sc.source.media_ref if sc.source else None) or ""
    return f"{sc.id}:{source_type(sc)}:{ref}"


def generation_prompt_fingerprint(project: Project, target: Optional[Project] = None) -> dict[str, Any]:
    """Build generation/display prompts and run fingerprint (text + anchors + continuity)."""
    tgt = target or project
    gen_prompt = build_combined_prompt(tgt, for_generation=True)
    text_hash = hashlib.sha256(gen_prompt.encode("utf-8")).hexdigest()[:24]
    active = [s for s in tgt.scenes if not s.excluded]
    cs = continuity_settings_for_run(project)
    gs = normalize_guide_settings(project.guide_settings)
    run_key = "\n".join([
        text_hash,
        "|".join(_scene_run_fingerprint(s) for s in active),
        cs.get("identity_pin_ref") or "",
        effective_negative_prompt(project),
        effective_postfix(project),
        "stack" if gs["stack_enabled"] else "",
    ])
    run_hash = hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:24]
    return {
        "generation_prompt": gen_prompt,
        "display_prompt": build_combined_prompt(tgt, for_generation=False),
        "prompt_hash": text_hash,
        "run_hash": run_hash,
    }


def build_scene_guides_payload(project: Project) -> Optional[dict]:
    """Per-scene guide lists for the chain sampler.

    None when stack_enabled is off (Studio default: carry_i2v_guides from scene 1
    template at frame 0). i2v anchors for mixed scenes are separate — see
    build_scene_anchors_payload().
    """
    gs = normalize_guide_settings(project.guide_settings)
    if not gs["stack_enabled"]:
        return None
    active = [s for s in project.scenes if not s.excluded]
    per_scene: list[Optional[list[dict]]] = []
    for i, sc in enumerate(active):
        if i == 0 or not scene_accepts_stacked_guides(sc):
            per_scene.append(None)
            continue
        entries: list[dict] = []
        for raw in (sc.guides or []):
            g = GuideEntry.from_dict(raw if isinstance(raw, dict) else {})
            if g.enabled:
                entries.append(g.to_dict())
        if gs["accumulate_prior"]:
            for j in range(i):
                ref = active[j]
                acc = dict(STUDIO_DEFAULT_GUIDE)
                acc["source"] = "scene"
                acc["scene_id"] = ref.id
                acc["scene_index"] = j
                entries.append(acc)
        if not entries:
            entries = [dict(STUDIO_DEFAULT_GUIDE)]
        per_scene.append(entries)
    return {
        "stack_enabled": True,
        "accumulate_prior": gs["accumulate_prior"],
        "scenes": per_scene,
    }
