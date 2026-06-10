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


def is_mixed_source(scene: Scene) -> bool:
    return (scene.source.type or "carry") == "mixed"


def mixed_anchor_entry(scene: Scene, scene_index: int) -> Optional[dict]:
    """Per-scene i2v anchor for mixed source (applied at frame 0 in the sampler)."""
    if not is_mixed_source(scene) or not scene.source.media_ref:
        return None
    return {
        "enabled": True,
        "source": "anchor",
        "scene_id": scene.id,
        "scene_index": scene_index,
        "media_ref": scene.source.media_ref,
        "frame_idx": 0,
        "apply_at": 0,
        "strength": 0.35,
    }


@dataclass
class SceneSource:
    """How a scene's latent is born. V1 ignores this (uniform chain); Phase 2 maps
    it onto EmptyLTXVLatent (empty/t2v) or LTXV Image to Video (image/i2v)."""
    type: str = "carry"  # "carry" | "empty" | "image" | "generated_frame" | "mixed"
    media_ref: Optional[str] = None              # asset id, for type == "image"
    frame_ref: Optional[dict[str, Any]] = None   # {scene_id, frame_idx}, for "generated_frame"
    target: Optional[str] = None                 # wire dest for the image: "port:<id>" | "node:<slotId>:<input>"

    @staticmethod
    def from_dict(d: Optional[dict]) -> "SceneSource":
        d = d or {}
        return SceneSource(
            type=d.get("type", "carry"),
            media_ref=d.get("media_ref"),
            frame_ref=d.get("frame_ref"),
            target=d.get("target"),
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
    #   {blur: 0..1, fade_in: sec, fade_out: sec, zoom: "none"|"in"|"out"}.
    effects: dict = field(default_factory=dict)
    # Gain applied to this clip's ORIGINAL (LTXAV) audio at render. 1.0 = unchanged, 0 = mute.
    audio_volume: float = 1.0
    # Forward-compat per-scene knobs (uniform values still win in V1).
    frames: Optional[int] = None
    fps: Optional[int] = None
    # How `frames`/`fps` are resolved:
    #   "project"  → ignore the per-scene value, use the project global.
    #   "timeline" → use the per-scene value; the timeline trim handle writes it live.
    #   "custom"   → use the per-scene value; the timeline trim handle is locked.
    # Defaults: length follows the timeline (trim-derived), fps follows the project.
    frames_mode: str = "timeline"
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
            frames=d.get("frames"),
            fps=d.get("fps"),
            frames_mode=str(d.get("frames_mode") or "timeline"),
            fps_mode=str(d.get("fps_mode") or "project"),
            width=d.get("width"),
            height=d.get("height"),
            source=SceneSource.from_dict(d.get("source")),
            rating=str(d.get("rating", "")),
            excluded=bool(d.get("excluded", False)),
            gen_unit_id=d.get("gen_unit_id"),
            cut_offset_frames=int(d.get("cut_offset_frames", 0) or 0),
            guides=list(d.get("guides") or []),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def eff_frames(self, project: "Project") -> int:
        if self.frames_mode == "project" or self.frames is None:
            return project.num_frames_per_scene
        return self.frames

    def eff_fps(self, project: "Project") -> int:
        if self.fps_mode == "project" or self.fps is None:
            return project.frame_rate
        return self.fps


@dataclass
class Project:
    id: str = field(default_factory=_new_id)
    name: str = "Untitled"
    anchor: str = ""               # text prepended to every scene (character anchor)
    # Optional master prompt in Studio combined syntax. When the user edits it and
    # hits Apply, the editor reparses it into anchor + scenes + transitions (normal
    # Studio behaviour). Stored verbatim so the field round-trips; not used at build.
    global_prompt: str = ""
    negative_prompt: str = ""      # passed to the neg primitive node in the graph
    # Marker separating the anchor from the first scene (Studio needs a trigger to
    # close segments[0]=anchor). Defaults resolved at assembly from the library.
    intro_transition: str = ""
    scenes: list[Scene] = field(default_factory=list)
    # Uniform chain settings (V1).
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
    # Widget-input overrides for the built-in FunPack nodes (only used when the
    # corresponding slot == "funpack"). Keys match ComfyUI widget/input names exactly.
    studio_inputs: dict = field(default_factory=dict)
    sampler_inputs: dict = field(default_factory=dict)
    # Audio editing (render-time mix). keep_original_audio=False drops the per-clip LTXAV
    # audio entirely. audio_tracks = inserted tracks mixed over the montage, each:
    #   {id, media_ref, start_sec, volume, label}. With no tracks and original off → silent.
    keep_original_audio: bool = True
    audio_tracks: list = field(default_factory=list)
    # Per-project pipeline config: the configured loader/node slots + linked inputs
    # (same shape as the global models.json). Empty {"slots": []} falls back to the
    # global default at build/read time; the editor seeds new projects from it.
    models: dict = field(default_factory=lambda: {"slots": []})
    # Guide stack toggles — empty / stack_enabled=false keeps Studio carry behaviour.
    guide_settings: dict = field(default_factory=dict)
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
            studio_inputs=dict(d.get("studio_inputs") or {}),
            sampler_inputs=dict(d.get("sampler_inputs") or {}),
            keep_original_audio=bool(d.get("keep_original_audio", True)),
            audio_tracks=list(d.get("audio_tracks") or []),
            models=dict(d.get("models") or {"slots": []}),
            guide_settings=dict(d.get("guide_settings") or {}),
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


_TRIGGER_RE = None
_TRIGGER_RE_BUILT = False


def _leading_trigger_re():
    """Regex matching a leading transition trigger (direct DB trigger or generic 'scene N'
    label) at the start of a scene's text. Cached. None if nothing is available."""
    global _TRIGGER_RE, _TRIGGER_RE_BUILT
    if _TRIGGER_RE_BUILT:
        return _TRIGGER_RE
    _TRIGGER_RE_BUILT = True
    import re
    try:
        try:
            from templates import load_custom_transition_triggers
        except ImportError:
            from ...templates import load_custom_transition_triggers  # type: ignore
        trigs = list(load_custom_transition_triggers().keys())
    except Exception:
        trigs = []
    parts = [re.escape(t) for t in sorted(trigs, key=len, reverse=True)]
    parts.append(r"scene\s+[-+]?\d+")  # built-in generic split label
    _TRIGGER_RE = re.compile(r"^\s*(?:" + "|".join(parts) + r")\b", re.IGNORECASE)
    return _TRIGGER_RE


def build_combined_prompt(project: Project, include_excluded: bool = False,
                          for_generation: bool = False) -> str:
    """The master prompt = anchor + scene texts.

    Display (for_generation=False): a verbatim join — transition triggers/shortcuts stay
    in-text exactly as typed, so it round-trips with the global prompt.

    Generation (for_generation=True): Studio must split into EXACTLY one scene per timeline
    scene, but a `carry` scene's text has no leading transition, so Studio would merge it
    with the previous one (only the first scene gets generated). So before any scene whose
    text doesn't already begin with a transition trigger we inject a separator (the seam's
    transition_to_next, else the generic 'scene N' label) to force the split. This affects
    only what's sent to Studio, never the displayed global prompt.
    """
    scenes = [s for s in project.scenes if include_excluded or not s.excluded]
    units = group_generative_units(scenes)
    parts: list[str] = []
    anchor = (project.anchor or "").strip()
    if anchor:
        parts.append(anchor)
    # Only inject separators for a MULTI-unit run: editorial subclips count as one unit.
    inject = for_generation and len(units) > 1
    trig_re = _leading_trigger_re() if inject else None
    for i, (_uid, group) in enumerate(units):
        root = gen_unit_root(group)
        text = (root.text or "").strip()
        if inject and not (text and trig_re and trig_re.match(text)):
            prev_root = gen_unit_root(units[i - 1][1]) if i > 0 else None
            marker = (project.intro_transition or "").strip() if i == 0 else ((prev_root.transition_to_next if prev_root else "") or "").strip()
            if not marker:
                marker = f"scene {i + 1}"
            parts.append(marker)
        if text:
            parts.append(text)
    return " ".join(p for p in parts if p).strip()


def build_scene_guides_payload(project: Project) -> Optional[dict]:
    """Per-scene guide lists for the chain sampler.

    None when stack_enabled is off AND no mixed sources (Studio default:
    carry_i2v_guides from scene 1 template at frame 0).

    Mixed sources always emit a payload for continuation scenes that need both
    the Studio prior guide and the scene's own anchor at frame 0.
    """
    gs = normalize_guide_settings(project.guide_settings)
    active = [s for s in project.scenes if not s.excluded]
    has_mixed = any(is_mixed_source(s) and s.source.media_ref for s in active)
    if not gs["stack_enabled"] and not has_mixed:
        return None

    per_scene: list[Optional[list[dict]]] = []
    uses_custom = False
    for i, sc in enumerate(active):
        if i == 0:
            per_scene.append(None)
            continue
        entries: list[dict] = []
        own_anchor = mixed_anchor_entry(sc, i)

        if gs["stack_enabled"]:
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

        if own_anchor:
            has_prior = any(e.get("source") in ("template", "scene") for e in entries)
            if not has_prior:
                entries.insert(0, dict(STUDIO_DEFAULT_GUIDE))
            entries.append(own_anchor)
            uses_custom = True
        elif gs["stack_enabled"] and not entries:
            entries = [dict(STUDIO_DEFAULT_GUIDE)]
            uses_custom = True
        elif not entries:
            per_scene.append(None)
            continue
        else:
            uses_custom = True

        per_scene.append(entries)

    if not uses_custom:
        return None
    return {
        "stack_enabled": bool(gs["stack_enabled"]),
        "accumulate_prior": gs["accumulate_prior"],
        "has_mixed": has_mixed,
        "scenes": per_scene,
    }
