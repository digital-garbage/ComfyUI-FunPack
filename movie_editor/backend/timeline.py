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


def normalize_character_bible(raw: Optional[dict]) -> dict[str, Any]:
    """Structured character profile — text merges into the anchor; face ref can drive identity pin."""
    raw = raw or {}
    return {
        "name": str(raw.get("name", "")).strip(),
        "appearance": str(raw.get("appearance", "")).strip(),
        "body": str(raw.get("body", "")).strip(),
        "wardrobe": str(raw.get("wardrobe", "")).strip(),
        "always_include": str(raw.get("always_include", "")).strip(),
        "never_include": str(raw.get("never_include", "")).strip(),
        "face_ref": raw.get("face_ref") or None,
        "body_ref": raw.get("body_ref") or None,
        "detail_ref": raw.get("detail_ref") or None,
        "sync_identity_pin": bool(raw.get("sync_identity_pin", True)),
    }


def build_character_bible_anchor(bible: Optional[dict]) -> str:
    """Render one character profile into anchor prose."""
    return build_characters_anchor([bible] if bible else [])


def build_characters_anchor(characters: list[Optional[dict]]) -> str:
    """Merge multiple character profiles into one anchor block."""
    blocks: list[str] = []
    for raw in characters:
        b = normalize_character_bible(raw)
        if not any(b[k] for k in ("name", "appearance", "body", "wardrobe", "always_include")):
            continue
        parts: list[str] = []
        if b["name"]:
            parts.append(f"Character: {b['name']}.")
        if b["appearance"]:
            parts.append(f"Appearance: {b['appearance']}.")
        if b["body"]:
            parts.append(f"Body: {b['body']}.")
        if b["wardrobe"]:
            parts.append(f"Wardrobe: {b['wardrobe']}.")
        if b["always_include"]:
            text = b["always_include"].rstrip(".")
            parts.append(text + ".")
        blocks.append(" ".join(parts).strip())
    return " ".join(blocks).strip()


def _load_character_map() -> dict[str, dict]:
    try:
        from .characters import load_character_map
    except ImportError:
        from characters import load_character_map  # type: ignore
    return load_character_map()


def _project_uses_scene_characters(project: Project) -> bool:
    return any((s.character_ids or []) for s in project.scenes)


def _characters_for_scene(scene: Scene, char_map: dict[str, dict]) -> list[dict]:
    return [char_map[cid] for cid in (scene.character_ids or []) if cid in char_map]


def _scene_character_prefix(scene: Scene, char_map: dict[str, dict]) -> str:
    return build_characters_anchor(_characters_for_scene(scene, char_map))


def effective_anchor(project: Project, char_map: Optional[dict[str, dict]] = None) -> str:
    """Manual project anchor + legacy project.character_bible when no per-scene characters."""
    char_map = char_map if char_map is not None else _load_character_map()
    parts: list[str] = []
    if not _project_uses_scene_characters(project):
        legacy = build_character_bible_anchor(project.character_bible)
        if legacy:
            parts.append(legacy)
    manual = (project.anchor or "").strip()
    if manual:
        parts.append(manual)
    return " ".join(parts).strip()


def effective_negative_prompt(project: Project, char_map: Optional[dict[str, dict]] = None) -> str:
    """Project negative plus never_include from scene characters (and legacy bible)."""
    char_map = char_map if char_map is not None else _load_character_map()
    manual = (project.negative_prompt or "").strip()
    never_parts: list[str] = []
    seen: set[str] = set()
    if not _project_uses_scene_characters(project):
        leg = normalize_character_bible(project.character_bible)["never_include"]
        if leg:
            never_parts.append(leg)
    for scene in project.scenes:
        if scene.excluded:
            continue
        for cid in (scene.character_ids or []):
            if cid in seen:
                continue
            seen.add(cid)
            c = char_map.get(cid)
            if not c:
                continue
            never = normalize_character_bible(c)["never_include"]
            if never:
                never_parts.append(never)
    never = ", ".join(never_parts).strip()
    if never and manual:
        return f"{manual}, {never}"
    return never or manual


def resolve_scene_identity_pin(
    scene: Scene,
    char_map: dict[str, dict],
    cs: dict[str, Any],
) -> Optional[str]:
    """First assigned character with a face ref drives the pin for this scene."""
    for cid in (scene.character_ids or []):
        c = char_map.get(cid)
        if c and c.get("face_ref"):
            return c["face_ref"]
    return cs.get("identity_pin_ref")


def character_media_refs_for_project(project: Project, char_map: Optional[dict[str, dict]] = None) -> list[str]:
    char_map = char_map if char_map is not None else _load_character_map()
    refs: list[str] = []
    if not _project_uses_scene_characters(project):
        refs.extend(character_bible_media_refs(project))
    seen: set[str] = set()
    for scene in project.scenes:
        if scene.excluded:
            continue
        for c in _characters_for_scene(scene, char_map):
            for key in ("face_ref", "body_ref", "detail_ref"):
                ref = c.get(key)
                if ref and ref not in seen:
                    seen.add(ref)
                    refs.append(ref)
    return refs


def character_bible_media_refs(project: Project) -> list[str]:
    b = normalize_character_bible(project.character_bible)
    refs: list[str] = []
    for key in ("face_ref", "body_ref", "detail_ref"):
        ref = b.get(key)
        if ref:
            refs.append(ref)
    return refs


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
    if ref and stype in ("image", "mixed", "generated_frame"):
        return ref
    return None


def _identity_pin_guide(media_ref: str, strength: float) -> dict:
    return {
        "enabled": True,
        "source": "image",
        "media_ref": media_ref,
        "frame_idx": 0,
        "apply_at": 0,
        "strength": max(0.25, min(0.5, float(strength))),
    }


def _prior_scene_guide(prior: Scene, strength: float) -> Optional[dict]:
    ref = _anchor_media_ref(prior)
    if ref:
        return _identity_pin_guide(ref, strength)
    return {
        "enabled": True,
        "source": "template",
        "frame_idx": 0,
        "apply_at": 0,
        "strength": max(0.25, min(0.5, float(strength))),
    }


def _decayed_strength(base: float, scene_index: int, decay: float) -> float:
    if scene_index <= 0 or decay >= 0.999:
        return base
    return max(0.25, min(0.5, base * (decay ** scene_index)))


def build_auto_continuity_guides(full: Project, target: Project) -> Optional[dict]:
    """Build per-run funpack_scene_guides when auto continuity is on."""
    cs = continuity_settings_for_run(full)
    if not cs["auto_enabled"] or normalize_guide_settings(full.guide_settings)["stack_enabled"]:
        return None

    active_full = [s for s in full.scenes if not s.excluded]
    active_target = [s for s in target.scenes if not s.excluded]
    if not active_target:
        return None

    char_map = _load_character_map()
    decay = cs["guide_decay"]

    def _pin_for(sc: Scene) -> Optional[str]:
        return resolve_scene_identity_pin(sc, char_map, cs)

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
    """Extra media-bin ids to copy for auto continuity (pin + prior anchors)."""
    cs = continuity_settings_for_run(full)
    char_map = _load_character_map()
    refs: list[str] = list(character_media_refs_for_project(full, char_map))
    if cs["auto_enabled"] and cs["identity_pin_ref"]:
        refs.append(cs["identity_pin_ref"])
    for sc in full.scenes:
        if sc.excluded:
            continue
        pin = resolve_scene_identity_pin(sc, char_map, cs)
        if pin:
            refs.append(pin)
    payload = build_auto_continuity_guides(full, target)
    if payload:
        for scene_list in payload.get("scenes") or []:
            for g in scene_list or []:
                if g.get("source") == "image" and g.get("media_ref"):
                    refs.append(g["media_ref"])
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


def is_mixed_source(scene: Scene) -> bool:
    return (scene.source.type or "carry") == "mixed"


def source_type(scene: Scene) -> str:
    return (scene.source.type if scene.source else None) or "carry"


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
    it onto EmptyLTXVLatent (empty/t2v) or LTXV Image to Video (image/i2v)."""
    type: str = "carry"  # "carry" | "empty" | "image" | "generated_frame" | "mixed"
    media_ref: Optional[str] = None              # asset id — image/mixed anchor (Img2Video)
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
    # Character library ids assigned to this scene (merged into its prompt on generate).
    character_ids: list = field(default_factory=list)

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
            character_ids=[str(x) for x in (d.get("character_ids") or []) if x],
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
    anchor: str = ""               # extra anchor text (character bible prepends before this)
    character_bible: dict = field(default_factory=dict)
    # Optional master prompt in Studio combined syntax. When the user edits it and
    # hits Apply, the editor reparses it into anchor + scenes + transitions (normal
    # Studio behaviour). Stored verbatim so the field round-trips; not used at build.
    global_prompt: str = ""
    negative_prompt: str = ""      # passed to the neg primitive node in the graph
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
    continuity_settings: dict = field(default_factory=dict)
    # Last queued generation prompt fingerprint — used to auto-reset Studio session when
    # the timeline text changes (avoids stale repair memory overriding new actions).
    generation_meta: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @staticmethod
    def from_dict(d: dict) -> "Project":
        return Project(
            id=d.get("id") or _new_id(),
            name=str(d.get("name", "Untitled")),
            anchor=str(d.get("anchor", "")),
            character_bible=dict(d.get("character_bible") or {}),
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
            continuity_settings=dict(d.get("continuity_settings") or {}),
            generation_meta=dict(d.get("generation_meta") or {}),
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
    char_map = _load_character_map()
    parts: list[str] = []
    anchor = effective_anchor(project, char_map)
    if anchor:
        parts.append(anchor)
    # Only inject separators for a MULTI-unit run: editorial subclips count as one unit.
    inject = for_generation and len(units) > 1
    trig_re = _leading_trigger_re() if inject else None
    for i, (_uid, group) in enumerate(units):
        root = gen_unit_root(group)
        text = (root.text or "").strip()
        char_prefix = _scene_character_prefix(root, char_map)
        if char_prefix:
            text = f"{char_prefix} {text}".strip() if text else char_prefix
        if inject and not (text and trig_re and trig_re.match(text)):
            prev_root = gen_unit_root(units[i - 1][1]) if i > 0 else None
            marker = (project.intro_transition or "").strip() if i == 0 else ((prev_root.transition_to_next if prev_root else "") or "").strip()
            if not marker:
                marker = f"scene {i + 1}"
            parts.append(marker)
        if text:
            parts.append(text)
    return " ".join(p for p in parts if p).strip()


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
    char_map = _load_character_map()
    char_key = "|".join(
        f"{s.id}:{','.join(s.character_ids or [])}" for s in active
    )
    run_key = "\n".join([
        text_hash,
        "|".join(_scene_run_fingerprint(s) for s in active),
        char_key,
        cs.get("identity_pin_ref") or "",
        effective_negative_prompt(project, char_map),
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
