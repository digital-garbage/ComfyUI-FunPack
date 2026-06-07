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


@dataclass
class SceneSource:
    """How a scene's latent is born. V1 ignores this (uniform chain); Phase 2 maps
    it onto EmptyLTXVLatent (empty/t2v) or LTXV Image to Video (image/i2v)."""
    type: str = "carry"  # "carry" (default) | "empty" | "image" | "generated_frame"
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

    @staticmethod
    def from_dict(d: dict) -> "Scene":
        return Scene(
            id=d.get("id") or _new_id(),
            text=str(d.get("text", "")),
            transition_to_next=str(d.get("transition_to_next", "")),
            transition_frames=d.get("transition_frames"),
            frames=d.get("frames"),
            fps=d.get("fps"),
            frames_mode=str(d.get("frames_mode") or "timeline"),
            fps_mode=str(d.get("fps_mode") or "project"),
            width=d.get("width"),
            height=d.get("height"),
            source=SceneSource.from_dict(d.get("source")),
            rating=str(d.get("rating", "")),
            excluded=bool(d.get("excluded", False)),
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
    # Per-project pipeline config: the configured loader/node slots + linked inputs
    # (same shape as the global models.json). Empty {"slots": []} falls back to the
    # global default at build/read time; the editor seeds new projects from it.
    models: dict = field(default_factory=lambda: {"slots": []})
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
            models=dict(d.get("models") or {"slots": []}),
            created_at=float(d.get("created_at", time.time())),
            updated_at=float(d.get("updated_at", time.time())),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


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
    parts: list[str] = []
    anchor = (project.anchor or "").strip()
    if anchor:
        parts.append(anchor)
    trig_re = _leading_trigger_re() if for_generation else None
    for i, scene in enumerate(scenes):
        text = (scene.text or "").strip()
        if for_generation:
            has_lead = bool(text and trig_re and trig_re.match(text))
            if not has_lead:
                if i == 0:
                    marker = (project.intro_transition or "").strip()
                else:
                    marker = (scenes[i - 1].transition_to_next or "").strip()
                if not marker:
                    marker = f"scene {i + 1}"
                parts.append(marker)
        if text:
            parts.append(text)
    return " ".join(p for p in parts if p).strip()
