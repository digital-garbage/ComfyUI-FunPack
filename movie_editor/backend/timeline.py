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
    type: str = "empty"  # "empty" | "image" | "generated_frame"
    media_ref: Optional[str] = None              # asset id, for type == "image"
    frame_ref: Optional[dict[str, Any]] = None   # {scene_id, frame_idx}, for "generated_frame"

    @staticmethod
    def from_dict(d: Optional[dict]) -> "SceneSource":
        d = d or {}
        return SceneSource(
            type=d.get("type", "empty"),
            media_ref=d.get("media_ref"),
            frame_ref=d.get("frame_ref"),
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
    source: SceneSource = field(default_factory=SceneSource)
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
            source=SceneSource.from_dict(d.get("source")),
            excluded=bool(d.get("excluded", False)),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class Project:
    id: str = field(default_factory=_new_id)
    name: str = "Untitled"
    anchor: str = ""               # text prepended to every scene (character anchor)
    # Marker separating the anchor from the first scene (Studio needs a trigger to
    # close segments[0]=anchor). Defaults resolved at assembly from the library.
    intro_transition: str = ""
    scenes: list[Scene] = field(default_factory=list)
    # Uniform chain settings (V1).
    seed: int = 1
    num_frames_per_scene: int = 97
    frame_rate: int = 25
    max_scenes: int = 8
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @staticmethod
    def from_dict(d: dict) -> "Project":
        return Project(
            id=d.get("id") or _new_id(),
            name=str(d.get("name", "Untitled")),
            anchor=str(d.get("anchor", "")),
            intro_transition=str(d.get("intro_transition", "")),
            scenes=[Scene.from_dict(s) for s in d.get("scenes", [])],
            seed=int(d.get("seed", 1)),
            num_frames_per_scene=int(d.get("num_frames_per_scene", 97)),
            frame_rate=int(d.get("frame_rate", 25)),
            max_scenes=int(d.get("max_scenes", 8)),
            created_at=float(d.get("created_at", time.time())),
            updated_at=float(d.get("updated_at", time.time())),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def build_combined_prompt(project: Project, include_excluded: bool = False) -> str:
    """Assemble the single prompt string Studio splits into scenes.

    Studio's split puts segments[0] (text before the FIRST transition trigger) as the
    anchor and segments[1:] as the generated scenes. So to get N scenes we must emit a
    trigger before EACH scene: `intro_transition` separates the anchor from scene 0,
    then each scene is preceded by the previous scene's `transition_to_next`.

    A separator is required at every seam or two scenes would merge. When a marker is
    empty we fall back to a generic `scene N` label (the built-in split pattern,
    no visual effect). NOTE: under Studio's default split placement="start" a generic
    label can leak into the scene text; the live parse_timeline preview makes the
    actual split visible so the user can pick a real library transition instead.

    Markers may themselves be shortcuts that expand to transition phrases, so emit
    them verbatim — Studio runs shortcuts -> transitions -> split (memory
    project-director-vision). `include_excluded=False` drops scenes flagged excluded.
    """
    parts: list[str] = []
    anchor = (project.anchor or "").strip()
    if anchor:
        parts.append(anchor)

    scenes = [s for s in project.scenes if include_excluded or not s.excluded]
    label_n = 1
    prev_marker = (project.intro_transition or "").strip() if anchor else ""
    for i, scene in enumerate(scenes):
        text = (scene.text or "").strip()
        # A separator is needed before this scene if there is preceding content
        # (the anchor, or an earlier scene) to split away from.
        need_sep = bool(anchor) or i > 0
        if need_sep:
            marker = prev_marker
            if not marker:
                marker = f"scene {label_n + 1}"
            parts.append(marker)
        label_n += 1
        if text:
            parts.append(text)
        prev_marker = (scene.transition_to_next or "").strip()

    return "\n".join(p for p in parts if p).strip()
