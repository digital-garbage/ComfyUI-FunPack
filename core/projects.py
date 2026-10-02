"""Projects: an ordered list of scenes, one JSON file each.

A project is the editor's whole saved state: the scenes and what a person did to
them on the timeline (trims, seams, effects, audio, overlays), the pipeline they
configured, and the renders each scene last produced. It is the one thing in the
app that outlives the code that wrote it, so everything read back is checked --
each field by type and range, the free-form ones (effects, audio and overlay
lanes, the pipeline's own settings) by shape and size, their inner keys left to
whatever consumes them. A field this module does not name is DROPPED on save:
that is how a hand-edited file or a hostile body is kept out of a path or a
command line, and it is also why a new editor feature must add its field here.

A scene's frames/fps follow the PROJECT until trimmed ("frames_mode"): a stale
per-scene value never overrides a project length change.

Synchronous, one file per project, stdlib json. Projects are small.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field

from . import config, media

#: A generated id, never a user-supplied name, is what reaches the filesystem.
_ID = re.compile(r"\A[0-9a-f]{12}\Z")

MAX_NAME = 120


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def is_id(value) -> bool:
    """True for something this module generated. Everything from a request is
    checked against this before it is used to build a path."""
    return isinstance(value, str) and bool(_ID.match(value))


def _clean_name(raw, fallback="Untitled") -> str:
    name = (raw if isinstance(raw, str) else "").strip()
    return name[:MAX_NAME] or fallback


MAX_SETTING = 16384


def _whole(value) -> int | None:
    """A whole positive number, or nothing. Everything read back out of a
    project file goes through here: the file outlives the code that wrote it."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 1 <= number <= MAX_SETTING else None


_TOKEN = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")
#: What a free-form field (effects, tracks, pipeline settings) may weigh once it is JSON.
MAX_BLOB = 4 * 1024 * 1024
FRAME_MODES = ("project", "timeline", "custom")
SOURCE_TYPES = ("carry", "empty", "image", "generated_frame", "mixed", "video", "v2v", "anchor_guide")


def is_token(value) -> bool:
    """A client-made id (scenes, tracks): letters, digits, `_` and `-`. Never a path --
    anything that builds a filename from one runs it through `safe_part`."""
    return isinstance(value, str) and bool(_TOKEN.match(value))


def safe_part(value) -> str:
    """`value` as one filename fragment: nothing but token characters."""
    return re.sub(r"[^A-Za-z0-9_-]", "", str(value or ""))[:64] or "x"


def _str(raw, default="", limit=MAX_BLOB) -> str:
    return raw[:limit] if isinstance(raw, str) else default


def _bool(raw, default=False) -> bool:
    return raw if isinstance(raw, bool) else default


def _num(raw, default=None, lo=None, hi=None):
    """A finite number within [lo, hi], else `default`. A bool is not a number here."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw != raw or raw in (float("inf"), float("-inf")):
        return default
    if (lo is not None and raw < lo) or (hi is not None and raw > hi):
        return default
    return raw


def _int(raw, default=None, lo=None, hi=None):
    n = _num(raw, None, lo, hi)
    return int(n) if n is not None else default


def _blob(raw, kind):
    """`raw` when it is a `kind` (dict/list) that survives JSON and is not huge, else empty."""
    if not isinstance(raw, kind):
        return kind()
    try:
        if len(json.dumps(raw)) > MAX_BLOB:
            return kind()
    except (TypeError, ValueError):
        return kind()
    return raw


def _dicts(raw) -> list:
    """A list of objects: the other kinds of row are dropped."""
    return [x for x in _blob(raw, list) if isinstance(x, dict)]


def _clean_effects(raw) -> dict:
    """Flat {name: bool | number | str}: what a clip's pixel effects are."""
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in list(raw.items())[:64]
            if isinstance(k, str) and len(k) <= 40 and (isinstance(v, (bool, int, float, str)) and (not isinstance(v, str) or len(v) <= 80)
                                       and (not isinstance(v, float) or v == v))}


def _clean_source(raw) -> dict:
    d = raw if isinstance(raw, dict) else {}
    out = {"type": d.get("type") if d.get("type") in SOURCE_TYPES else "carry"}
    for key in ("media_ref", "target"):
        out[key] = _str(d.get(key), None, 200) or None
    ref = d.get("frame_ref")
    out["frame_ref"] = _blob(ref, dict) or None
    out["guide_strength"] = _num(d.get("guide_strength"), None, 0, 1)
    return out


#: What a person can say about a result: it was good, or it was not.
#:
#: Binary because that is all anything downstream reads. dev proved it on H3 --
#: the steering only ever takes the SIGN of a rating, never its size, so a
#: ten-point scale was a UI claiming a precision nothing used, and at the sample
#: sizes this runs at a mediocre middle rating only diluted the direction.
#:
#: Core keeps the word and nothing else. What either of them MEANS belongs to
#: whatever learns from them, which is not here.
RATINGS = ("liked", "disliked")
#: The label the picker stored on a scene ("10", "Disliked: bad image", ...): kept as the
#: picker wrote it, turned into liked/disliked only where a taste key is taught.
MAX_RATING = 80


@dataclass
class Scene:
    id: str = field(default_factory=_new_id)
    text: str = ""
    #: The asset this scene last produced, so a reload shows the timeline the
    #: user left rather than an empty one they have to regenerate.
    result: str | None = None
    #: How long this clip RUNS on the timeline -- a crop, made here. It is not
    #: what a regenerate uses: that reads the project, because the crop was a
    #: timeline decision and a regenerate is a new scene.
    length: int | None = None
    rating: str = ""
    #: A media library id, picked to size this scene's generation -- not
    #: included in it. v4's "drop a picture on the timeline" only ever fed a
    #: resolution/aspect-ratio node; the picture's own pixels went nowhere. Per
    #: scene, not per project: different shots can want a different canvas.
    source_image: str | None = None
    #: Media library ids, in the order they were added -- reference images for
    #: whatever the pipeline's own reference input wants them for. Per scene
    #: for the same reason source_image is: a reference is about THIS shot.
    references: list[str] = field(default_factory=list)
    #: A prompt trigger at the seam AFTER this scene, and the post-decode pixel
    #: transition there (length in frames): a crossfade never re-encodes to latent.
    transition_to_next: str = ""
    transition_frames: int | None = None
    video_transition: str = ""
    #: Per-clip pixel effects, flat: blur, fades, zoom, flips, crop, fit, reverse.
    effects: dict = field(default_factory=dict)
    #: Gain on this clip's own sound; separated = its sound lives on an audio track.
    audio_volume: float = 1.0
    audio_separated: bool = False
    #: Length/fps/size, used only in the "timeline"/"custom" modes; "project" follows the project.
    frames: int | None = None
    fps: int | None = None
    frames_mode: str = "project"
    fps_mode: str = "project"
    width: int | None = None
    height: int | None = None
    #: How the scene's latent is born (carry / image / video clip / ...).
    source: dict = field(default_factory=lambda: _clean_source(None))
    excluded: bool = False
    #: Timeline cuts of one generated clip share the root scene's id; only the root
    #: (cut_offset_frames == 0) owns prompt, rating and source.
    gen_unit_id: str | None = None
    cut_offset_frames: int = 0
    guides: list = field(default_factory=list)
    #: Slip edit: where in the generated media this clip starts and how long it runs.
    source_in: float = 0.0
    source_dur: float | None = None
    #: The generative state a clip had before it was locked to a video clip.
    scene_archive: dict | None = None
    gap_after_sec: float = 0.0
    #: Gone from the plan but its generated clip stays on the timeline.
    removed_from_plan: bool = False

    def eff_frames(self, project: "Project") -> int:
        """How many frames this clip runs. Only the "timeline"/"custom" modes use the scene's
        own number: in "project" mode a stale `frames` is ignored so the clip tracks the project."""
        if self.frames_mode in ("timeline", "custom") and self.frames is not None:
            return self.frames
        return project.num_frames_per_scene

    def eff_fps(self, project: "Project") -> int:
        if self.fps_mode in ("timeline", "custom") and self.fps is not None:
            return self.fps
        return project.frame_rate

    @staticmethod
    def from_dict(d) -> "Scene":
        d = d if isinstance(d, dict) else {}
        sid = d.get("id")
        result = d.get("result")
        rating = d.get("rating")
        source_image = d.get("source_image")
        raw_refs = d.get("references")
        guide = d.get("gen_unit_id")
        archive = _blob(d.get("scene_archive"), dict)
        return Scene(
            id=sid if is_token(sid) else _new_id(),
            text=_str(d.get("text")),
            result=result if isinstance(result, str) else None,
            length=_whole(d.get("length")),
            rating=_str(rating, "", MAX_RATING),
            source_image=source_image if media.is_id(source_image) else None,
            references=[r for r in raw_refs if media.is_id(r)] if isinstance(raw_refs, list) else [],
            transition_to_next=_str(d.get("transition_to_next"), "", 200),
            transition_frames=_int(d.get("transition_frames"), None, 0, MAX_SETTING),
            video_transition=_str(d.get("video_transition"), "", 40),
            effects=_clean_effects(d.get("effects")),
            audio_volume=_num(d.get("audio_volume"), 1.0, 0.0, 10.0),
            audio_separated=_bool(d.get("audio_separated")),
            frames=_int(d.get("frames"), None, 1, MAX_SETTING),
            fps=_int(d.get("fps"), None, 1, 1000),
            frames_mode=d.get("frames_mode") if d.get("frames_mode") in FRAME_MODES else "project",
            fps_mode=d.get("fps_mode") if d.get("fps_mode") in FRAME_MODES else "project",
            width=_int(d.get("width"), None, 1, MAX_SETTING),
            height=_int(d.get("height"), None, 1, MAX_SETTING),
            source=_clean_source(d.get("source")),
            excluded=_bool(d.get("excluded")),
            gen_unit_id=guide if is_token(guide) else None,
            cut_offset_frames=_int(d.get("cut_offset_frames"), 0, 0, 10 ** 7),
            guides=_dicts(d.get("guides")),
            source_in=_num(d.get("source_in"), 0.0, 0.0, 10 ** 6),
            source_dur=_num(d.get("source_dur"), None, 0.0, 10 ** 6),
            scene_archive=archive or None,
            gap_after_sec=_num(d.get("gap_after_sec"), 0.0, 0.0, 10 ** 5),
            removed_from_plan=_bool(d.get("removed_from_plan")),
        )


def _clean_video(raw) -> dict:
    """What the app is holding for the pipeline's `project.video` inputs.

    Core does not know what a video setting IS. Which of them exist is the
    pipeline's business -- it says so with a role -- and naming width and height
    here would be core naming an implementation. What it knows is that a project
    file is the one thing in this app that outlives the code that wrote it, so
    everything read back out of one is checked: a whole positive number, or gone.
    """
    if not isinstance(raw, dict):
        return {}
    clean = {}
    for key, value in raw.items():
        # `True` is an int in Python and would land in a width.
        if not isinstance(key, str) or isinstance(value, bool):
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= MAX_SETTING:
            clean[key] = number
    return clean


def _clean_variables(raw) -> list:
    """[{"name": str, "value": str}, ...] -- anything else in a slot is
    dropped rather than guessed at, the same rule every other field reads
    back out of a project file follows. A name that is empty after stripping
    the leading `$` is not a variable anyone could ever reference, so it is
    dropped rather than kept as a row nothing can ever match."""
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").lstrip("$").strip()
        if not name:
            continue
        value = item.get("value")
        out.append({"name": name[:MAX_NAME], "value": value if isinstance(value, str) else ""})
    return out


def _clean_templates(raw) -> list:
    """[{"name", "anchor", "scenes": [str], "variables": [{name, value}]}]. A template is
    the scenes themselves, not a Story text: the cut word it was saved under
    may no longer be one. A v4 template (a single `prompt` string) is kept as
    it came, and read by whoever applies it. Nameless or non-object entries go;
    a name twice keeps the last."""
    if not isinstance(raw, list):
        return []
    out: dict[str, dict] = {}
    for t in raw:
        if not isinstance(t, dict):
            continue
        name = str(t.get("name") or "").strip()[:MAX_NAME]
        if not name:
            continue
        item = {"name": name, "variables": _clean_variables(t.get("variables"))}
        if isinstance(t.get("anchor"), str):
            item["anchor"] = t["anchor"]
        scenes = t.get("scenes")
        if isinstance(scenes, list):
            item["scenes"] = [s if isinstance(s, str) else "" for s in scenes]
        if isinstance(t.get("prompt"), str):
            item["prompt"] = t["prompt"]
        if "scenes" not in item and "prompt" not in item:
            continue                  # holds no text: applying it would wipe the Story
        out.pop(name, None)
        out[name] = item
    return list(out.values())


def _clean_editor_settings(raw) -> dict:
    """Editor preferences that travel with the project (a new rental has an
    empty browser). Values are whatever the editor wrote -- bool, number,
    string, small object -- so only the shape is checked: a string-keyed object."""
    return {k: v for k, v in raw.items() if isinstance(k, str)} if isinstance(raw, dict) else {}


@dataclass
class Project:
    id: str = field(default_factory=_new_id)
    name: str = "Untitled"
    scenes: list[Scene] = field(default_factory=list)
    #: Settings the whole project is generated at -- size, length -- rather than
    #: any one scene. A scene cropped on the timeline and regenerated comes back
    #: at the project's length: the crop was a timeline decision and a regenerate
    #: is a new scene.
    video: dict = field(default_factory=dict)
    #: A negative prompt, sent at whatever "project.negative" input the
    #: pipeline declares. Project-level for the same reason `video` is:
    #: something set rarely, once, not re-typed per scene.
    negative: str = ""
    #: Text prepended to every scene's prompt. v4's "anchor" -- the shared
    #: subject/setting line a whole project holds once instead of retyping
    #: per scene.
    anchor: str = ""
    #: Text appended to every scene's prompt. v4's "postfix" -- symmetric to
    #: anchor, but with its own on/off switch: v4 proved that a postfix
    #: someone spent time writing needs to be turned off for a quick test
    #: without losing it, which clearing the field would do. Anchor never
    #: grew the same need in v4 and does not get one here either.
    postfix: str = ""
    postfix_enabled: bool = True
    #: ponytail: three flat strings (plus one bool), not a dict keyed by role
    #: name -- an earlier version of this comment said to generalise once a
    #: second project-level text role showed up, and now three have. Tried
    #: it: postfix's own enabled flag does not fit a plain role->string map
    #: any better than a fourth field would, so the "simpler" shape turns out
    #: not to be. Three named fields stay more readable than an abstraction
    #: that has to special-case one of its own four entries anyway.
    #:
    #: $name -> text, substituted into anchor/scene/postfix at generation
    #: (core/prompt_build.py). A LIST, not a dict: order is what a person set,
    #: and preserving it is what makes "the row I just added" stay at the
    #: bottom instead of jumping around alphabetically on the next save.
    variables: list = field(default_factory=list)
    #: Saved prompts: the scenes + variables, reapplied in one pick.
    prompt_templates: list = field(default_factory=list)
    active_prompt_template: str = ""
    editor_settings: dict = field(default_factory=dict)
    updated_at: float = 0.0
    created_at: float = 0.0
    #: What every scene is generated at unless trimmed on the timeline.
    seed: int = 1
    num_frames_per_scene: int = 97
    frame_rate: int = 25
    width: int = 768
    height: int = 512
    max_scenes: int = 8
    #: The pre-roll marker between the anchor and the first scene, and the negative the editor edits.
    intro_transition: str = ""
    negative_prompt: str = ""
    conditioning_slot: str = "funpack"
    sampler_slot: str = "funpack"
    #: "i2v" or "t2v": decides what is EXPECTED (an anchor picture), never what is allowed.
    generation_mode: str = "i2v"
    studio_inputs: dict = field(default_factory=dict)
    sampler_inputs: dict = field(default_factory=dict)
    h3_references: list = field(default_factory=list)
    #: Media ids marked "R", in mark order (R1, R2, ...): the order IS the identity.
    references: list = field(default_factory=list)
    refinement_key: str = "default"
    #: Audio: keep each clip's own sound, plus lanes mixed over the montage.
    keep_original_audio: bool = True
    audio_tracks: list = field(default_factory=list)
    overlay_lanes: list = field(default_factory=list)
    overlay_tracks: list = field(default_factory=list)
    #: The pipeline the person configured (loader slots and linked inputs).
    models: dict = field(default_factory=lambda: {"slots": []})
    guide_settings: dict = field(default_factory=dict)
    continuity_settings: dict = field(default_factory=dict)
    generation_meta: dict = field(default_factory=dict)
    #: What each scene last rendered ({media, inSec, promptId}) and the removed scenes whose
    #: clip still previews: a reload shows the timeline the person left.
    scene_renders: dict = field(default_factory=dict)
    scene_ghosts: list = field(default_factory=list)
    #: Cut order (scene ids in TIMELINE order), empty = follow the plan; True once reordered by hand.
    timeline_order: list = field(default_factory=list)
    timeline_manually_ordered: bool = False

    @staticmethod
    def from_dict(d) -> "Project":
        d = d if isinstance(d, dict) else {}
        pid = d.get("id")
        raw = d.get("scenes")
        negative = d.get("negative")
        anchor = d.get("anchor")
        postfix = d.get("postfix")
        postfix_enabled = d.get("postfix_enabled")
        return Project(
            id=pid if is_id(pid) else _new_id(),
            name=_clean_name(d.get("name")),
            scenes=[Scene.from_dict(s) for s in (raw if isinstance(raw, list) else [])],
            video=_clean_video(d.get("video")),
            negative=negative if isinstance(negative, str) else "",
            anchor=anchor if isinstance(anchor, str) else "",
            postfix=postfix if isinstance(postfix, str) else "",
            # True unless the file explicitly says False -- absent (a project
            # from before this field existed) must read as "on", the same as
            # v4's own default, not as "off" because nothing was there to say.
            postfix_enabled=postfix_enabled if isinstance(postfix_enabled, bool) else True,
            variables=_clean_variables(d.get("variables")),
            prompt_templates=_clean_templates(d.get("prompt_templates")),
            active_prompt_template=str(d.get("active_prompt_template") or "")[:MAX_NAME],
            editor_settings=_clean_editor_settings(d.get("editor_settings")),
            updated_at=float(d.get("updated_at") or 0.0),
            created_at=_num(d.get("created_at"), 0.0, 0.0),
            seed=_int(d.get("seed"), 1, 0, 2 ** 63),
            num_frames_per_scene=_int(d.get("num_frames_per_scene"), 97, 1, MAX_SETTING),
            frame_rate=_int(d.get("frame_rate"), 25, 1, 1000),
            width=_int(d.get("width"), 768, 1, MAX_SETTING),
            height=_int(d.get("height"), 512, 1, MAX_SETTING),
            max_scenes=_int(d.get("max_scenes"), 8, 1, 10 ** 4),
            intro_transition=_str(d.get("intro_transition"), "", 200),
            negative_prompt=_str(d.get("negative_prompt")),
            conditioning_slot=_str(d.get("conditioning_slot"), "funpack", 120) or "funpack",
            sampler_slot=_str(d.get("sampler_slot"), "funpack", 120) or "funpack",
            generation_mode="t2v" if d.get("generation_mode") == "t2v" else "i2v",
            studio_inputs=_blob(d.get("studio_inputs"), dict),
            sampler_inputs=_blob(d.get("sampler_inputs"), dict),
            h3_references=_dicts(d.get("h3_references")),
            references=[r for r in _blob(d.get("references"), list) if isinstance(r, str) and r][:256],
            refinement_key=_str(d.get("refinement_key"), "default", 64) or "default",
            keep_original_audio=_bool(d.get("keep_original_audio"), True),
            audio_tracks=_dicts(d.get("audio_tracks")),
            overlay_lanes=_dicts(d.get("overlay_lanes")),
            overlay_tracks=_dicts(d.get("overlay_tracks")),
            models=_blob(d.get("models"), dict) or {"slots": []},
            guide_settings=_blob(d.get("guide_settings"), dict),
            continuity_settings=_blob(d.get("continuity_settings"), dict),
            generation_meta=_blob(d.get("generation_meta"), dict),
            scene_renders=_blob(d.get("scene_renders"), dict),
            scene_ghosts=_dicts(d.get("scene_ghosts")),
            timeline_order=[i for i in _blob(d.get("timeline_order"), list) if is_token(i)],
            timeline_manually_ordered=_bool(d.get("timeline_manually_ordered")),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _dir():
    config.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    return config.PROJECTS_DIR


def _path(project_id: str):
    # is_id, not a sanitiser: a name that merely survives cleaning can still be
    # "..", and a store that repairs a bad id quietly is one that writes
    # somewhere nobody asked for.
    if not is_id(project_id):
        raise ValueError(f"not a project id: {project_id!r}")
    return _dir() / f"{project_id}.json"


def listing() -> list[dict]:
    """Every project, newest first, as the little the picker needs."""
    out = []
    for p in _dir().glob("*.json"):
        if not is_id(p.stem):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue  # an unreadable file is not a reason to have no list
        if not isinstance(d, dict):
            continue
        out.append({
            "id": p.stem,
            "name": _clean_name(d.get("name"), p.stem),
            "scenes": len(d.get("scenes") or []),
            "updated_at": float(d.get("updated_at") or 0.0),
        })
    out.sort(key=lambda x: x["updated_at"], reverse=True)
    return out


def get(project_id: str) -> Project | None:
    try:
        path = _path(project_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        return Project.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def save(project: Project) -> Project:
    project.updated_at = time.time()
    path = _path(project.id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(project.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(path)  # a concurrent read never sees a half-written project
    return project


def create(name=None) -> Project:
    """A new project with one empty scene: a timeline with nothing on it cannot
    be typed into, so a brand-new project would need an Add before it could be
    used at all."""
    return save(Project(name=_clean_name(name), scenes=[Scene()]))


def delete(project_id: str) -> bool:
    try:
        path = _path(project_id)
    except ValueError:
        return False
    if not path.exists():
        return False
    path.unlink()
    return True
