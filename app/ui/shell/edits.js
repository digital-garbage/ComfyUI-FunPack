// Changes to a project's scenes, as plain functions on the open project: `project.edit((p) => addScene(p))`.
// Each mutates in place and answers truthy when it changed something. They know what a scene IS (gen units,
// ghosts, the cut order) and nothing about the screen.
import { trackFor } from "./audio.js";
import {
  genUnitId, isVideoClip, isSubclip, effFps, effFrames, seconds, snapFrames, LTX_GRID,
} from "./scenes.js";

const clone = (x) => JSON.parse(JSON.stringify(x));
const newId = () => "c" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
const index = (p, id) => (p.scenes || []).findIndex((s) => s.id === id);

/** A fresh, empty scene on the end of the plan. */
export function addScene(p, sourceType = "carry") {
  const sc = { id: newId(), text: "", transition_to_next: "", source: { type: sourceType }, excluded: false, frames_mode: "project", fps_mode: "project" };
  p.scenes.push(sc);
  return sc;
}

// A scene that had a render leaves a ghost clip behind; one that never rendered just goes.
function ghostOrDrop(p, sc, afterId, pending) {
  const renders = p.scene_renders || (p.scene_renders = {});
  const r = renders[sc.id];
  delete renders[sc.id];
  p.scene_ghosts = (p.scene_ghosts || []).filter((g) => g.id !== sc.id);
  if (!(r && r.media) && !pending) return;
  p.scene_ghosts.push({
    id: sc.id, afterSceneId: afterId || null, gen_unit_id: genUnitId(sc), text: sc.text || "",
    frames: sc.frames, frames_mode: sc.frames_mode, fps: sc.fps, fps_mode: sc.fps_mode,
    effects: clone(sc.effects || {}), audio_volume: sc.audio_volume,
    media: (r && r.media) || null, inSec: (r && r.inSec) || 0, pendingGen: Boolean(pending && !(r && r.media)),
  });
}

/** Remove one scene. A root that owned a cut unit hands its prompt, source and offset to the next cut. */
export function removeScene(p, id, { pending = false } = {}) {
  const at = index(p, id);
  if (at < 0) return false;
  const sc = p.scenes[at];
  const unit = genUnitId(sc);
  const mates = p.scenes.filter((s) => genUnitId(s) === unit);
  if (!isSubclip(sc) && mates.length > 1) {
    const next = mates.find((s) => s.id !== id);
    next.cut_offset_frames = 0;
    next.text = sc.text || next.text;
    next.rating = sc.rating || next.rating;
    next.source = clone(sc.source || next.source || {});
    if (p.scene_variants && p.scene_variants[id]) { p.scene_variants[next.id] = p.scene_variants[id]; delete p.scene_variants[id]; }       // the takes follow the unit's new root
    const removedFrames = sc.frames || 0;
    mates.filter((s) => s.id !== id && (s.cut_offset_frames || 0) > (sc.cut_offset_frames || 0))
      .forEach((s) => { s.cut_offset_frames = Math.max(0, s.cut_offset_frames - removedFrames); });
  }
  p.audio_tracks = (p.audio_tracks || []).filter((t) => !(t.kind === "separated" && t.scene_id === id));       // its sound goes with it: a lane with no clip could not be reached
  const anchor = at > 0 ? p.scenes[at - 1].id : null;
  p.scene_ghosts = (p.scene_ghosts || []).map((g) => (g.afterSceneId === id ? { ...g, afterSceneId: anchor } : g));
  ghostOrDrop(p, sc, anchor, pending);
  p.scenes = p.scenes.filter((s) => s.id !== id);
  if (p.timeline_order) p.timeline_order = p.timeline_order.filter((x) => x !== id);
  return true;
}

/** Out of the plan but still on the timeline: a unit that has a render is kept, excluded from generation. */
export function removeFromPlan(p, id) {
  const sc = p.scenes.find((s) => s.id === id);
  if (!sc) return false;
  const unit = p.scenes.filter((s) => genUnitId(s) === genUnitId(sc));
  const rendered = isVideoClip(sc) || unit.some((s) => p.scene_renders && p.scene_renders[s.id] && p.scene_renders[s.id].media);
  if (!rendered) return removeScene(p, id);
  unit.forEach((s) => { s.excluded = true; s.removed_from_plan = true; });
  return true;
}

export function restoreToPlan(p, id) {
  const sc = p.scenes.find((s) => s.id === id);
  if (!sc) return false;
  p.scenes.filter((s) => genUnitId(s) === genUnitId(sc)).forEach((s) => { s.excluded = false; s.removed_from_plan = false; });
  return true;
}

/** Move a clip within the CUT (never the plan) to `to`, an index among the other clips. */
export function reorder(p, id, to, order) {
  const ids = order.slice();
  const from = ids.indexOf(id);
  if (from < 0) return false;
  ids.splice(from, 1);
  const at = Math.max(0, Math.min(ids.length, to));
  ids.splice(at, 0, id);
  if (ids.every((x, i) => x === order[i])) return false;
  p.timeline_order = ids;
  p.timeline_manually_ordered = true;     // from now on an edit to the plan must keep this order
  return true;
}

const modeToTimeline = (sc) => { if ((sc.frames_mode || "project") === "project") sc.frames_mode = "timeline"; };

/** Run for `sec` (from dragging an edge): whole grid steps, at least one step in the direction of the drag. */
export function resize(p, id, sec, grid = LTX_GRID) {
  const sc = p.scenes.find((s) => s.id === id);
  if (!sc || sc.frames_mode === "custom") return false;
  const fps = effFps(sc, p);
  const now = seconds(sc, p);
  const target = Math.max(0.1, sec);
  if (Math.abs(target - now) <= 0.02) return false;
  const cur = Math.max(1, Math.round(now * fps));
  if (isVideoClip(sc)) {
    sc.source_dur = target;
    sc.frames = snapFrames(target * fps, "round", grid);
  } else {
    const raw = target * fps;
    let frames = raw < cur - 0.01 ? snapFrames(raw, "floor", grid) : raw > cur + 0.01 ? snapFrames(raw, "ceil", grid) : cur;
    if (frames === cur) frames = raw < cur ? snapFrames(cur - grid.step, "floor", grid) : cur + grid.step;
    if (frames === cur) return false;
    sc.frames = frames;
  }
  modeToTimeline(sc);
  return true;
}

/** A separated clip's sound follows its picture's in-point (`shorten`: the head was cut off, so the sound loses its head too). */
function followLane(p, id, d, shorten) {
  const lane = trackFor(p, id);
  if (!lane) return;
  lane.pinned_in_sec = (lane.pinned_in_sec || 0) + d; lane.source_in_sec = lane.pinned_in_sec;
  if (shorten && lane.full_dur != null) lane.full_dur = Math.max(0.1, lane.full_dur - d);
  if (shorten && lane.pinned_dur != null) { lane.pinned_dur = Math.max(0.1, lane.pinned_dur - d); lane.source_dur = lane.pinned_dur; }
}

/** Cut `sec` off the start: the clip begins later in its source and runs shorter. */
export function trimLeft(p, id, sec, grid = LTX_GRID) {
  const sc = p.scenes.find((s) => s.id === id);
  if (!sc || sc.frames_mode === "custom") return false;
  const trim = Math.max(0.05, sec);
  if (isVideoClip(sc)) {
    const dur = sc.source_dur != null ? sc.source_dur : seconds(sc, p);
    if (trim >= dur - 0.05) return false;
    sc.source_in = (sc.source_in || 0) + trim;
    followLane(p, id, trim, true);
    sc.source_dur = Math.max(0.1, dur - trim);
    modeToTimeline(sc);
    return true;
  }
  const fps = effFps(sc, p);
  const cur = Math.max(1, Math.round(seconds(sc, p) * fps));
  const raw = cur - Math.max(1, Math.round(trim * fps));
  if (raw < grid.step + grid.base) return false;
  let next = snapFrames(raw, "floor", grid);
  if (next >= cur) next = Math.max(grid.step + grid.base, cur - grid.step);
  if (next >= cur) return false;
  sc.source_in = (sc.source_in || 0) + (cur - next) / fps;
  followLane(p, id, (cur - next) / fps, true);
  sc.frames = next;
  modeToTimeline(sc);
  return true;
}

/** Slide what plays within the clip without changing its length. */
export function slip(p, id, deltaSec) {
  const sc = p.scenes.find((s) => s.id === id);
  if (!sc || !deltaSec) return false;
  const before = sc.source_in || 0;
  sc.source_in = Math.max(0, before + deltaSec);
  followLane(p, id, sc.source_in - before, false);
  return true;
}

/** Cut a clip in two at `atFrames` (default the middle). Both halves stay one generated unit. */
export function split(p, id, atFrames, grid = LTX_GRID) {
  const at = index(p, id);
  if (at < 0) return false;
  const sc = p.scenes[at];
  const fps = effFps(sc, p);
  let frames = effFrames(sc, p);
  if (isVideoClip(sc) && sc.source_dur != null) frames = snapFrames(sc.source_dur * fps, "round", grid);
  const cut = snapFrames(atFrames != null ? atFrames : frames / 2, "round", grid);
  if (cut <= grid.step + grid.base || cut >= frames) return false;
  const cutSec = cut / fps;
  const unit = genUnitId(sc);
  const second = clone(sc);
  second.id = newId();
  second.gen_unit_id = unit;
  second.cut_offset_frames = (sc.cut_offset_frames || 0) + cut;
  second.frames = snapFrames(frames - cut, "round", grid);
  second.text = "";
  second.rating = "";
  if (sc.audio_separated) { const lane = trackFor(p, sc.id); second.audio_separated = false; second.audio_volume = lane && lane.volume != null ? lane.volume : 1; }       // the lane stays with the first half; the second plays its own sound
  sc.gen_unit_id = unit;
  // The seam between the halves is an internal hard cut: the outgoing edge now belongs to the second half.
  sc.frames = cut;
  sc.transition_to_next = ""; sc.transition_frames = null; sc.video_transition = "";
  modeToTimeline(sc); modeToTimeline(second);
  const playsFromSource = isVideoClip(sc) || (sc.source && sc.source.type === "v2v" && sc.source.media_ref);
  if (playsFromSource) {
    const srcIn = sc.source_in || 0;
    const total = sc.source_dur != null ? sc.source_dur : frames / fps;
    sc.source_dur = cutSec;
    second.source_in = srcIn + cutSec;
    second.source_dur = Math.max(0.1, Math.min(second.frames / fps, total - cutSec));
  }
  p.scenes.splice(at + 1, 0, second);
  if (p.timeline_manually_ordered && Array.isArray(p.timeline_order)) {
    const oi = p.timeline_order.indexOf(sc.id);
    if (oi >= 0) p.timeline_order.splice(oi + 1, 0, second.id);
  }
  // The render survives the cut: the second half plays the same video from where the first ends.
  const renders = p.scene_renders || {};
  const r = renders[id];
  if (r && r.media) {
    delete r.durationSec;                       // a split is an explicit re-length: both halves go back to plan layout
    renders[second.id] = { media: r.media, inSec: (r.inSec || 0) + cutSec, renderPrompt: r.renderPrompt ? { ...r.renderPrompt } : null, ...(r.promptId ? { promptId: r.promptId } : {}) };
  }
  return second;
}
