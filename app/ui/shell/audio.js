// Audio lanes. A clip's own sound can be pulled onto a lane of its own ("separated"): the lane plays the sound pinned from
// that clip's picture, and the clip itself goes quiet. A separated lane follows its clip around the timeline.
import { segments, isVideoClip } from "./scenes.js";

const newId = () => "t" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);

export const trackFor = (p, sceneId) => (p.audio_tracks || []).find((t) => t.kind === "separated" && t.scene_id === sceneId);

/** Does this clip have sound to pull out: a bin video clip, or a generated clip with a render. */
export const hasEmbeddedAudio = (sc, p) => {
  if (!sc || sc.excluded || sc.audio_separated) return false;
  if (isVideoClip(sc)) return Boolean(sc.source && sc.source.media_ref);
  return Boolean(((p.scene_renders || {})[sc.id] || {}).media);
};

/** Pull a clip's sound onto its own lane. -> the lane, or false (nothing to pull, or already pulled). */
export function separate(p, sceneId) {
  const at = (p.scenes || []).findIndex((s) => s.id === sceneId), sc = p.scenes[at];
  if (!hasEmbeddedAudio(sc, p) || trackFor(p, sceneId)) return false;
  const seg = segments(p).find((s) => s.kind === "scene" && s.id === sceneId);
  const dur = sc.source_dur != null ? sc.source_dur : seg.dur;
  const render = (p.scene_renders || {})[sceneId];
  const video = isVideoClip(sc);
  const inSec = video ? sc.source_in || 0 : (render.inSec || 0) + (sc.source_in || 0);
  const lane = { id: newId(), kind: "separated", scene_id: sceneId, start_sec: seg.start, source_in_sec: inSec, source_dur: dur,
    pinned_media: video ? null : JSON.parse(JSON.stringify(render.media)), pinned_bin_ref: video ? sc.source.media_ref : null,
    pinned_in_sec: inSec, pinned_dur: dur, full_dur: dur, volume: sc.audio_volume != null ? sc.audio_volume : 1, label: `${video ? "V" : "S"}${at + 1} audio` };
  p.audio_tracks = [...(p.audio_tracks || []), lane];
  sc.audio_separated = true;
  sc.audio_volume = 0;
  return lane;
}

/** Take a lane away; a separated one gives its clip its sound (and volume) back. */
export function removeTrack(p, id) {
  const t = (p.audio_tracks || []).find((x) => x.id === id);
  if (!t) return false;
  const sc = t.kind === "separated" && (p.scenes || []).find((s) => s.id === t.scene_id);
  if (sc) { sc.audio_separated = false; if (t.volume != null) sc.audio_volume = t.volume; }
  p.audio_tracks = p.audio_tracks.filter((x) => x.id !== id);
  return true;
}

/** A separated lane sits where its clip sits. Run after every edit; changes nothing when nothing moved. */
export function syncSeparated(p) {
  const tracks = p.audio_tracks || [];
  if (!tracks.some((t) => t.kind === "separated")) return;
  const segs = segments(p).filter((s) => s.kind === "scene");
  const starts = new Map(segs.map((s) => [s.id, s.start])), durs = new Map(segs.map((s) => [s.id, s.dur]));
  for (const t of tracks) {
    if (t.kind !== "separated" || !starts.has(t.scene_id)) continue;
    const sc = p.scenes.find((s) => s.id === t.scene_id);
    if (sc.excluded && !sc.removed_from_plan) continue;       // a clip removed from the plan still plays, so its sound still follows it
    const offset = Math.max(-starts.get(t.scene_id), t.offset_sec || 0), start = starts.get(t.scene_id) + offset;       // where the person dragged it, relative to its clip
    if (Math.abs((t.start_sec || 0) - start) > 0.001) t.start_sec = start;
    const room = durs.get(t.scene_id) - offset;
    if (t.pinned_dur == null) continue;
    const own = Math.min(t.user_dur != null ? t.user_dur : Infinity, t.full_dur != null ? t.full_dur : t.pinned_dur);
    const want = Math.max(0.1, Math.min(own, room));       // never past the end of its picture, and back to its full length when the picture grows again
    if (Math.abs(t.pinned_dur - want) > 0.001) { t.pinned_dur = want; t.source_dur = want; }
  }
}

/** Slide a separated lane against its clip (it still follows the clip when the clip moves). */
export function moveLane(p, id, deltaSec) {
  const t = (p.audio_tracks || []).find((x) => x.id === id);
  if (!t || !deltaSec) return false;
  if (t.kind !== "separated") {       // a file laid over the cut: it simply starts later or earlier
    const was = t.start_sec || 0;
    t.start_sec = Math.max(0, was + deltaSec);
    return t.start_sec !== was;
  }
  const seg = segments(p).find((s) => s.kind === "scene" && s.id === t.scene_id);
  const was = t.offset_sec || 0;
  t.offset_sec = Math.max(seg ? -seg.start : -Infinity, was + deltaSec);       // never before 0: a slide past the edge must not be owed on the way back
  return t.offset_sec !== was;
}

/** Trim a separated lane's sound. "in" cuts the head off (positive) or gives it back (negative, never before the sound begins);
 *  "out" sets the tail (never past the sound's own length). */
export function trimLane(p, id, edge, deltaSec) {
  const t = (p.audio_tracks || []).find((x) => x.id === id);
  if (!t || !deltaSec) return false;
  if (t.kind !== "separated") {       // a file laid over the cut: its own in-point and length
    const dur = t.source_dur, inAt = t.source_in_sec || 0;
    if (dur == null) return false;
    if (edge === "out") { t.source_dur = Math.max(0.1, dur + deltaSec); return true; }
    const d = Math.min(Math.max(deltaSec, -inAt, -(t.start_sec || 0)), dur - 0.1);
    if (!d) return false;
    t.source_in_sec = inAt + d; t.source_dur = dur - d; t.start_sec = (t.start_sec || 0) + d;
    return true;
  }
  if (t.pinned_dur == null) return false;
  const full = t.full_dur != null ? t.full_dur : t.pinned_dur;
  if (edge === "out") { t.user_dur = Math.max(0.1, Math.min(full, t.pinned_dur + deltaSec)); return true; }
  const d = Math.min(Math.max(deltaSec, -(t.pinned_in_sec || 0), -(t.start_sec || 0)), t.pinned_dur - 0.1);       // nor earlier than the timeline's start
  if (!d) return false;
  t.pinned_in_sec = (t.pinned_in_sec || 0) + d; t.source_in_sec = t.pinned_in_sec;
  t.full_dur = Math.max(0.1, full - d);
  if (t.user_dur != null) t.user_dur = Math.max(0.1, t.user_dur - d);
  t.offset_sec = (t.offset_sec || 0) + d;
  return true;
}

/** Cut a file lane in two at `atSec` (timeline seconds); both halves keep playing what they did. -> the new (later) lane, or false. */
export function splitTrack(p, id, atSec) {
  const t = (p.audio_tracks || []).find((x) => x.id === id);
  if (!t || t.kind === "separated" || t.source_dur == null) return false;
  const start = t.start_sec || 0, cut = atSec - start;
  if (!(cut > 0.05 && cut < t.source_dur - 0.05)) return false;
  const tail = { ...t, id: newId(), start_sec: atSec, source_in_sec: (t.source_in_sec || 0) + cut, source_dur: t.source_dur - cut };
  t.source_dur = cut;
  p.audio_tracks = [...p.audio_tracks, tail];
  return tail;
}
