// Bringing files from the media bin onto the timeline: a video as a clip of its own, a sound as an audio lane.
import { snapFrames } from "./scenes.js";

const newId = () => "c" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);

/** A video from the bin as a clip at the end of the cut. `seconds` is its length when known (else the project's own length until it is). */
export function addVideoClip(p, asset, seconds) {
  if (!asset || asset.kind !== "video") return false;
  const id = newId(), fps = p.frame_rate || 25, dur = seconds > 0 ? seconds : null;
  const sc = { id, gen_unit_id: id, text: "", transition_to_next: "", source: { type: "video", media_ref: asset.id }, source_dur: dur,
    frames_mode: dur != null ? "timeline" : "project", frames: dur != null ? snapFrames(dur * fps) : null, excluded: false };
  p.scenes.push(sc);
  return sc;
}

/** A sound from the bin as an audio lane starting at `startSec`. */
export function addAudioTrack(p, asset, startSec, seconds) {
  if (!asset || asset.kind !== "audio") return false;
  const lane = { id: "t" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36), kind: "overlay", media_ref: asset.id, start_sec: Math.max(0, startSec || 0),
    source_in_sec: 0, source_dur: seconds > 0 ? seconds : null, volume: 1, label: asset.name || "Audio" };
  p.audio_tracks = [...(p.audio_tracks || []), lane];
  return lane;
}
