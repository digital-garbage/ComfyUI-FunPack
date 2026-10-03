import test from "node:test";
import assert from "node:assert/strict";
import { hasEmbeddedAudio, moveLane, removeTrack, trimLane, separate, syncSeparated, trackFor } from "../audio.js";

const media = { filename: "a.mp4", subfolder: "", type: "output" };
const proj = () => ({ num_frames_per_scene: 50, frame_rate: 25, scenes: [{ id: "a", text: "" }, { id: "b", text: "", audio_volume: 0.5 }, { id: "v", source: { type: "video", media_ref: "m1" }, source_dur: 3 }],
  scene_renders: { a: { media, inSec: 0 }, b: { media, inSec: 1 } }, audio_tracks: [] });

test("only a clip with a render or a bin video has sound to pull out", () => {
  const p = proj();
  assert.equal(hasEmbeddedAudio(p.scenes[0], p), true);
  assert.equal(hasEmbeddedAudio(p.scenes[2], p), true);
  assert.equal(hasEmbeddedAudio({ id: "x" }, p), false);
  assert.equal(separate(p, "x"), false);
});

test("separate: the lane takes the clip's sound, place and volume; the clip goes quiet; a second time does nothing", () => {
  const p = proj();
  const lane = separate(p, "b");
  assert.deepEqual([lane.kind, lane.scene_id, lane.start_sec, lane.source_in_sec, lane.source_dur, lane.volume, lane.label], ["separated", "b", 2, 1, 2, 0.5, "S2 audio"]);
  assert.deepEqual(lane.pinned_media, media);
  assert.deepEqual([p.scenes[1].audio_separated, p.scenes[1].audio_volume], [true, 0]);
  assert.equal(separate(p, "b"), false);
  assert.equal(trackFor(p, "b"), lane);
});

test("a bin video clip's lane points at the bin file", () => {
  const p = proj();
  const lane = separate(p, "v");
  assert.deepEqual([lane.pinned_bin_ref, lane.pinned_media, lane.source_dur, lane.label], ["m1", null, 3, "V3 audio"]);
});

test("removing the lane gives the clip its sound and volume back", () => {
  const p = proj();
  const lane = separate(p, "b");
  assert.equal(removeTrack(p, lane.id), true);
  assert.deepEqual([p.scenes[1].audio_separated, p.scenes[1].audio_volume, p.audio_tracks.length], [false, 0.5, 0]);
  assert.equal(removeTrack(p, "nope"), false);
});

test("a separated lane follows its clip when the cut changes, and an untouched project is left alone", () => {
  const p = proj();
  const lane = separate(p, "b");
  p.scenes.unshift({ id: "n", text: "" });                 // a new first clip pushes everything 2 s later
  syncSeparated(p);
  assert.equal(lane.start_sec, 4);
  const before = JSON.stringify(p);
  syncSeparated(p);
  assert.equal(JSON.stringify(p), before);
});

import { removeScene, split } from "../edits.js";

test("splitting a separated clip leaves the lane with the first half, which it is cut to; the second half gets its own sound back", () => {
  const p = proj();
  p.num_frames_per_scene = 97;
  separate(p, "a");
  assert.ok(split(p, "a", 49));
  const lane = p.audio_tracks[0];
  syncSeparated(p);
  const [first, second] = [p.scenes[0], p.scenes[1]];
  assert.deepEqual([first.audio_separated, second.audio_separated, second.audio_volume], [true, false, 1]);
  assert.ok(lane.pinned_dur <= 49 / 25 + 0.01, "the lane is no longer than the first half");
});

test("removing a separated clip removes its lane (a lane with no clip could not be reached)", () => {
  const p = proj();
  separate(p, "b");
  removeScene(p, "b");
  assert.equal(p.audio_tracks.length, 0);
});

test("trimming or slipping a separated clip moves its sound's in-point with the picture", async () => {
  const { trimLeft, slip } = await import("../edits.js");
  const p = { scenes: [{ id: "v", source: { type: "video", media_ref: "x" }, source_in: 0, source_dur: 5, frames: 125 }], audio_tracks: [] };
  const lane = separate(p, "v");
  trimLeft(p, "v", 1);
  assert.equal(lane.pinned_in_sec, 1); assert.equal(lane.pinned_dur, 4);
  slip(p, "v", 0.5);
  assert.equal(lane.pinned_in_sec, 1.5); assert.equal(lane.pinned_dur, 4);
});

test("a shortened lane grows back with its picture, and a clip removed from the plan keeps its lane in step", () => {
  const p = proj();
  const lane = separate(p, "a");              // 2s clip, full 2
  p.num_frames_per_scene = 25; syncSeparated(p);
  assert.equal(lane.pinned_dur, 1);
  p.num_frames_per_scene = 50; syncSeparated(p);
  assert.equal(lane.pinned_dur, 2);
  Object.assign(p.scenes[0], { excluded: true, removed_from_plan: true }); p.scenes.unshift({ id: "z", text: "" }); syncSeparated(p);
  assert.ok(lane.start_sec > 0);
});

test("a lane can be slid against its clip, keeps that offset when the clip moves, and never starts before zero", () => {
  const p = proj();
  const lane = separate(p, "b");                       // clip b starts at 2s
  moveLane(p, lane.id, 0.5); syncSeparated(p);
  assert.equal(lane.start_sec, 2.5);
  p.scenes.unshift({ id: "z", text: "" }); syncSeparated(p);        // the clip moved to 4s: the lane keeps its 0.5s offset
  assert.equal(lane.start_sec, 4.5);
  moveLane(p, lane.id, -99); syncSeparated(p);
  assert.equal(lane.start_sec, 0);
});

test("trimming a lane cuts its sound, not its clip; the tail can grow back, never past the sound's own length", () => {
  const p = proj();
  const lane = separate(p, "b");                       // 2s of sound, in-point 1s
  trimLane(p, lane.id, "out", -0.5); syncSeparated(p);
  assert.equal(lane.pinned_dur, 1.5);
  trimLane(p, lane.id, "out", +5); syncSeparated(p);
  assert.equal(lane.pinned_dur, 2);
  trimLane(p, lane.id, "in", 0.5); syncSeparated(p);
  assert.deepEqual([lane.pinned_in_sec, lane.pinned_dur, lane.start_sec], [1.5, 1.5, 2.5]);      // the head is gone and the rest still starts where it sounded
  assert.equal(trimLane(p, lane.id, "in", 99) && lane.pinned_dur >= 0.1, true);
});

test("sliding past the timeline start is not owed on the way back; trimming in cannot reach before 0", () => {
  const p = proj(), lane = separate(p, "b");
  syncSeparated(p);
  moveLane(p, lane.id, -9);
  assert.equal(lane.offset_sec, -2);
  moveLane(p, lane.id, 1);
  assert.equal(lane.offset_sec, -1);
  const q = proj(), first = separate(q, "a");
  const before = first.pinned_in_sec;
  assert.equal(trimLane(q, first.id, "in", -0.5), false);
  assert.equal(first.pinned_in_sec, before);
});
