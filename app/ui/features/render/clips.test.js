import test from "node:test";
import assert from "node:assert/strict";
import { clipSpecs } from "./clips.js";

const p = { num_frames_per_scene: 97, frame_rate: 25,
  scenes: [{ id: "a" }, { id: "b" }, { id: "c", excluded: true }, { id: "d", excluded: true, removed_from_plan: true, gap_after_sec: 1.5 }, { id: "v", source: { type: "video", media_ref: "m1" }, source_dur: 2, source_in: 1 }],
  scene_renders: { a: { media: { filename: "a.mp4", subfolder: "", type: "output" }, inSec: 0.5 }, c: { media: { filename: "c.mp4" } }, d: { media: { filename: "d.mp4" } } } };

test("playing clips come out in order; scenes with nothing to play are counted, left out ones are not (removed-from-plan ones still play)", () => {
  const { clips, missing } = clipSpecs(p);
  assert.deepEqual(clips.map((c) => c.scene_id), ["a", "d", "v"]);
  assert.equal(missing, 1);                              // b has no render
  assert.equal(clips[0].in, 0.5);
  assert.equal(clips[1].gap_after, 1.5, "a pause after a scene is the clip's gap_after");
  assert.deepEqual([clips[2].bin_media_ref, clips[2].in, clips[2].dur], ["m1", 1, 2]);
});

test("only the picked clips are kept, still in timeline order, and unpicked ones are not counted as missing", () => {
  const { clips, missing } = clipSpecs(p, new Set(["v", "a"]));
  assert.deepEqual(clips.map((c) => c.scene_id), ["a", "v"]);
  assert.equal(missing, 0);
});

test("a clip picked by hand is exported even when it is excluded from the full run", () => {
  assert.deepEqual(clipSpecs(p, new Set(["c"])).clips.map((c) => c.scene_id), ["c"]);
});

test("a clip whose sound was separated is quiet in the render; others keep their volume", () => {
  const q = { num_frames_per_scene: 50, frame_rate: 25, scenes: [{ id: "a", audio_separated: true, audio_volume: 0 }, { id: "b", audio_volume: 0.4 }, { id: "c" }],
    scene_renders: { a: { media: { filename: "a" } }, b: { media: { filename: "b" } }, c: { media: { filename: "c" } } } };
  assert.deepEqual(clipSpecs(q).clips.map((x) => x.volume), [0, 0.4, 1]);
});

test("a clip's effects and its seam transition reach the render spec (frames become seconds at the clip's fps)", () => {
  const q = { frame_rate: 25, num_frames_per_scene: 97, scenes: [{ id: "a", effects: { flip_h: true }, video_transition: "crossfade", transition_frames: 10 }, { id: "b" }],
    scene_renders: { a: { media: { filename: "a.mp4" } }, b: { media: { filename: "b.mp4" } } } };
  const [a, b] = clipSpecs(q).clips;
  assert.deepEqual([a.fx, a.transition, a.tdur], [{ flip_h: true }, "crossfade", 0.4]);
  assert.deepEqual([b.fx, b.transition, b.tdur], [{}, "", 0]);
});
