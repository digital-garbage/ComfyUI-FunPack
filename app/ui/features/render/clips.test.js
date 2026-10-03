import test from "node:test";
import assert from "node:assert/strict";
import { clipSpecs } from "./clips.js";

const p = { num_frames_per_scene: 97, frame_rate: 25,
  scenes: [{ id: "a" }, { id: "b" }, { id: "c", excluded: true }, { id: "v", source: { type: "video", media_ref: "m1" }, source_dur: 2, source_in: 1 }],
  scene_renders: { a: { media: { filename: "a.mp4", subfolder: "", type: "output" }, inSec: 0.5 }, c: { media: { filename: "c.mp4" } } } };

test("playing clips come out in order; scenes with nothing to play are counted, left-out ones are not", () => {
  const { clips, missing } = clipSpecs(p);
  assert.deepEqual(clips.map((c) => c.scene_id), ["a", "v"]);
  assert.equal(missing, 1);                              // b has no render
  assert.equal(clips[0].in, 0.5);
  assert.deepEqual([clips[1].bin_media_ref, clips[1].in, clips[1].dur], ["m1", 1, 2]);
});
