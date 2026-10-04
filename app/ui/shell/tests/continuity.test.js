import test from "node:test";
import assert from "node:assert/strict";
import { previousClip } from "../continuity.js";

const media = { filename: "a.mp4", subfolder: "", type: "output" };
const p = () => ({ num_frames_per_scene: 97, frame_rate: 25, scenes: [{ id: "a" }, { id: "b" }, { id: "c", excluded: true }, { id: "d" }, { id: "v", source: { type: "video", media_ref: "m" } }, { id: "e" }],
  scene_renders: { a: { media, inSec: 0.5 } } });

test("a shot continues from the clip right before its unit, if that one has a render", () => {
  const q = p();
  assert.equal(previousClip(q, q.scenes[0]).why, "it is the first clip");
  const got = previousClip(q, q.scenes[1]);
  assert.deepEqual([got.sceneId, got.render.inSec, Math.round(got.dur * 25)], ["a", 0.5, 97]);
  assert.match(previousClip(q, q.scenes[3]).why, /left out/);
  assert.match(previousClip(q, q.scenes[5]).why, /imported video/);
  assert.match(previousClip(q, q.scenes[4]).why, /no render/);
});

test("the second half of a cut unit looks past its own unit", () => {
  const q = p();
  q.scenes = [{ id: "a" }, { id: "b" }, { id: "b2", gen_unit_id: "b", cut_offset_frames: 40 }];
  assert.equal(previousClip(q, q.scenes[2]).sceneId, "a");
});
