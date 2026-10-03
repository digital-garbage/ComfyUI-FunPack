import test from "node:test";
import assert from "node:assert/strict";
import { seconds, planSeconds, snapFrames, segments, orderedScenes, totalSeconds, clock, genUnitId, LTX_GRID } from "../scenes.js";

const P = { num_frames_per_scene: 97, frame_rate: 25, scenes: [{ id: "a" }, { id: "b" }, { id: "c", gap_after_sec: 1 }], scene_renders: {} };

test("a scene follows the project's length until trimmed", () => {
  assert.equal(planSeconds(P.scenes[0], P), 97 / 25);
  assert.equal(planSeconds({ id: "x", frames_mode: "timeline", frames: 49 }, P), 49 / 25);
});

test("a render keeps the length it was made at when the project changes", () => {
  const p = { ...P, scene_renders: { a: { durationSec: 5 } } };
  assert.equal(seconds(p.scenes[0], p), 5);
  assert.equal(seconds({ id: "a", frames_mode: "timeline", frames: 25 }, p), 1, "an explicit trim wins over the frozen length");
});

test("frames snap onto the model's grid, never below one step", () => {
  assert.equal(snapFrames(100), 97);
  assert.equal(snapFrames(100, "ceil"), 105);
  assert.equal(snapFrames(1), 9);
  assert.equal(snapFrames(40, "round", { step: 17, base: 5 }), 39);
});

test("the cut follows the plan until a person orders it, then keeps their order and heals around deletions", () => {
  assert.deepEqual(orderedScenes(P).map((s) => s.id), ["a", "b", "c"]);
  const p = { ...P, timeline_manually_ordered: true, timeline_order: ["c", "gone", "a"] };
  assert.deepEqual(orderedScenes(p).map((s) => s.id), ["c", "a", "b"]);
});

test("segments lay scenes, ghosts and pauses end to end", () => {
  const p = { ...P, scene_ghosts: [{ id: "g", afterSceneId: "a", durationSec: 2 }] };
  const segs = segments(p);
  assert.deepEqual(segs.map((s) => s.id), ["a", "ghost:g", "b", "c", "gap:c"]);
  assert.equal(segs[2].start, 97 / 25 + 2);
  assert.equal(totalSeconds(p), 3 * (97 / 25) + 2 + 1);
});

test("a gen unit is its own id unless a cut says otherwise", () => {
  assert.equal(genUnitId({ id: "a" }), "a");
  assert.equal(genUnitId({ id: "b", gen_unit_id: "a" }), "a");
  assert.equal(clock(125), "02:05");
  assert.deepEqual(LTX_GRID, { step: 8, base: 1, fps: null });
});
