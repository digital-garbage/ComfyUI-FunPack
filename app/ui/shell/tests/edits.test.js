import test from "node:test";
import assert from "node:assert/strict";
import { addScene, removeScene, removeFromPlan, restoreToPlan, reorder, resize, trimLeft, slip, split } from "../edits.js";
import { seconds, segments } from "../scenes.js";

const mk = (extra = {}) => ({
  num_frames_per_scene: 97, frame_rate: 25, scene_renders: {}, scene_ghosts: [],
  scenes: [{ id: "a", text: "A", source: { type: "empty" } }, { id: "b", text: "B", source: { type: "carry" } }],
  ...extra,
});

test("a new scene goes on the end, empty", () => {
  const p = mk();
  const sc = addScene(p, "carry");
  assert.deepEqual([p.scenes.length, sc.text, sc.source.type], [3, "", "carry"]);
});

test("removing a scene with a render leaves a ghost where it was; one without just goes", () => {
  const p = mk({ scene_renders: { b: { media: { filename: "x.mp4" }, inSec: 2 } } });
  removeScene(p, "b");
  assert.deepEqual(p.scenes.map((s) => s.id), ["a"]);
  assert.deepEqual([p.scene_ghosts[0].id, p.scene_ghosts[0].afterSceneId, p.scene_ghosts[0].inSec], ["b", "a", 2]);
  removeScene(p, "a");
  assert.equal(p.scene_ghosts.length, 1, "a scene that never rendered leaves nothing");
});

test("removing the root of a cut hands its prompt and source to the next cut", () => {
  const p = mk();
  const second = split(p, "a", 49);
  assert.ok(second);
  removeScene(p, "a");
  const left = p.scenes.find((s) => s.id === second.id);
  assert.deepEqual([left.cut_offset_frames, left.text, left.source.type], [0, "A", "empty"]);
});

test("a unit with a render stays on the timeline when taken out of the plan, and can come back", () => {
  const p = mk({ scene_renders: { a: { media: { filename: "x.mp4" } } } });
  removeFromPlan(p, "a");
  assert.deepEqual([p.scenes[0].excluded, p.scenes[0].removed_from_plan], [true, true]);
  restoreToPlan(p, "a");
  assert.deepEqual([p.scenes[0].excluded, p.scenes[0].removed_from_plan], [false, false]);
});

test("reordering the cut pins an order and never touches the plan", () => {
  const p = mk();
  assert.equal(reorder(p, "a", 1, ["a", "b"]), true);
  assert.deepEqual(p.timeline_order, ["b", "a"]);
  assert.equal(p.timeline_manually_ordered, true);
  assert.deepEqual(p.scenes.map((s) => s.id), ["a", "b"]);
  assert.equal(reorder(p, "b", 0, ["b", "a"]), false, "dropping where it already is changes nothing");
});

test("dragging an edge lands on the model's frame grid and pins the scene's own length", () => {
  const p = mk();
  assert.equal(resize(p, "a", 2), true);
  const a = p.scenes[0];
  assert.equal(a.frames_mode, "timeline");
  assert.equal(a.frames % 8, 1, "8k+1");
  assert.ok(Math.abs(seconds(a, p) - 2) < 0.3);
  assert.equal(resize(p, "a", seconds(a, p) + 0.01), false, "a nudge under the tolerance is nothing");
});

test("trimming the start begins later in the source and runs shorter", () => {
  const p = mk();
  assert.equal(trimLeft(p, "a", 1), true);
  const a = p.scenes[0];
  assert.ok(a.source_in > 0.5 && a.frames < 97);
  assert.equal(slip(p, "a", -100), true);
  assert.equal(a.source_in, 0, "never before the start of the source");
});

test("a split makes two clips of one unit that keep playing the same render", () => {
  const p = mk({ scene_renders: { a: { media: { filename: "x.mp4" }, inSec: 1, durationSec: 4 } } });
  const second = split(p, "a", 49);
  assert.equal(p.scenes.length, 3);
  assert.deepEqual([p.scenes[0].frames, second.frames, second.gen_unit_id, second.cut_offset_frames], [49, 49, "a", 49]);
  assert.equal(p.scene_renders[second.id].inSec, 1 + 49 / 25);
  assert.equal(p.scene_renders.a.durationSec, undefined);
  assert.deepEqual(segments(p).map((s) => s.id).slice(0, 2), ["a", second.id]);
  assert.equal(split(p, "a", 5), false, "no cut inside the first grid step");
});

test("removing the first cut of a unit hands its takes to the clip that becomes the root", async () => {
  const { removeScene } = await import("../edits.js");
  const p = { scenes: [{ id: "a", text: "x", frames: 8 }, { id: "b", gen_unit_id: "a", cut_offset_frames: 8, frames: 8 }], scene_variants: { a: [{ promptId: "p" }] }, scene_renders: {} };
  removeScene(p, "a");
  assert.deepEqual(Object.keys(p.scene_variants), ["b"]);
});
