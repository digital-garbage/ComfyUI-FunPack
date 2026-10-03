import test from "node:test";
import assert from "node:assert/strict";
import { build, chop } from "./build.js";

const media = (n) => ({ filename: `${n}.mp4`, subfolder: "", type: "output" });
const project = () => ({ num_frames_per_scene: 97, frame_rate: 25,
  scenes: [{ id: "L", text: "lead" }, { id: "A", text: "a" }, { id: "B", text: "b" }, { id: "N", text: "no render" }],
  scene_renders: { L: { media: media("L") }, A: { media: media("A"), inSec: 1 }, B: { media: media("B") } } });

test("chop: segments shrink by the decay, never below 9 frames, and cover the whole length", () => {
  const segs = chop(97, 40, 0.5);
  assert.deepEqual(segs.map((s) => s.start), [0, 40, 60, 70, 79, 88]);
  assert.ok(segs.every((s) => s.frames >= 9));
  assert.equal(chop(97, 100, 1).length, 1);
});

test("build: lead pieces alternate with pool pieces, each pool clip is read front to back, renders are reused", () => {
  const p = project();
  const n = build(p, { leadId: "L", poolIds: ["A"], segmentFrames: 33, decay: 1, random: () => 0 });
  const added = p.scenes.slice(4);
  assert.equal(n, added.length);
  assert.equal(added.length % 2, 0);
  assert.deepEqual(added.filter((_, i) => i % 2 === 0).map((s) => s.gen_unit_id), added.filter((_, i) => i % 2 === 0).map(() => "L"));
  const fromA = added.filter((_, i) => i % 2 === 1);
  assert.ok(fromA.every((s) => s.gen_unit_id === "A"));
  assert.deepEqual(fromA.map((s) => s.cut_offset_frames).slice(0, 2), [0, 33], "A is read in order");
  assert.equal(p.scene_renders[fromA[1].id].inSec, 1 + 33 / 25, "a piece plays from where it starts in A's render");
  assert.equal(p.scene_renders[fromA[1].id].media.filename, "A.mp4");
  assert.equal(added[added.length - 1].transition_to_next, "");
  assert.ok(added.every((s) => s.frames_mode === "timeline" && s.source_in === 0));
  assert.equal(fromA[1].text, "", "a later part's prompt belongs to the first part");
});

test("build: nothing is built without a rendered lead and a rendered pool; the project is left alone", () => {
  const p = project(), before = JSON.stringify(p);
  assert.equal(build(p, { leadId: "N", poolIds: ["A"] }), 0);
  assert.equal(build(p, { leadId: "L", poolIds: ["N"] }), 0);
  assert.equal(JSON.stringify(p), before);
});
