import test from "node:test";
import assert from "node:assert/strict";
import { build, chop } from "./build.js";
import { isGenerative, genUnitId } from "../../shell/scenes.js";
import { removeScene } from "../../shell/edits.js";

const media = (n) => ({ filename: `${n}.mp4`, subfolder: "", type: "output" });
const project = () => ({ num_frames_per_scene: 97, frame_rate: 25,
  scenes: [{ id: "L", text: "lead" }, { id: "A", text: "a" }, { id: "B", text: "b" }, { id: "N", text: "no render" }],
  scene_renders: { L: { media: media("L") }, A: { media: media("A"), inSec: 1 }, B: { media: media("B") } } });

test("chop: segments shrink by the decay, never below 9 frames, and cover the whole length", () => {
  const segs = chop(97, 40, 0.5);
  assert.ok(segs.every((s) => s.frames >= 9));
  assert.ok(segs.every((s, i) => i === 0 || s.start === segs[i - 1].start + segs[i - 1].frames), "pieces butt up exactly: no overlap, no gap");
  assert.ok(segs.every((s) => s.start + s.frames <= 97), "no piece reaches past the clip");
  assert.equal(chop(97, 100, 1).length, 1);
});

test("build: lead pieces alternate with pool pieces, each pool clip is read front to back, renders are reused", () => {
  const p = project();
  const n = build(p, { leadId: "L", poolIds: ["A"], segmentFrames: 33, decay: 1, random: () => 0 });
  const added = p.scenes.slice(4);
  assert.equal(n, added.length);
  assert.equal(added.length % 2, 0);
  assert.ok(added.every((s) => s.reused && s.gen_unit_id === s.id && !isGenerative(s)), "every piece is its own unit that Generate never makes");
  assert.ok(!p.scenes.slice(0, 4).some((s) => genUnitId(s) !== s.id), "the originals gain no unit mates, so Generate cannot lengthen them");
  const fromL = added.filter((_, i) => i % 2 === 0);
  assert.deepEqual(fromL.map((s) => p.scene_renders[s.id].inSec).slice(0, 2), [0, 33 / 25], "the lead plays in order");
  const fromA = added.filter((_, i) => i % 2 === 1);
  assert.deepEqual(fromA.map((s) => p.scene_renders[s.id].inSec).slice(0, 2), [1, 1 + 33 / 25], "A is read in order");
  assert.equal(p.scene_renders[fromA[1].id].inSec, 1 + 33 / 25, "a piece plays from where it starts in A's render");
  assert.equal(p.scene_renders[fromA[1].id].media.filename, "A.mp4");
  assert.equal(added[added.length - 1].transition_to_next, "");
  assert.ok(added.every((s) => s.frames_mode === "timeline" && s.source_in === 0));

  const before = added.map((s) => [s.id, p.scene_renders[s.id].inSec]);
  p.scenes.find((s) => s.id === "L").frames_mode = "timeline";
  removeScene(p, "L");
  assert.deepEqual(added.map((s) => [s.id, p.scene_renders[s.id].inSec]), before, "deleting the original leaves the montage as it was");
});

test("build: nothing is built without a rendered lead and a rendered pool; the project is left alone", () => {
  const p = project(), before = JSON.stringify(p);
  assert.equal(build(p, { leadId: "N", poolIds: ["A"] }), 0);
  assert.equal(build(p, { leadId: "L", poolIds: ["N"] }), 0);
  assert.equal(JSON.stringify(p), before);
});

test("build: deleting a piece leaves no ghost, and an off-grid clip is never read past its end", () => {
  const p = project();
  p.scene_renders.L = { media: media("L") };
  p.scenes[0].frames = 103; p.scenes[0].frames_mode = "timeline";           // 103 is off the 8k+1 grid; its nearest grid length, 105, is past the end
  build(p, { leadId: "L", poolIds: ["A"], segmentFrames: 33, decay: 1, random: () => 0 });
  const lastLead = p.scenes.filter((s) => s.reused).filter((_, i) => i % 2 === 0).pop();
  assert.ok(p.scene_renders[lastLead.id].inSec * 25 + lastLead.frames <= 103, "no piece plays past the clip");
  const piece = p.scenes.find((s) => s.reused);
  removeScene(p, piece.id);
  assert.equal((p.scene_ghosts || []).some((g) => g.id === piece.id), false);
});
