import test from "node:test";
import assert from "node:assert/strict";
import { takesOf, stepTake } from "../takes.js";

const proj = () => ({
  scenes: [{ id: "a", rating: "10" }, { id: "b", gen_unit_id: "a", cut_offset_frames: 8 }],
  scene_renders: { a: { media: { filename: "3.mp4" }, promptId: "p3", inSec: 0 }, b: { media: { filename: "3.mp4" }, promptId: "p3", inSec: 2 } },
  scene_variants: { a: [{ media: { filename: "1.mp4" }, promptId: "p1", rating: "1" }, { media: { filename: "2.mp4" }, promptId: "p2", rating: "" }, { media: { filename: "3.mp4" }, promptId: "p3", rating: "" }] },
});

test("the clip's render is found among its takes, from any clip of the unit", () => {
  const p = proj();
  assert.equal(takesOf(p, p.scenes[1]).at, 2);
  assert.equal(takesOf(p, p.scenes[1]).head.id, "a");
});

test("stepping puts another take on every clip of the unit, keeps each clip's own cut, and carries the ratings", () => {
  const p = proj();
  assert.equal(stepTake(p, p.scenes[0], -3), false);                  // before the first take
  assert.equal(stepTake(p, p.scenes[0], -1), true);
  assert.equal(p.scene_renders.a.media.filename, "2.mp4");
  assert.equal(p.scene_renders.b.media.filename, "2.mp4");
  assert.equal(p.scene_renders.b.inSec, 2, "the cut half still plays from where it begins");
  assert.equal(p.scene_variants.a[2].rating, "10", "the take that left keeps what it was rated");
  assert.equal(stepTake(p, p.scenes[0], -1), true);
  assert.equal(p.scenes[0].rating, "1", "the take that arrived brings its rating");
  assert.equal(stepTake(p, p.scenes[0], -1), false, "nothing before the first");
});

test("a clip whose render is not one of its takes cannot be stepped", () => {
  const p = proj();
  p.scene_renders.a.promptId = "other";
  assert.equal(stepTake(p, p.scenes[0], 1), false);
});
