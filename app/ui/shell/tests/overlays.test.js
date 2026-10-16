import test from "node:test";
import assert from "node:assert/strict";
import * as ov from "../overlays.js";

test("the first overlay makes its lane; later ones go on the top lane", () => {
  const p = { width: 1000 };
  const a = ov.addText(p, 2, { text: " Hi " }), b = ov.addImage(p, "m1", "pic", 1);
  assert.equal(p.overlay_lanes.length, 1);
  assert.deepEqual([a.text, a.start_sec, a.lane_id === b.lane_id, b.width_px], ["Hi", 2, true, 350]);
  const lane = ov.addLane(p);
  assert.equal(ov.addText(p, 0).lane_id, lane.id);
});

test("trim in keeps the end, never before 0 nor past the end; out sets the length", () => {
  const p = {}, o = ov.addText(p, 1);
  ov.trim(p, o.id, "in", 0.5);
  assert.deepEqual([o.start_sec, o.duration_sec], [1.5, 2.5]);
  ov.trim(p, o.id, "in", -9);
  assert.deepEqual([o.start_sec, o.duration_sec], [0, 4]);
  ov.trim(p, o.id, "in", 99);
  assert.ok(Math.abs(o.duration_sec - 0.1) < 1e-9);
  ov.trim(p, o.id, "out", -5);
  assert.equal(o.duration_sec, 0.1);
});

test("move clamps at 0; restack walks lanes and stops at the edge; removing a lane takes its overlays", () => {
  const p = {}, o = ov.addText(p, 1), l2 = ov.addLane(p);
  ov.move(p, o.id, -5);
  assert.equal(o.start_sec, 0);
  assert.equal(ov.restack(p, o.id, 1), true);
  assert.equal(o.lane_id, l2.id);
  assert.equal(ov.restack(p, o.id, 1), false);
  ov.removeLane(p, l2.id);
  assert.equal(ov.find(p, o.id), undefined);
});

test("stack order: lower lane first, then earlier start; a lane-less overlay sits at the bottom", () => {
  const p = {}, a = ov.addText(p, 5), l2 = ov.addLane(p), b = ov.addText(p, 1), c = ov.addText(p, 0);
  p.overlay_tracks.push({ id: "orphan", lane_id: "gone", start_sec: 9 });
  assert.deepEqual(ov.inStackOrder(p).map((o) => o.id), [a.id, "orphan", c.id, b.id]);
  assert.equal(l2.id, b.lane_id);
});
