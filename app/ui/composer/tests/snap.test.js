import test from "node:test";
import assert from "node:assert/strict";
import { snapDelta } from "../internals/snap.js";

test("an edge within reach lands on the anchor; one outside does not move", () => {
  assert.equal(snapDelta(0.95, [0], [1, 5], 0.125), 1);
  assert.equal(snapDelta(0.5, [0], [1, 5], 0.125), 0.5);
});

test("of several edges the nearest one decides; with no anchors nothing moves", () => {
  assert.ok(Math.abs(snapDelta(1.02, [0, 2], [3.0], 0.1) - 1.0) < 1e-9);       // the right edge (2) is 0.02 from 3
  assert.equal(snapDelta(0.7, [0], [], 0.1), 0.7);
});

test("an anchor the edge already sits on does not hold it there", () => {
  assert.equal(snapDelta(0.05, [2], [2], 0.1), 0.05);
});
