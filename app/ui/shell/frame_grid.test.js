import test from "node:test";
import assert from "node:assert/strict";
import { gridOf, learnGrid } from "./frame_grid.js";

test("the frame grid is read from the model's own length input: H3 17k+5, LTX 8k+1, a plain count none", async () => {
  const slots = [{ id: "r2v", node: "H3Node", roles: [{ at: "project.video", input: "length", drives: "frames" }] },
    { id: "cam", node: "Plain", roles: [{ at: "project.video", input: "length", drives: "frames" }] }];
  const api = { describeNodes: async () => ({ nodes: { H3Node: { widgets: [{ name: "length", min: 5, step: 17 }] }, Plain: { widgets: [{ name: "length", min: 1, step: 1 }] } } }) };
  assert.equal(gridOf(slots), null, "not described yet: no grid, nothing snaps");
  assert.equal(await learnGrid(api, slots), true);
  assert.deepEqual(gridOf(slots), { step: 17, base: 5, fps: null }, "the strictest input wins");
  assert.equal(await learnGrid(api, slots), false, "asked once");
  const ltx = [{ node: "LtxNode", roles: [{ drives: "frames", input: "length" }] }];
  await learnGrid({ describeNodes: async () => ({ nodes: { LtxNode: { widgets: [{ name: "length", min: 1, step: 8 }] } } }) }, ltx);
  assert.deepEqual(gridOf(ltx), { step: 8, base: 1, fps: null });
});
