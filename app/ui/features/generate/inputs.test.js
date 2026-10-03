import test from "node:test";
import assert from "node:assert/strict";
import { buildInputs } from "./inputs.js";

const slots = [
  { id: "p", roles: [{ at: "generation.prompt", input: "text" }] },
  { id: "s", roles: [{ at: "assets.source_image" }] },
  { id: "k", roles: [{ at: "project.video", input: "steps" }, { at: "project.negative", input: "neg" }] },
];

test("each value lands on the slot whose role asks for it", async () => {
  const { inputs } = await buildInputs({ project: { video: { steps: 8 }, negative: "blur" }, scene: { text: "a cat", source_image: "m1" }, slots,
    expand: async ({ text }) => ({ text: `[${text}]` }) });
  assert.deepEqual(inputs, { p: { text: "[a cat]" }, s: { media_id: "m1" }, k: { steps: 8, neg: "blur" } });
});

test("a failing expander sends the typed text", async () => {
  const { inputs } = await buildInputs({ project: {}, scene: { text: "x" }, slots, expand: async () => { throw new Error("down"); } });
  assert.equal(inputs.p.text, "x");
});

test("a role that drives length is fed from the project number the timeline draws", async () => {
  const s = [{ id: "k", roles: [{ at: "project.video", input: "length", drives: "frames" }, { at: "project.video", input: "width" }] }];
  const { inputs } = await buildInputs({ project: { num_frames_per_scene: 49, video: { length: 97, width: 640 } }, scene: {}, slots: s, expand: async () => null });
  assert.deepEqual(inputs.k, { length: 49, width: 640 });
});

test("no prompt role is reported, not silently ignored", async () => {
  const { noPrompt } = await buildInputs({ project: {}, scene: { text: "x" }, slots: [], expand: async () => null });
  assert.equal(noPrompt, true);
});

test("each build draws a fresh seed, and a unit's own length overrides the project's", async () => {
  const s = [{ id: "n", roles: [{ at: "generation.seed", input: "seed" }, { at: "project.video", input: "length", drives: "frames" }] }];
  const args = { project: { num_frames_per_scene: 97 }, scene: {}, slots: s, expand: async () => null, frames: 49 };
  const a = await buildInputs({ ...args, seed: () => 1 }), b = await buildInputs({ ...args, seed: () => 2 });
  assert.deepEqual([a.inputs.n.seed, b.inputs.n.seed, a.inputs.n.length], [1, 2, 49]);
});

test("starting shots from a prompt sends no picture even when the scene has one", async () => {
  const { inputs } = await buildInputs({ project: { generation_mode: "t2v" }, scene: { text: "a cat", source_image: "m1" }, slots, expand: async ({ text }) => ({ text }) });
  assert.equal(inputs.s.media_id, "");
});

test("a hook's inputs are merged and its notes come back", async () => {
  const hook = ({ scene }) => ({ inputs: { k: { extra: scene.text } }, notes: ["said"] });
  const r = await buildInputs({ project: {}, scene: { text: "x" }, slots, hooks: [hook, () => null], expand: async () => null });
  assert.equal(r.inputs.k.extra, "x");
  assert.deepEqual(r.notes, ["said"]);
});
