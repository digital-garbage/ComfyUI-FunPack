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
