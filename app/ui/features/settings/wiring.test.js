import test from "node:test";
import assert from "node:assert/strict";
import { labelOf, shownValues, sourcesFor } from "./wiring.js";

const specs = { A: { title: "Loader", outputs: ["MODEL", "VAE"], output_names: ["model", "vae"] }, B: { title: "Loader", outputs: ["INT"] }, C: { title: "Sampler", widgets: [{ name: "steps", type: "INT", default: 8 }, { name: "mode", type: "COMBO", choices: ["x", "y"] }, { name: "on", type: "BOOLEAN" }] } };
const slots = [{ id: "a", node: "A" }, { id: "b", node: "B" }, { id: "c", node: "C" }];

test("two slots of one title are told apart by id", () => {
  assert.equal(labelOf(slots[0], specs, slots), "Loader (a)");
  assert.equal(labelOf(slots[2], specs, slots), "Sampler");
});

test("only outputs of the right type, from other slots, can feed an input", () => {
  assert.deepEqual(sourcesFor(slots[2], "VAE", slots, specs).map((s) => s.label), ["Loader (a) · vae"]);
  assert.deepEqual(sourcesFor(slots[2], "INT", slots, specs).map((s) => s.value), ["b\u00000"]);
  assert.equal(sourcesFor(slots[0], "VAE", slots, specs).length, 0);       // not itself
});

test("what the form shows is sent for a slot with nothing set, and nothing already set is overwritten", () => {
  assert.deepEqual(shownValues({ id: "c", inputs: {} }, specs.C), { steps: 8, mode: "x", on: false });
  assert.deepEqual(shownValues({ id: "c", inputs: { steps: 4 } }, specs.C), { mode: "x", on: false });
  assert.deepEqual(shownValues({ id: "c", inputs: {} }, specs.C, "steps"), { steps: 8 });
});
