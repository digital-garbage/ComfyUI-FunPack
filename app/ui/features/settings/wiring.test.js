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

test("a field shows only in the mode it belongs to: every rule it names must hold, defaults count", async () => {
  const { showing, shownValues } = await import("./wiring.js");
  const spec = { widgets: [
    { name: "sla", type: "BOOLEAN", default: false },
    { name: "sla_method", type: "COMBO", choices: ["sla", "sol-attn", "vsa"], default: "sla", shows: [{ input: "sla", when: [true] }] },
    { name: "sla_sparsity", type: "FLOAT", default: 0.9, shows: [{ input: "sla", when: [true] }, { input: "sla_method", when: ["sla", "vsa"] }] },
    { name: "sla_tau", type: "FLOAT", default: 1.3, shows: [{ input: "sla", when: [true] }, { input: "sla_method", when: ["sol-attn"] }] },
  ] };
  const drawn = (inputs) => spec.widgets.filter((w) => showing(w, { inputs }, spec)).map((w) => w.name);
  assert.deepEqual(drawn({}), ["sla"]);
  assert.deepEqual(drawn({ sla: true }), ["sla", "sla_method", "sla_sparsity"]);
  assert.deepEqual(drawn({ sla: true, sla_method: "sol-attn" }), ["sla", "sla_method", "sla_tau"]);
  assert.deepEqual(shownValues({ inputs: { sla: true, sla_method: "sol-attn" } }, spec), { sla_tau: 1.3 });
});
