import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let models;
test.before(async () => { setupDom(); ({ models } = await import("./models.js")); });
test.after(() => teardownDom());

test("Revert goes back to the pipeline of the project now open, not the one open when the page was", async () => {
  let live = "A", heard, restored;
  const ps = { slots: () => [{ id: "model", node: "L", inputs: { file: live } }], snapshot: () => ({ slots: [{ id: "model", node: "L", inputs: { file: live } }] }),
    ensureLoaded: async () => {}, subscribe: (fn) => { heard = fn; return () => {}; }, restore: async (snap) => { restored = snap.slots[0].inputs.file; return { refused: [] }; },
    loading: () => false, loadError: () => null, saveNotes: () => [], incomplete: () => [], refused: () => [], queueable: () => true, removedIds: () => [], offered: () => [] };
  const page = models({ pipeline: ps, api: { describeNodes: async () => ({ nodes: { L: { widgets: [] } } }) } })();
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 0));
  live = "B";
  heard(ps.slots(), "load");
  await new Promise((r) => setTimeout(r, 0));
  [...page.node.querySelectorAll("button")].find((b) => b.textContent.includes("Revert")).click();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(restored, "B");
});

test("a node the server says needs something shows the orange dot, whichever way the message names it", async () => {
  const slots = [{ id: "model", node: "L", inputs: { ckpt_name: "" } }, { id: "s", node: "S", inputs: {}, roles: [{ at: "project.video", input: "amount", label: "Length" }] }];
  const ps = { slots: () => slots, snapshot: () => ({ slots }), ensureLoaded: async () => {}, subscribe: () => () => {}, restore: async () => ({ refused: [] }),
    loading: () => false, loadError: () => null, saveNotes: () => [], incomplete: () => ["model.ckpt_name is '', which is not one of: a", "Length is 5000, above the largest 4096 it takes"],
    refused: () => [], queueable: () => false, removedIds: () => [], offered: () => [] };
  const page = models({ pipeline: ps, api: { describeNodes: async () => ({ nodes: { L: { widgets: [] }, S: { widgets: [] } } }) } })();
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 0));
  const dots = [...page.node.querySelectorAll(".fp-node-state")];
  assert.equal(dots.length, 2);
  assert.ok(dots.every((d) => d.classList.contains("fp-needs")));
});

test("a file input nothing has chosen shows “choose”, not the first file, and the one file can be picked", async () => {
  const { widgetControl } = await import("./models.js");
  const picked = [];
  const ctl = widgetControl({ name: "model_name", type: "COMBO", choices: ["only.safetensors"] }, undefined, (v) => picked.push(v));
  const select = ctl.node.querySelector("select") || ctl.node;
  assert.equal(select.value, "");
  select.value = "only.safetensors";
  select.dispatchEvent(new Event("change", { bubbles: true }));
  assert.deepEqual(picked, ["only.safetensors"]);
});

test("the simple view shows only nodes that ask for a choice (files, sampler, broken); “Show all nodes” brings back the rest", async () => {
  globalThis.localStorage = window.localStorage; localStorage.clear();
  const slots = [{ id: "m", node: "FunPackDiffusionModelLoader", group: "Loaders", inputs: { model_name: "h3.safetensors" } },
    { id: "k", node: "FunPackSampler", group: "Sampling", inputs: { steps: 8 } },
    { id: "pm", node: "FunPackPromptMarkup", group: "Prompt", inputs: { text: "" } }];
  const ps = { slots: () => slots, snapshot: () => ({ slots }), ensureLoaded: async () => {}, subscribe: () => () => {}, restore: async () => ({ refused: [] }),
    loading: () => false, loadError: () => null, saveNotes: () => [], incomplete: () => [], refused: () => [], queueable: () => true, removedIds: () => [], offered: () => [] };
  const nodes = { FunPackDiffusionModelLoader: { widgets: [{ name: "model_name", type: "COMBO", choices: ["h3.safetensors"] }] },
    FunPackSampler: { widgets: [{ name: "steps", type: "INT" }] }, FunPackPromptMarkup: { widgets: [{ name: "text", type: "STRING" }] } };
  const page = models({ pipeline: ps, api: { describeNodes: async () => ({ nodes }) } })();
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 0));
  const cards = () => [...page.node.querySelectorAll(".fp-node-card .fp-node-title")].map((n) => n.textContent).join("|");
  assert.equal(page.node.querySelectorAll(".fp-node-card").length, 2, cards());
  assert.match(page.node.textContent, /1 more only connect features/);
  [...page.node.querySelectorAll(".cx-toggle-row")].find((r) => /Show all nodes/.test(r.textContent)).querySelector("input").click();
  await new Promise((r) => setTimeout(r, 0));
  assert.ok(page.node.textContent.includes("Prompt"), "the groups are back");
  page.node.remove(); localStorage.clear(); delete globalThis.localStorage;
});

test("a problem that names no slot first (a loop) still shows its nodes in the simple view, and says its words", async () => {
  const slots = [{ id: "a", node: "PlumbA", inputs: {} }, { id: "b", node: "PlumbB", inputs: {} }, { id: "c", node: "PlumbC", inputs: {} }];
  const ps = { slots: () => slots, snapshot: () => ({ slots }), ensureLoaded: async () => {}, subscribe: () => () => {}, restore: async () => ({ refused: [] }),
    loading: () => false, loadError: () => null, saveNotes: () => [], incomplete: () => ["a feeds b feeds a: a slot cannot end up feeding itself"],
    refused: () => [], queueable: () => false, removedIds: () => [], offered: () => [] };
  const nodes = { PlumbA: { widgets: [] }, PlumbB: { widgets: [] }, PlumbC: { widgets: [] } };
  const page = models({ pipeline: ps, api: { describeNodes: async () => ({ nodes }) } })();
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(page.node.querySelectorAll(".fp-node-card").length, 2, "a and b, not c");
  assert.match(page.node.textContent, /cannot end up feeding itself/);
  page.node.remove();
});

test("a field still on screen for a node that was swapped away does not put its value on the new node", async () => {
  let slots = [{ id: "k", node: "FunPackSampler", inputs: { steps: 8 } }];
  const saved = [];
  const ps = { slots: () => slots, snapshot: () => ({ slots }), ensureLoaded: async () => {}, subscribe: () => () => {}, restore: async () => ({ refused: [] }),
    save: async (b) => { saved.push(b); }, loading: () => false, loadError: () => null, saveNotes: () => [], incomplete: () => [], refused: () => [],
    queueable: () => true, removedIds: () => [], offered: () => [] };
  const nodes = { FunPackSampler: { widgets: [{ name: "steps", type: "INT", min: 1, max: 100 }] }, Other: { widgets: [] } };
  const page = models({ pipeline: ps, api: { describeNodes: async () => ({ nodes }), packProviders: async () => ({}) } })();
  document.body.append(page.node);
  await new Promise((r) => setTimeout(r, 0));
  page.node.querySelector(".fp-node-card").click();                    // open it
  const field = page.node.querySelector("input[type=number]");
  slots = [{ id: "k", node: "Other", inputs: {} }];                     // the swap's answer lands; the field stays
  field.value = "20";
  field.dispatchEvent(new window.Event("change", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(saved, []);
  assert.match(page.node.textContent, /was swapped for Other: the value was not kept/);
  page.node.remove();
});
