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
