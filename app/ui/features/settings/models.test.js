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
