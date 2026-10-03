import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let hub;
test.before(async () => { setupDom(); hub = (await import("./index.js")).default; });
test.after(() => teardownDom());

const rig = (phase) => {
  const handlers = {};
  const doc = { id: "P", num_frames_per_scene: 97, frame_rate: 25, scenes: [{ id: "a", text: "x" }, { id: "b", gen_unit_id: "a", cut_offset_frames: 49 }] };
  const g = { DONE: "done", FAILED: "failed", CANCELLED: "cancelled", run: { state: { phase, images: [{ filename: "a.mp4" }] } },
    state() { return this.run.state; }, on: (k, fn) => { handlers[k] = fn; }, subscribe: () => () => {}, cancel() {},
    waitForTerminal: () => Promise.resolve("done") };
  const project = { project: doc, get scenes() { return doc.scenes; }, selected: null,
    edit: (fn) => fn(doc), editFor: async (_id, fn) => fn(doc) };
  const app = { project, pipeline: { slots: () => [] }, generate: g, on: () => () => {} };
  hub.setup({ host: document.createElement("div"), app });
  return { handlers, doc };
};

test("a run that finished before the page reloaded lands on every clip of its unit", async () => {
  const { handlers, doc } = rig("done");
  handlers.adopt({ sceneId: "a", projectId: "P" });
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(doc.scene_renders.a.media.filename, "a.mp4");
  assert.ok(doc.scene_renders.b.inSec > 0, "the cut half plays from its cut");
});
