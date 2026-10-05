import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let hub;
test.before(async () => { setupDom(); hub = (await import("./index.js")).default; });
test.after(() => teardownDom());

const rig = (phase, extra = {}) => {
  const handlers = {};
  const doc = { id: "P", num_frames_per_scene: 97, frame_rate: 25, scenes: [{ id: "a", text: "x" }, { id: "b", gen_unit_id: "a", cut_offset_frames: 49 }] };
  const g = { DONE: "done", FAILED: "failed", CANCELLED: "cancelled", run: { state: { phase, images: [{ filename: "a.mp4" }] } },
    state() { return this.run.state; }, on: (k, fn) => { handlers[k] = fn; }, subscribe: () => () => {}, cancel() {},
    waitForTerminal: () => Object.assign(Promise.resolve("done"), { cancel() {} }), ...extra };
  const project = { project: doc, get scenes() { return doc.scenes; }, selected: null,
    edit: (fn) => fn(doc), editFor: async (_id, fn) => fn(doc) };
  const app = { project, selection: { ids: [] }, lastRun: { n: 0 }, runner: {}, pipeline: { slots: () => [{ id: "s", roles: [{ at: "generation.seed", input: "seed" }], inputs: {} }] }, generate: g, api: { newTasteGeneration: () => Promise.resolve({}) }, on: () => () => {} };
  const host = document.createElement("div");
  hub.setup({ host, app });
  return { handlers, doc, host, app };
};

test("a run that finished before the page reloaded lands on every clip of its unit", async () => {
  const { handlers, doc } = rig("done");
  handlers.adopt({ sceneId: "a", projectId: "P" });
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(doc.scene_renders.a.media.filename, "a.mp4");
  assert.ok(doc.scene_renders.b.inSec > 0, "the cut half plays from its cut");
});

test("a queue ComfyUI refuses is said in the refusal's own words, not thrown", async () => {
  const { host } = rig("idle", { generate: async () => false, run: { state: { phase: "idle", images: [], error: { message: "node 7 refused the graph" } } } });
  host.querySelector("button").click();           // ▶ Generate
  await new Promise((r) => setTimeout(r, 20));
  assert.match(document.body.textContent, /node 7 refused the graph/);
});

test("a batch of takes runs the unit N times, keeps every render as a take, and starts one taste generation for all of them", async () => {
  let queued = 0, fresh = 0, n = 0;
  const { app, doc } = rig("idle", { generate: async () => { queued += 1; return true; }, waitForTerminal: () => Object.assign(Promise.resolve("done"), { cancel() {} }) });
  app.api.newTasteGeneration = () => { fresh += 1; return Promise.resolve({}); };
  const g = app.generate;
  Object.defineProperty(g.run.state, "images", { get: () => [{ filename: `t${++n}.mp4` }] });
  g.run.state.promptId = "p";
  await app.runner.units(["a"], 3);
  assert.equal(queued, 3);
  assert.equal(fresh, 1, "all takes stay pending for a rating");
  assert.deepEqual(doc.scene_variants.a.map((t) => t.media.filename), ["t1.mp4", "t2.mp4", "t3.mp4"]);
  assert.equal(doc.scene_renders.a.media.filename, "t3.mp4");
});

test("the take that was on the clip keeps its rating when a newer one replaces it; only the last twenty-four are kept", async () => {
  const { app, doc } = rig("idle", { generate: async () => true });
  doc.scenes[0].rating = "10";
  doc.scene_renders = { a: { media: { filename: "old.mp4" }, promptId: "p0" } };
  doc.scene_variants = { a: [{ media: { filename: "old.mp4" }, promptId: "p0", rating: "" }] };
  app.generate.run.state.promptId = "p1";
  await app.runner.units(["a"], 1);
  assert.equal(doc.scene_variants.a[0].rating, "10");
  doc.scene_variants.a = Array.from({ length: 30 }, (_, i) => ({ media: { filename: `${i}.mp4` }, promptId: `q${i}`, rating: "" }));
  await app.runner.units(["a"], 1);
  assert.equal(doc.scene_variants.a.length, 24);
});

test("a render from before takes becomes a take; the oldest unrated take goes first when the cap is hit, never a rated one", async () => {
  const { app, doc } = rig("idle", { generate: async () => true });
  doc.scenes[0].rating = "10";
  doc.scene_renders = { a: { media: { filename: "legacy.mp4" } } };
  app.generate.run.state.promptId = "p1";
  await app.runner.units(["a"], 1);
  assert.deepEqual(doc.scene_variants.a.map((t) => [t.media.filename, t.rating]), [["legacy.mp4", "10"], ["a.mp4", ""]]);
  doc.scene_variants.a = Array.from({ length: 24 }, (_, i) => ({ media: { filename: `${i}.mp4` }, promptId: `q${i}`, rating: i === 0 ? "10" : "" }));
  await app.runner.units(["a"], 1);
  assert.equal(doc.scene_variants.a[0].media.filename, "0.mp4", "the rated one stays");
  assert.ok(!doc.scene_variants.a.some((t) => t.media.filename === "1.mp4"), "the oldest unrated went");
});

test("a batch with no seed input says so and makes nothing", async () => {
  let queued = 0;
  const { app } = rig("idle", { generate: async () => { queued += 1; return true; } });
  app.pipeline.slots = () => [];
  await app.runner.units(["a"], 3);
  assert.equal(queued, 0);
  assert.match(document.body.textContent, /no seed input/);
});

test("Generate refuses, saying why, while the open project's pipeline is not the live one", async () => {
  let queued = 0;
  const { host, app } = rig("idle", { generate: async () => { queued += 1; return true; } });
  app.pipelineOwned = () => false;
  host.querySelector("button").click();
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(queued, 0);
  assert.match(document.body.textContent, /pipeline is not loaded yet/);
});
