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
  assert.match(document.body.textContent, /Not started: this project.s pipeline is not loaded yet/);
});

test("a run uses the pipeline as it was at the click, whatever is edited while its shots are made", async () => {
  const sent = [];
  const { app, doc } = rig("idle", { generate: async (a) => { sent.push(a.slots); app.pipeline.slots = () => [{ id: "s", roles: [], inputs: { steps: 99 } }]; return true; } });
  doc.scenes.push({ id: "c", text: "y" });
  app.pipeline.slots = () => [{ id: "s", roles: [{ at: "generation.seed", input: "seed" }], inputs: { steps: 20 } }];
  await app.runner.units(["a", "c"]);
  assert.equal(sent.length, 2);
  assert.ok(sent.every((s) => s[0].inputs.steps === 20));
});

test("a scene edited while an earlier shot of the run is making goes as it read at the click; the run's own render still reaches the next shot", async () => {
  const seen = [];
  let calls = 0;
  const { app, doc } = rig("idle", { generate: async () => { calls += 1; if (calls === 1) doc.scenes.find((s) => s.id === "c").text = "edited"; return true; } });
  doc.scenes.push({ id: "c", text: "at click" });
  app.inputHooks = [async ({ project, scene }) => { seen.push([scene.text, Boolean((project.scene_renders || {}).a)]); return {}; }];
  await app.runner.units(["a", "c"]);
  assert.deepEqual(seen, [["x", false], ["at click", true]]);
});

test("Generate this scene clicked while a run is going says so, and queues nothing more", async () => {
  let queued = 0, finish;
  const { app, doc, host } = rig("idle", { generate: async () => { queued += 1; return true; }, waitForTerminal: () => Object.assign(new Promise((r) => { finish = r; }), { cancel() {} }) });
  let heard;
  app.on = (fn) => { heard = fn; return () => {}; };
  host.remove();
  const h2 = document.createElement("div");
  hub.setup({ host: h2, app });
  app.project.selected = doc.scenes[0];
  const before = document.body.textContent;
  app.runner.units(["a"]);
  await new Promise((r) => setTimeout(r, 0));
  heard("generate.scene");
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(queued, 1);
  assert.match(document.body.textContent.replace(before, ""), /already going/);
  finish("cancelled");
});

test("an edit still on its way when Generate is clicked (the Story box splitting) lands before the run reads the project", async () => {
  const seen = [];
  const { app, doc } = rig("idle", { generate: async () => true });
  app.beforeRun = [async () => { await new Promise((r) => setTimeout(r, 10)); doc.scenes[0].text = "typed last"; }];
  app.inputHooks = [({ scene }) => { seen.push(scene.text); return {}; }];
  await app.runner.units(["a"]);
  assert.deepEqual(seen, ["typed last"]);
});

test("a run that fails while running says which node and ComfyUI's own reason", async () => {
  const { app } = rig("idle", { generate: async () => true, waitForTerminal: () => Object.assign(Promise.resolve("failed"), { cancel() {} }) });
  app.generate.run.state.error = { node: "SaveVideo", message: "Saving image outside the output folder is not allowed." };
  const before = document.body.textContent;
  await app.runner.units(["a"]);
  assert.match(document.body.textContent.replace(before, ""), /failed in SaveVideo: Saving image outside the output folder/);
});

test("a project that already has a run in ComfyUI's queue (another tab) is not queued again, and the click says why", async () => {
  let queued = 0;
  const { app } = rig("idle", { generate: async () => { queued += 1; return true; } });
  app.api.projectQueued = async (id) => id === "P";
  const before = document.body.textContent;
  await app.runner.units(["a"]);
  assert.equal(queued, 0);
  assert.match(document.body.textContent.replace(before, ""), /already has a run in ComfyUI's queue/);
});

test("another tab's run that starts between this run's shots stops this run before its next shot", async () => {
  let queued = 0, asks = 0;
  const { app, doc } = rig("idle", { generate: async () => { queued += 1; return true; } });
  doc.scenes.push({ id: "c", text: "y" });
  app.api.projectQueued = async () => ++asks > 1;
  await app.runner.units(["a", "c"]);
  assert.equal(queued, 1);
});

test("a refusal said while a shot is being queued names the scene", async () => {
  const handlers = {};
  const { app, doc } = rig("idle", { on: (k, fn) => { handlers[k] = fn; }, generate: async () => { handlers.say("Length is 5000, above the largest 4096 it takes"); return false; } });
  doc.scenes.push({ id: "c", text: "y" });
  const before = document.body.textContent;
  await app.runner.units(["c"]);
  assert.match(document.body.textContent.replace(before, ""), /Scene 3: Length is 5000/);
});

test("a scene removed while it was being made: the result is not dropped in silence, it says where the file is", async () => {
  const { app, doc } = rig("idle", { generate: async () => { doc.scenes.splice(0, 2); return true; } });
  app.project.editFor = async (_id, fn) => fn(doc);
  const before = document.body.textContent;
  await app.runner.units(["a"]);
  assert.match(document.body.textContent.replace(before, ""), /was removed while it was being made.*a\.mp4/);
});
