import test from "node:test";
import assert from "node:assert";
import { createPipelineState } from "../pipeline_state.js";

let lastApi;
function load(posts, opts = {}) {
  const defaults = [
    { id: "model", node: "Loader", group: "Loaders", inputs: { file: "", dtype: "bf16" } },
    { id: "gen", node: "Gen", group: "Sampling", inputs: { steps: 8, model: ["model", 0] } },
  ];
  const API = {
      pipeline: async () => ({ slots: JSON.parse(JSON.stringify(defaults)), incomplete: [], refused: [], queueable: true }),
      editPipeline: async (body) => {
        posts.push(body);
        if (opts.delay) await new Promise((r) => setTimeout(r, opts.delay));
        const bad = (body.slots || []).findIndex((s) => typeof s.node !== "string");
        if (bad >= 0) { const e = new Error(`slot ${bad} has no node`); throw e; } for (const [id, edits] of Object.entries(body.inputs || {})) { const sl = body.slots.find((x) => x.id === id); if (sl) sl.inputs = { ...sl.inputs, ...edits }; }
        return { slots: body.slots, incomplete: [], refused: [], queueable: true }; },
      modules: async () => ({ modules: [] }),
      probeFamily: async () => ({}),
  };
  lastApi = API;
  return createPipelineState(API);
}

test("a project's saved pipeline lays its values over the server's default, keeping what the default added", async () => {
  const posts = [];
  const PS = load(posts);
  await PS.adopt([
    { id: "model", node: "Loader", group: "Mine", inputs: { file: "h3.safetensors" } },
    { id: "gen", node: "OtherNode", inputs: { steps: 99 } },                 // a swapped node: the person's swap stays
    { id: "extra", node: "Added", inputs: { x: 1 } },                         // the person's own
  ]);
  const by = Object.fromEntries(PS.slots().map((s) => [s.id, s]));
  assert.deepStrictEqual(by.model.inputs, { file: "h3.safetensors", dtype: "bf16" });
  assert.strictEqual(by.model.group, "Mine");
  assert.deepStrictEqual([by.gen.node, by.gen.inputs.steps], ["OtherNode", 99]);
  assert.ok(by.extra);
});

test("adopting is the project's own copy, so it is not announced as an edit; a real edit is", async () => {
  const posts = [];
  const PS = load(posts);
  let heard = 0;
  PS.subscribe(() => { heard++; });
  await PS.adopt([{ id: "model", node: "Loader", inputs: { file: "a" } }]);
  assert.strictEqual(heard, 0);
  await PS.save({ inputs: { model: { file: "b" } } });
  assert.strictEqual(heard, 1);
});

test("nothing saved leaves the session's pipeline as it is", async () => {
  const PS = load([]);
  await PS.ensureLoaded();
  const before = JSON.stringify(PS.slots());
  await PS.adopt([]);
  assert.strictEqual(JSON.stringify(PS.slots()), before);
});

test("a v4-shaped or hostile saved slot cannot brick the pipeline", async () => {
  const PS = load([]);
  await PS.adopt([
    { id: "model", node: "Loader", inputs: { file: "h3.safetensors" } },
    { id: "abc123", node_class: "OldV4Node", inputs: {} },
    { id: "bad", node: "X", inputs: [1, 2] },
  ]);
  assert.deepStrictEqual(PS.slots().map((s) => s.id), ["model", "gen"]);
  assert.strictEqual(PS.slots()[0].inputs.file, "h3.safetensors");
  await PS.save({ inputs: { model: { file: "new" } } });
  assert.strictEqual(PS.slots()[0].inputs.file, "new");
});

test("an answer about the previous project's pipeline does not overwrite the one just opened", async () => {
  const PS = load([], { delay: 30 });
  await PS.ensureLoaded();
  const inflight = PS.save({ inputs: { model: { file: "A-file" } } });
  await new Promise((r) => setTimeout(r, 5));
  await PS.adopt([{ id: "model", node: "Loader", inputs: { file: "B-file" } }]);
  await inflight;
  assert.strictEqual(PS.slots()[0].inputs.file, "B-file");
});

test("nothing loaded (ComfyUI unreachable) leaves slots null, never an empty pipeline", async () => {
  const PS = load([]);
  lastApi.pipeline = async () => { throw new Error("down"); };
  const P2 = createPipelineState(lastApi);
  await P2.adopt([{ id: "model", node: "Loader", inputs: {} }]);
  assert.strictEqual(P2.slots(), null);
});

test("a project's pipeline waits out an unreachable server and goes in on the first successful load", async () => {
  const PS = load([]);
  const good = lastApi.pipeline;
  let up = false;
  lastApi.pipeline = async () => { if (!up) throw new Error("down"); return good(); };
  await PS.adopt([{ id: "model", node: "Loader", inputs: { file: "h3.safetensors" } }]);
  assert.strictEqual(PS.slots(), null);
  up = true;
  await PS.ensureLoaded();
  assert.strictEqual(PS.slots().find((s) => s.id === "model").inputs.file, "h3.safetensors");
});

test("adopt says so when the project's pipeline could not go in", async () => {
  const PS = load([]);
  await PS.ensureLoaded();
  lastApi.editPipeline = async () => { throw new Error("tunnel"); };
  assert.strictEqual(await PS.adopt([{ id: "model", node: "Loader", inputs: { file: "x" } }]), false);
  assert.strictEqual(await PS.adopt([]), true);
});

test("an edit made while a project's pipeline goes in lands on that pipeline", async () => {
  const PS = load([], { delay: 30 });
  await PS.ensureLoaded();
  const adopting = PS.adopt([{ id: "model", node: "Loader", inputs: { file: "B-file", dtype: "fp8" } }]);
  const edit = PS.save({ inputs: { model: { dtype: "int8" } } });
  await Promise.all([adopting, edit]);
  const model = PS.slots().find((s) => s.id === "model");
  assert.deepStrictEqual([model.inputs.file, model.inputs.dtype], ["B-file", "int8"]);
});

test("a failed module-list fetch is retried on the next save, not remembered as done", async () => {
  const posts = [];
  const PS = load(posts);
  let calls = 0;
  lastApi.modules = async () => { if (++calls === 1) throw new Error("down"); return { modules: [{ id: "m" }] }; };
  await PS.ensureLoaded();
  assert.deepStrictEqual(Object.keys(PS.modulesById()), []);
  await PS.save({ inputs: { model: { file: "z.safetensors" } } });
  assert.deepStrictEqual(Object.keys(PS.modulesById()), ["m"]);
});


test("a slot the person removed stays removed when the project is opened again", async () => {
  const PS = load([]);
  await PS.adopt([{ id: "model", node: "Loader", inputs: {} }], ["gen"]);
  assert.deepStrictEqual(PS.slots().map((s) => s.id), ["model"]);
  assert.deepStrictEqual(PS.removedIds(), ["gen"]);
});

test("a structural edit is sent on its own and its refusal is handed back", async () => {
  const posts = [];
  const PS = load(posts);
  await PS.ensureLoaded();
  lastApi.editPipeline = async (body) => {
    posts.push(body);
    return { slots: body.slots, refused: ["nope"], incomplete: [], queueable: false };
  };
  const res = await PS.edit({ action: "remove", slot: "gen" });
  assert.deepStrictEqual(res.refused, ["nope"]);
  assert.strictEqual(posts.at(-1).action, "remove");
  assert.deepStrictEqual(PS.removedIds(), []);                 // refused: nothing is remembered as removed
});

test("a default link the person unwired stays unwired when the project is opened again", async () => {
  const PS = load([]);
  await PS.adopt([{ id: "model", node: "Loader", inputs: { file: "x" } },
                  { id: "gen", node: "Gen", inputs: { steps: 8 } }], [], { gen: ["model"] });
  assert.strictEqual("model" in PS.slots().find((s) => s.id === "gen").inputs, false);
  assert.deepStrictEqual(PS.unwiredMap(), { gen: ["model"] });
});

test("a default link an older project never had (an update added it) is kept", async () => {
  const PS = load([]);
  await PS.adopt([{ id: "model", node: "Loader", inputs: { file: "x" } },
                  { id: "gen", node: "Gen", inputs: { steps: 8 } }]);       // nothing recorded as unwired
  assert.strictEqual("model" in PS.slots().find((s) => s.id === "gen").inputs, true);
});

test("a group moved while a save is in flight is still moved when it lands", async () => {
  const PS = load([], { delay: 30 });
  await PS.ensureLoaded();
  const inflight = PS.save({ inputs: { model: { file: "z" } } });
  await new Promise((r) => setTimeout(r, 5));
  await PS.setGroup("gen", "Mine");
  await inflight;
  assert.strictEqual(PS.slots().find((s) => s.id === "gen").group, "Mine");
});

test("a default slot swapped for the same node, or removed and added back, does not get its links back on reopen", async () => {
  const PS = load([]);
  await PS.ensureLoaded();
  const API = lastApi;
  API.editPipeline = async (body) => ({ refused: [], incomplete: [], queueable: true,
    slots: body.slots.map((s) => (s.id === "gen" ? { ...s, inputs: {} } : s)) });
  await PS.edit({ action: "replace", slot: "gen", node: "Gen" });
  assert.deepStrictEqual(PS.unwiredMap(), { gen: ["model"] });
  const saved = JSON.parse(JSON.stringify(PS.slots()));
  const again = load([]);
  await again.adopt(saved, [], PS.unwiredMap());
  assert.strictEqual("model" in again.slots().find((s) => s.id === "gen").inputs, false);
});

test("a project with every slot removed does not get the default back", async () => {
  const posts = [];
  const PS = load(posts);
  await PS.adopt([], ["model", "gen"]);
  assert.ok(posts.length > 0);
  assert.deepStrictEqual(posts.at(-1).slots, []);
});

test("a group change queued behind a save in flight is not re-stamped over a later group", async () => {
  const PS = load([], { delay: 20 });
  await PS.ensureLoaded();
  const inflight = PS.save({ inputs: { model: { file: "z" } } });
  const first = PS.setGroup("gen", "Mine");
  await Promise.all([inflight, first]);
  await PS.setGroup("gen", "Other");
  assert.strictEqual(PS.slots().find((s) => s.id === "gen").group, "Other");
});

test("restore puts back the pipeline as snapshotted: slots, removed, unwired, and tells the project", async () => {
  const PS = load([]);
  await PS.ensureLoaded();
  const snap = PS.snapshot();
  await PS.save({ inputs: { model: { file: "changed" } } });
  lastApi.editPipeline = async (body) => ({ refused: [], incomplete: [], queueable: true, slots: body.slots });
  await PS.edit({ action: "remove", slot: "gen" });
  assert.deepStrictEqual(PS.removedIds(), ["gen"]);
  let heard = 0;
  PS.subscribe(() => { heard++; });
  const res = await PS.restore(snap);
  assert.deepStrictEqual(res.refused, []);
  assert.deepStrictEqual(PS.slots(), snap.slots);
  assert.deepStrictEqual(PS.removedIds(), []);
  assert.strictEqual(heard, 1);
});

test("a refused restore leaves the pipeline as it is", async () => {
  const PS = load([]);
  await PS.ensureLoaded();
  const snap = PS.snapshot();
  await PS.save({ inputs: { model: { file: "changed" } } });
  lastApi.editPipeline = async () => ({ refused: ["no"], slots: [] });
  const res = await PS.restore(snap);
  assert.deepStrictEqual(res.refused, ["no"]);
  assert.strictEqual(PS.slots().find((s) => s.id === "model").inputs.file, "changed");
});

test("a module switched off is remembered with the settings, listed as off, and can be switched back on", async () => {
  const posts = [];
  const PS = load(posts);
  await PS.ensureLoaded();
  lastApi.modules = async () => ({ modules: [{ id: "sharpen", settings: {} }, { id: "alg", settings: {} }], control: {} });
  await PS.refreshControl();
  await PS.setOff("sharpen", true);
  assert.strictEqual(PS.isOff("sharpen"), true);
  assert.strictEqual(PS.isOff("alg"), false);
  assert.deepStrictEqual(posts.at(-1).values._off, { modules: ["sharpen"] });
  await PS.setOff("alg", true);
  await PS.setOff("sharpen", false);
  assert.deepStrictEqual(posts.at(-1).values._off, { modules: ["alg"] });
});

test("Disable all enhancements is saved on its own, hides every module, and keeps the per-module choices underneath", async () => {
  const posts = [];
  const PS = load(posts);
  await PS.ensureLoaded();
  lastApi.modules = async () => ({ modules: [{ id: "sharpen", settings: {} }, { id: "alg", settings: {} }], control: {} });
  await PS.refreshControl();
  await PS.setOff("alg", true);
  await PS.setAllOff(true);
  assert.strictEqual(PS.allOff(), true);
  assert.deepStrictEqual(posts.at(-1).values._off, { modules: ["alg"], all: true });
  assert.deepStrictEqual(PS.activeModules(), []);
  await PS.setAllOff(false);
  assert.strictEqual(PS.allOff(), false);
  assert.strictEqual(PS.isOff("alg"), true);
});

test("settled() resolves only after the edits made so far were sent", async () => {
  const posts = [];
  const PS = load(posts);
  await PS.ensureLoaded();
  const first = PS.setAllOff(true);
  PS.setAllOff(false);
  await PS.settled();
  assert.strictEqual(posts.at(-1).values._off.all, false);
  await first;
});

test("a project opened after another does not inherit the first one's pending switch", async () => {
  const PS = load([]);
  await PS.ensureLoaded();
  await PS.setOff("sharpen", true);
  await PS.adopt([{ id: "model", node: "Loader", inputs: { file: "x" } }]);
  assert.strictEqual(PS.isOff("sharpen"), false);
});

test("the settings of a Generate are taken once: frozenInputs is a copy, wired inputs are left to their node", async () => {
  const PS = load([]);
  await PS.ensureLoaded();
  const frozen = PS.frozenInputs();
  assert.deepStrictEqual(frozen.model, { file: "", dtype: "bf16" });
  assert.deepStrictEqual(frozen.gen, { steps: 8 });                       // `model: ["model", 0]` is fed by a node
  await PS.save({ inputs: { gen: { steps: 4 } } });
  assert.strictEqual(PS.slots().find((s) => s.id === "gen").inputs.steps, 4);
  assert.strictEqual(frozen.gen.steps, 8);                                // the snapshot does not follow the edit
  frozen.model.dtype = "fp16";
  assert.strictEqual(PS.slots().find((s) => s.id === "model").inputs.dtype, "bf16");   // and is not the live object
});
