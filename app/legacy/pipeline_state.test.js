const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");

function load(posts, opts = {}) {
  const defaults = [
    { id: "model", node: "Loader", group: "Loaders", inputs: { file: "", dtype: "bf16" } },
    { id: "gen", node: "Gen", group: "Sampling", inputs: { steps: 8, model: ["model", 0] } },
  ];
  global.window = {
    MovieEditorAPI: {
      pipeline: async () => ({ slots: JSON.parse(JSON.stringify(defaults)), incomplete: [], refused: [], queueable: true }),
      editPipeline: async (body) => {
        posts.push(body);
        if (opts.delay) await new Promise((r) => setTimeout(r, opts.delay));
        const bad = (body.slots || []).findIndex((s) => typeof s.node !== "string");
        if (bad >= 0) { const e = new Error(`slot ${bad} has no node`); throw e; } for (const [id, edits] of Object.entries(body.inputs || {})) { const sl = body.slots.find((x) => x.id === id); if (sl) sl.inputs = { ...sl.inputs, ...edits }; }
        return { slots: body.slots, incomplete: [], refused: [], queueable: true }; },
      modules: async () => ({ modules: [] }),
      probeFamily: async () => ({}),
    },
  };
  delete require.cache[require.resolve("./pipeline_state.js")];
  require("./pipeline_state.js");
  return global.window.PipelineState;
}

test("a project's saved pipeline lays its values over the server's default, keeping what the default added", async () => {
  const posts = [];
  const PS = load(posts);
  await PS.adopt([
    { id: "model", node: "Loader", group: "Mine", inputs: { file: "h3.safetensors" } },
    { id: "gen", node: "OtherNode", inputs: { steps: 99 } },                 // a different node: not the same slot
    { id: "extra", node: "Added", inputs: { x: 1 } },                         // the person's own
  ]);
  const by = Object.fromEntries(PS.slots().map((s) => [s.id, s]));
  assert.deepStrictEqual(by.model.inputs, { file: "h3.safetensors", dtype: "bf16" });
  assert.strictEqual(by.model.group, "Mine");
  assert.strictEqual(by.gen.inputs.steps, 8);
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
  global.window.MovieEditorAPI.pipeline = async () => { throw new Error("down"); };
  delete require.cache[require.resolve("./pipeline_state.js")];
  require("./pipeline_state.js");
  const P2 = global.window.PipelineState;
  await P2.adopt([{ id: "model", node: "Loader", inputs: {} }]);
  assert.strictEqual(P2.slots(), null);
});

test("a project's pipeline waits out an unreachable server and goes in on the first successful load", async () => {
  const PS = load([]);
  const good = global.window.MovieEditorAPI.pipeline;
  let up = false;
  global.window.MovieEditorAPI.pipeline = async () => { if (!up) throw new Error("down"); return good(); };
  await PS.adopt([{ id: "model", node: "Loader", inputs: { file: "h3.safetensors" } }]);
  assert.strictEqual(PS.slots(), null);
  up = true;
  await PS.ensureLoaded();
  assert.strictEqual(PS.slots().find((s) => s.id === "model").inputs.file, "h3.safetensors");
});

test("adopt says so when the project's pipeline could not go in", async () => {
  const PS = load([]);
  await PS.ensureLoaded();
  global.window.MovieEditorAPI.editPipeline = async () => { throw new Error("tunnel"); };
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
  global.window.MovieEditorAPI.modules = async () => { if (++calls === 1) throw new Error("down"); return { modules: [{ id: "m" }] }; };
  await PS.ensureLoaded();
  assert.deepStrictEqual(Object.keys(PS.modulesById()), []);
  await PS.save({ inputs: { model: { file: "z.safetensors" } } });
  assert.deepStrictEqual(Object.keys(PS.modulesById()), ["m"]);
});
