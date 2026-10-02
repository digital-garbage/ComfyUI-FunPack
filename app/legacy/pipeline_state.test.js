const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");

function load(posts) {
  const defaults = [
    { id: "model", node: "Loader", group: "Loaders", inputs: { file: "", dtype: "bf16" } },
    { id: "gen", node: "Gen", group: "Sampling", inputs: { steps: 8, model: ["model", 0] } },
  ];
  global.window = {
    MovieEditorAPI: {
      pipeline: async () => ({ slots: JSON.parse(JSON.stringify(defaults)), incomplete: [], refused: [], queueable: true }),
      editPipeline: async (body) => { posts.push(body); return { slots: body.slots, incomplete: [], refused: [], queueable: true }; },
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
