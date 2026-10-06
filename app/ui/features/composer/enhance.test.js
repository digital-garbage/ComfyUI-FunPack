import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let enhance;
test.before(async () => { setupDom(); ({ enhance } = await import("./enhance.js")); });
test.after(() => teardownDom());

test("the Enhance tab follows the open project's pipeline: no enhancer there, no toggle shown", async () => {
  let live = [{ id: "e", node: "FunPackEnhancePrompt", inputs: { enabled: true } }], heard;
  const ps = { slots: () => live, ensureLoaded: async () => {}, save: async () => {}, subscribe: (fn) => { heard = fn; return () => {}; } };
  const api = { enhancerDefaults: async () => ({}), enhancerRuns: async () => ({ runs: [] }) };
  const page = enhance({ pipeline: ps, api }, () => {});
  document.body.append(page.node);
  assert.match(page.node.textContent, /Enhance prompt/);
  live = [{ id: "s", node: "FunPackSampler", inputs: {} }];              // another project opened
  heard(live);
  assert.match(page.node.textContent, /No prompt enhancer here/);
});

test("a run that ends while Enhance is open shows its final prompt without reopening the tab", async () => {
  const ps = { slots: () => [{ id: "e", node: "FunPackEnhancePrompt", inputs: {} }], ensureLoaded: async () => {}, save: async () => {}, subscribe: () => () => {} };
  let runs = [], tell;
  const api = { enhancerDefaults: async () => ({}), enhancerRuns: async () => ({ runs }) };
  const generate = { subscribe: (fn) => { tell = fn; fn({ phase: "idle" }); return () => {}; } };
  const page = enhance({ pipeline: ps, api, generate }, () => {});
  await new Promise((r) => setTimeout(r, 0));
  assert.match(page.node.textContent, /Nothing yet/);
  runs = [{ prompt_id: "p", status: "off: the prompt ran as typed", before: "a red fox", after: "a red fox at dusk" }];
  tell({ phase: "running" }); tell({ phase: "done" });
  await new Promise((r) => setTimeout(r, 0));
  assert.match(page.node.textContent, /a red fox at dusk/);
});

test("Chat on a pipeline with no enhancer says so instead of taking comments that would reach nothing", async () => {
  const { chat } = await import("./chat.js");
  let slots = [{ id: "s", node: "Sampler" }], heard;
  const app = { project: { project: { id: "P", scenes: [] }, pref: (_k, d) => d, setPref: () => {} }, selection: {}, lastRun: { n: 0 },
    api: { enhancerRuns: async () => ({ runs: [] }) }, on: () => () => {}, pipeline: { slots: () => slots, subscribe: (fn) => { heard = fn; return () => {}; } } };
  const page = chat(app, () => {});
  assert.match(page.node.textContent, /No prompt enhancer here/);
  assert.equal(page.node.querySelector("textarea"), null);
  slots = [{ id: "e", node: "FunPackEnhancePrompt" }];
  heard(slots, "edit");
  assert.ok(page.node.querySelector("textarea"));
});
