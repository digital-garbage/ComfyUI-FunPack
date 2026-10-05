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
