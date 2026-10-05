import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let hub;
test.before(async () => { setupDom(); hub = (await import("./index.js")).default; });
test.after(() => teardownDom());

test("the Sampler window shows the pipeline's steps and sampler, and a change saves into that slot", async () => {
  const saved = [];
  const slot = { id: "sampler", node: "S", inputs: { steps: 20, sampler_name: "euler", seed: 0 },
    roles: [{ at: "generation.sampling", input: "steps", label: "Steps" }, { at: "generation.sampling", input: "sampler_name", label: "Sampler" }, { at: "generation.seed", input: "seed" }] };
  const ps = { slots: () => [slot], activeModules: () => [], useful: () => true, usefulness: () => () => true, currentValues: () => ({}), loading: () => false, loadError: () => null,
    subscribe: () => () => {}, ensureLoaded: async () => {}, save: async (b) => { saved.push(b); Object.assign(slot.inputs, b.inputs.sampler); } };
  const api = { describeNodes: async () => ({ nodes: { S: { widgets: [{ name: "steps", type: "INT", min: 1, max: 100 }, { name: "sampler_name", type: "COMBO", choices: ["euler", "dpmpp_2m"] }, { name: "seed", type: "INT" }] } } }) };
  const host = document.createElement("div");
  document.body.append(host);
  hub.setup({ host, app: { pipeline: ps, api, on: () => () => {}, actions: null } });
  await new Promise((r) => setTimeout(r, 0));
  const button = host.querySelector("button");
  assert.equal(button.hidden, false, "shown with no sampling module, because the pipeline marks sampling inputs");
  button.click();
  await new Promise((r) => setTimeout(r, 0));
  const text = document.body.textContent;
  assert.match(text, /Steps/);
  assert.match(text, /Sampler/);
  const select = [...document.querySelectorAll("select")].find((s) => [...s.options].some((o) => o.value === "dpmpp_2m"));
  select.value = "dpmpp_2m";
  select.dispatchEvent(new Event("change", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 0));
  assert.deepStrictEqual(saved, [{ inputs: { sampler: { sampler_name: "dpmpp_2m" } } }]);
});

test("when the sampler node's options cannot be read, the window says where to change them", async () => {
  const slot = { id: "sampler", node: "S", inputs: { steps: 20 }, roles: [{ at: "generation.sampling", input: "steps", label: "Steps" }] };
  const ps = { slots: () => [slot], activeModules: () => [], useful: () => true, usefulness: () => () => true, currentValues: () => ({}), loading: () => false, loadError: () => null,
    subscribe: () => () => {}, ensureLoaded: async () => {}, save: async () => {} };
  const host = document.createElement("div");
  document.body.append(host);
  hub.setup({ host, app: { pipeline: ps, api: { describeNodes: async () => { throw new Error("offline"); } }, on: () => () => {}, actions: null } });
  await new Promise((r) => setTimeout(r, 0));
  host.querySelector("button").click();
  await new Promise((r) => setTimeout(r, 0));
  assert.match(document.body.textContent, /could not be read from the sampler node/);
});

test("typing a value back to what the window opened with is saved, though the window was not redrawn in between", async () => {
  let live = [{ id: "sampler", node: "S", inputs: { steps: 7 }, roles: [{ at: "generation.sampling", input: "steps", label: "Steps" }] }];
  const saved = [];
  const ps = { slots: () => live, activeModules: () => [], useful: () => true, usefulness: () => () => true, currentValues: () => ({}), loading: () => false, loadError: () => null,
    subscribe: () => () => {}, ensureLoaded: async () => {},
    save: async (b) => { saved.push(b.inputs.sampler.steps); live = [{ ...live[0], inputs: { ...live[0].inputs, ...b.inputs.sampler } }]; } };
  const host = document.createElement("div");
  document.body.append(host);
  hub.setup({ host, app: { pipeline: ps, api: { describeNodes: async () => ({ nodes: { S: { widgets: [{ name: "steps", type: "INT", min: 1, max: 100 }] } } }) }, on: () => () => {}, actions: null } });
  await new Promise((r) => setTimeout(r, 0));
  host.querySelector("button").click();
  await new Promise((r) => setTimeout(r, 0));
  const input = [...document.querySelectorAll("input[type=number]")].pop();
  for (const v of ["20", "7"]) {
    input.value = v;
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    await new Promise((r) => setTimeout(r, 0));
  }
  assert.deepStrictEqual(saved, [20, 7]);
});

test("a value that could not be saved is not left on screen: the window shows what will run and says why", async () => {
  const live = [{ id: "sampler", node: "S", inputs: { steps: 7 }, roles: [{ at: "generation.sampling", input: "steps", label: "Steps" }] }];
  const ps = { slots: () => live, activeModules: () => [], useful: () => true, usefulness: () => () => true, currentValues: () => ({}), loading: () => false, loadError: () => null,
    subscribe: () => () => {}, ensureLoaded: async () => {}, save: async () => {}, saveNotes: () => ["Could not save: down"] };
  const host = document.createElement("div");
  document.body.append(host);
  hub.setup({ host, app: { pipeline: ps, api: { describeNodes: async () => ({ nodes: { S: { widgets: [{ name: "steps", type: "INT", min: 1, max: 100 }] } } }) }, on: () => () => {}, actions: null } });
  await new Promise((r) => setTimeout(r, 0));
  host.querySelector("button").click();
  await new Promise((r) => setTimeout(r, 0));
  let input = [...document.querySelectorAll("input[type=number]")].pop();
  input.value = "25";
  input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  await new Promise((r) => setTimeout(r, 0));
  input = [...document.querySelectorAll("input[type=number]")].pop();
  assert.equal(input.value, "7");
  assert.match(document.body.textContent, /Could not save: down/);
});

test("a save queued behind another is not called failed: the window waits for it to land", async () => {
  let live = [{ id: "sampler", node: "S", inputs: { steps: 7 }, roles: [{ at: "generation.sampling", input: "steps", label: "Steps" }] }];
  let land;
  const ps = { slots: () => live, activeModules: () => [], useful: () => true, usefulness: () => () => true, currentValues: () => ({}), loading: () => false, loadError: () => null,
    subscribe: () => () => {}, ensureLoaded: async () => {}, saveNotes: () => [],
    save: async (b) => { land = () => { live = [{ ...live[0], inputs: { ...live[0].inputs, ...b.inputs.sampler } }]; }; },     // returns before it lands
    settled: async () => { await new Promise((r) => setTimeout(r, 5)); land(); } };
  const host = document.createElement("div");
  document.body.append(host);
  const toasts = document.querySelectorAll("[class*=toast]").length;
  hub.setup({ host, app: { pipeline: ps, api: { describeNodes: async () => ({ nodes: { S: { widgets: [{ name: "steps", type: "INT", min: 1, max: 100 }] } } }) }, on: () => () => {}, actions: null } });
  await new Promise((r) => setTimeout(r, 0));
  host.querySelector("button").click();
  await new Promise((r) => setTimeout(r, 0));
  const input = [...document.querySelectorAll("input[type=number]")].pop();
  input.value = "30";
  input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(live[0].inputs.steps, 30);
  assert.equal(document.querySelectorAll("[class*=toast]").length, toasts);
});

test("two quick edits of one input: the first is not called failed because the second replaced it", async () => {
  let live = [{ id: "sampler", node: "S", inputs: { steps: 7 }, roles: [{ at: "generation.sampling", input: "steps", label: "Steps" }] }];
  let queued = null;
  const ps = { slots: () => live, activeModules: () => [], useful: () => true, usefulness: () => () => true, currentValues: () => ({}), loading: () => false, loadError: () => null,
    subscribe: () => () => {}, ensureLoaded: async () => {}, saveNotes: () => [],
    save: async (b) => { queued = { ...(queued || {}), ...b.inputs.sampler }; },                 // queued: lands at settle, the newest winning
    settled: async () => { await new Promise((r) => setTimeout(r, 5)); if (queued) { live = [{ ...live[0], inputs: { ...live[0].inputs, ...queued } }]; queued = null; } } };
  const host = document.createElement("div");
  document.body.append(host);
  const toasts = document.querySelectorAll("[class*=toast]").length;
  hub.setup({ host, app: { pipeline: ps, api: { describeNodes: async () => ({ nodes: { S: { widgets: [{ name: "steps", type: "INT", min: 1, max: 100 }] } } }) }, on: () => () => {}, actions: null } });
  await new Promise((r) => setTimeout(r, 0));
  host.querySelector("button").click();
  await new Promise((r) => setTimeout(r, 0));
  const input = [...document.querySelectorAll("input[type=number]")].pop();
  for (const v of ["20", "25"]) { input.value = v; input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true })); }
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(live[0].inputs.steps, 25);
  assert.equal(document.querySelectorAll("[class*=toast]").length, toasts);
});
