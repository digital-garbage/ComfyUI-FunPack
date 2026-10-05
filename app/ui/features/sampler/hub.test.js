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
  const ps = { slots: () => [slot], activeModules: () => [], useful: () => true, currentValues: () => ({}), loading: () => false, loadError: () => null,
    subscribe: () => () => {}, ensureLoaded: async () => {}, save: async (b) => { saved.push(b); } };
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
  const ps = { slots: () => [slot], activeModules: () => [], useful: () => true, currentValues: () => ({}), loading: () => false, loadError: () => null,
    subscribe: () => () => {}, ensureLoaded: async () => {}, save: async () => {} };
  const host = document.createElement("div");
  document.body.append(host);
  hub.setup({ host, app: { pipeline: ps, api: { describeNodes: async () => { throw new Error("offline"); } }, on: () => () => {}, actions: null } });
  await new Promise((r) => setTimeout(r, 0));
  host.querySelector("button").click();
  await new Promise((r) => setTimeout(r, 0));
  assert.match(document.body.textContent, /could not be read from the sampler node/);
});
