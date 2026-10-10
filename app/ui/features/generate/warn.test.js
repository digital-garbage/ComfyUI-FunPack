import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let warn;
test.before(async () => { setupDom(); warn = (await import("./warn.js")).default; });
test.after(() => teardownDom());

const chip = (host) => [...host.querySelectorAll("button")].find((b) => /No Taste key/.test(b.textContent));

test("the No Taste key chip shows only while a learning feature is on and the key is empty", () => {
  const taste = { id: "taste" };
  let values = { taste: { key: "" } }, learning = true, listener;
  const pipeline = { slots: () => [], allOff: () => false, modulesById: () => ({ taste }), activeModules: () => [taste],
    useful: (m) => m === taste && learning, currentValues: () => values, subscribe: (fn) => { listener = fn; return () => {}; } };
  const host = document.createElement("div");
  warn.setup({ host, app: { pipeline } });
  assert.equal(chip(host).hidden, false, "learning on, no key: said");
  values = { taste: { key: "  portraits " } }; listener();
  assert.equal(chip(host).hidden, true, "a key is set");
  values = { taste: { key: "" } }; learning = false; listener();
  assert.equal(chip(host).hidden, true, "nothing learns, so a missing key changes nothing");
});

test("the two-sparse-attentions chip shows only when SLA and a native sparse node are both in the pipeline", () => {
  let slots = [{ id: "loader", node: "FunPackLoader", inputs: { sla: true } }, { id: "sa", node: "BlockSparseAttention", inputs: {} }];
  let listener;
  const pipeline = { slots: () => slots, allOff: () => false, modulesById: () => ({}), activeModules: () => [], useful: () => false,
    currentValues: () => ({}), subscribe: (fn) => { listener = fn; return () => {}; } };
  const host = document.createElement("div");
  warn.setup({ host, app: { pipeline } });
  const twice = () => [...host.querySelectorAll("button")].find((b) => /Two sparse attentions/.test(b.textContent));
  assert.equal(twice().hidden, false, "both present: said");
  slots = [{ id: "loader", node: "FunPackLoader", inputs: { sla: false } }, { id: "sa", node: "BlockSparseAttention", inputs: {} }];
  listener();
  assert.equal(twice().hidden, true, "SLA off: the native node alone is not a clash");
});

test("the slow-torch chip shows the server's sentence when this build runs int8 slowly, and stays hidden otherwise", async () => {
  const pipeline = { slots: () => [], allOff: () => false, modulesById: () => ({}), activeModules: () => [], useful: () => false,
    currentValues: () => ({}), subscribe: () => () => {} };
  const slowChip = (host) => [...host.querySelectorAll("button")].find((b) => /Slow torch/.test(b.textContent));
  const tick = () => new Promise((r) => setTimeout(r, 0));
  let host = document.createElement("div");
  warn.setup({ host, app: { pipeline, api: { torchBuild: async () => ({ slow_int8: "torch is built for CUDA 12.8: reinstall" }) } } });
  await tick();
  assert.equal(slowChip(host).hidden, false);
  assert.match(slowChip(host).title, /CUDA 12\.8/);
  host = document.createElement("div");
  warn.setup({ host, app: { pipeline, api: { torchBuild: async () => ({ slow_int8: null }) } } });
  await tick();
  assert.equal(slowChip(host).hidden, true, "a good build says nothing");
  host = document.createElement("div");
  warn.setup({ host, app: { pipeline, api: { torchBuild: async () => { throw new Error("offline"); } } } });
  await tick();
  assert.equal(slowChip(host).hidden, true, "an unreadable build is not a claim");
});
