import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let taste;
test.before(async () => { setupDom(); ({ taste } = await import("./taste.js")); });
test.after(() => teardownDom());
const tick = () => new Promise((r) => setTimeout(r, 0));

test("the panel shows block influence: the switch, the clip counts, and each pair against chance", async () => {
  const sets = [];
  const app = { api: {
    tasteKeys: async () => ({ keys: [] }),
    blockInfluence: async () => ({ key: "fox", enabled: true, problem: null }),
    blockInfluenceGroups: async () => ({ key: "fox", used: 8, counts: { liked: 2, image: 2, composition: 2, both: 2 },
      cos: { "image vs composition": 0.5 }, chance: { "image vs composition": { mean: 0.02, p95: 0.4 } } }),
    setBlockInfluence: async (on) => { sets.push(on); return { enabled: on }; },
  } };
  const page = taste(app)();
  document.body.append(page.node);
  await tick(); await tick();
  const text = page.node.textContent;
  assert.match(text, /Record block influence/);
  assert.match(text, /liked 2, image 2, composition 2, both 2/);
  assert.match(text, /image vs composition/);
  assert.match(text, /\+0\.50/);
  assert.match(text, /95th \+0\.40/);
  const toggle = page.node.querySelector('input[type="checkbox"]');
  assert.equal(toggle.checked, true);
  toggle.click(); await tick(); await tick();
  assert.deepEqual(sets, [false]);
});

test("nothing recorded yet is said, not shown as a comparison", async () => {
  const app = { api: {
    tasteKeys: async () => ({ keys: [] }),
    blockInfluence: async () => ({ key: "none", enabled: false, problem: null }),
    blockInfluenceGroups: async () => ({ key: "none", used: 0, counts: {}, cos: {}, chance: {}, note: "nothing recorded yet" }),
  } };
  const page = taste(app)();
  document.body.append(page.node);
  await tick(); await tick();
  assert.match(page.node.textContent, /nothing recorded yet/);
});

test("a failed report still shows the switch, so recording can be turned off", async () => {
  const sets = [];
  const app = { api: {
    tasteKeys: async () => ({ keys: [] }),
    blockInfluence: async () => ({ key: "fox", enabled: true, problem: null }),
    blockInfluenceGroups: async () => { throw new Error("HTTP 500"); },
    setBlockInfluence: async (on) => { sets.push(on); return { enabled: on }; },
  } };
  const page = taste(app)();
  document.body.append(page.node);
  await tick(); await tick();
  const toggle = page.node.querySelector('input[type="checkbox"]');
  assert.ok(toggle, "the switch is still there");
  toggle.click(); await tick(); await tick();
  assert.deepEqual(sets, [false]);
  assert.match(page.node.textContent, /HTTP 500/);
});

test("the block influence readout shows flatness, and Clear asks before it clears", async () => {
  const cleared = [];
  const app = { api: {
    tasteKeys: async () => ({ keys: [] }),
    blockInfluence: async () => ({ key: "fox", enabled: true, problem: null, runs: 9, skipped: 1, used: 8, flatness: 0.123, mean_novelty: 0.4 }),
    blockInfluenceGroups: async () => ({ key: "fox", used: 0, counts: {}, cos: {}, chance: {}, note: "nothing recorded yet" }),
    blockInfluenceExportUrl: (k) => `/x/${k}`,
    clearBlockInfluence: async (k) => { cleared.push(k); return {}; },
  } };
  const page = taste(app)();
  document.body.append(page.node);
  await tick(); await tick();
  assert.match(page.node.textContent, /0\.123/);
  assert.match(page.node.textContent, /0\.400/);
  assert.match(page.node.textContent, /Download/);
  assert.match(page.node.textContent, /Clear/);
});
