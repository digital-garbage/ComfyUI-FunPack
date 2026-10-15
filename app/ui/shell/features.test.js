import test from "node:test";
import assert from "node:assert/strict";
import { offer, _reset } from "./mounts.js";
import { loadFeatures } from "./features.js";

const host = () => { const nodes = []; return { childNodes: nodes, append: (n) => nodes.push(n) }; };
const hubs = {
  ok: { id: "ok", mount: "a", setup: ({ host: h }) => h.append({ remove() {} }) },
  throws: { id: "throws", mount: "a", setup: () => { throw new Error("boom"); } },
  nowhere: { id: "nowhere", mount: "zzz", setup() {} },
  needy: { id: "needy", mount: "a", needs: ["gpu"], setup() {} },
};
const run = (names) => loadFeatures(names, { load: async (n) => ({ default: hubs[n] }), has: () => false });

test("a feature that loads is mounted; one that throws, names no region, or lacks a need is hidden, and the rest still load", async () => {
  _reset(); offer("a", host());
  const { mounted, hidden } = await run(["throws", "nowhere", "needy", "ok"]);
  assert.deepEqual(mounted.map((m) => m.id), ["ok"]);
  assert.deepEqual(hidden.map((h) => [h.id, h.why.split(" ")[0]]), [["throws", "boom"], ["nowhere", "no"], ["needy", "needs"]]);
});

test("what a feature put on the page before it threw is taken away", async () => {
  _reset(); const h = host(); offer("a", h);
  hubs.throws.setup = ({ host: x }) => { x.append({ remove() { h.childNodes.length = 0; } }); throw new Error("late"); };
  await run(["throws"]);
  assert.equal(h.childNodes.length, 0);
});

test("a feature that subscribed and then threw stops hearing the app", async () => {
  _reset(); offer("a", host());
  const heard = new Set();
  const app = { on: (fn) => { heard.add(fn); return () => heard.delete(fn); } };
  hubs.throws.setup = ({ app: a }) => { a.on(() => {}); throw new Error("late"); };
  await loadFeatures(["throws"], { load: async (n) => ({ default: hubs[n] }), app, has: () => true });
  assert.equal(heard.size, 0);
});

test("a feature whose backend module is not installed is hidden; unknown (server down) hides nothing", async () => {
  _reset(); offer("a", host());
  const mod = { id: "needs-render", mount: "a", modules: ["render"], setup() {} };
  const load = async () => ({ default: mod });
  const gone = await loadFeatures(["x"], { load, app: { hasModule: (id) => id !== "render" }, has: () => true });
  assert.deepEqual(gone.hidden.map((h) => h.why), ["module render is not installed"]);
  const here = await loadFeatures(["x"], { load, app: { hasModule: () => true }, has: () => true });
  assert.equal(here.mounted.length, 1);
  const unknown = await loadFeatures(["x"], { load, app: {}, has: () => true });      // no hasModule: nothing to judge by
  assert.equal(unknown.mounted.length, 1);
});
