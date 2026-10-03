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
