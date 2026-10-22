import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "./_dom.js";

// jsdom has no layout, so rects are supplied; what is under test is the
// contract: the node animates toward the origin, and teardown waits for it.
function rig() {
  setupDom();
  document.documentElement.style.setProperty("--dur-3", "260ms");
  window.matchMedia = () => ({ matches: false });
  const rect = (l, t, w, h) => ({ left: l, top: t, width: w, height: h, right: l + w, bottom: t + h });
  const node = document.createElement("div");
  node.getBoundingClientRect = () => rect(100, 100, 400, 300);
  const origin = document.createElement("button");
  origin.getBoundingClientRect = () => rect(300, 700, 80, 24);
  const runs = [];
  node.animate = (keys, opts) => { const a = { keys, opts, cancel() {} }; runs.push(a); return a; };
  return { node, origin, runs };
}

test("genie out animates in reverse toward the origin and tears down when it ends", async () => {
  const { genieOut } = await import("../internals/genie.js");
  const { node, origin, runs } = rig();
  let done = 0;
  genieOut(node, origin, () => { done += 1; });
  assert.equal(runs.length, 1);
  assert.equal(runs[0].opts.direction, "reverse");
  assert.match(runs[0].keys[0].transform, /translate\(40px, 462px\)/);       // button centre minus window centre
  assert.equal(done, 0);
  runs[0].onfinish();
  assert.equal(done, 1);
  teardownDom();
});

test("without an origin or without animate, teardown is immediate", async () => {
  const { genieOut } = await import("../internals/genie.js");
  const { node, origin } = rig();
  let done = 0;
  genieOut(node, null, () => { done += 1; });
  node.animate = undefined;
  genieOut(node, origin, () => { done += 1; });
  assert.equal(done, 2);
  teardownDom();
});
