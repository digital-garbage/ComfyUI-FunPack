import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let hub;
test.before(async () => { setupDom(); hub = (await import("./index.js")).default; });
test.after(() => teardownDom());

test("choosing an image in the bin makes it the selected scene's source", async () => {
  const set = [];
  const project = { selected: { id: "s1" }, setScene: (...a) => set.push(a) };
  const api = { media: async () => ({ media: [{ id: "m1", name: "cat.png", kind: "image" }, { id: "m2", name: "a.wav", kind: "audio" }] }) };
  const host = document.createElement("div");
  hub.setup({ host, app: { project, api, on: () => () => {} } });
  await new Promise((r) => setTimeout(r, 10));
  const cells = [...host.querySelectorAll(".cx-cell")];
  assert.equal(cells.length, 2);
  const named = (n) => cells.find((x) => x.textContent.includes(n));
  named("cat.png").click();
  assert.deepEqual(set, [["s1", "source_image", "m1"]]);
  named("a.wav").click();                                   // audio: refused, nothing set
  assert.equal(set.length, 1);
});
