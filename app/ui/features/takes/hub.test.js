import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let hub;
test.before(async () => { setupDom(); hub = (await import("./index.js")).default; });
test.after(() => teardownDom());

test("it adds a Takes button and a Scene-tab section that steps through the takes, and removes both when torn down", () => {
  const doc = { scenes: [{ id: "a", rating: "" }], scene_renders: { a: { media: { filename: "2.mp4" }, promptId: "p2" } },
    scene_variants: { a: [{ media: { filename: "1.mp4" }, promptId: "p1", rating: "" }, { media: { filename: "2.mp4" }, promptId: "p2", rating: "" }] } };
  const project = { project: doc, selected: doc.scenes[0], edit: (fn) => fn(doc) };
  const app = { project, selection: { ids: [] }, runner: { units() {} }, sceneSections: [] };
  const host = document.createElement("div");
  const off = hub.setup({ host, app });
  assert.match(host.textContent, /Takes/);
  const [section] = app.sceneSections;
  assert.equal(section.key(doc.scenes[0], doc), "2|1|");
  assert.equal(section.rows(doc.scenes[0], doc).length, 2);
  doc.scene_variants.a.pop();
  assert.deepEqual(section.rows(doc.scenes[0], doc), [], "one take is nothing to step through");
  off();
  assert.equal(app.sceneSections.length, 0);
  assert.equal(host.children.length, 0);
});
