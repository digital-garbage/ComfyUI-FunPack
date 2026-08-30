// Mounting a module's panel.
//
// The guarantee this file exists to check is stated in panels.js: a module that
// failed is indistinguishable from one nobody installed. No placeholder, no
// greyed row, nothing on screen. It had never been tested against a module that
// fails AFTER its panel is built, which is the only case where there is
// something on screen to leave behind.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let mountAll, offer, settle, resetMounts, resetValues;
test.before(async () => {
  setupDom();
  await import("../../composer/composer.js");
  ({ mountAll } = await import("../panels.js"));
  ({ offer, settle, _reset: resetMounts } = await import("../mounts.js"));
  ({ _reset: resetValues } = await import("../values.js"));
});
test.after(() => teardownDom());

function region() {
  resetMounts();
  resetValues();
  const host = document.createElement("div");
  document.body.replaceChildren(host);
  offer("generation.model", host);
  return host;
}

const spec = (over = {}) => ({
  id: "demo", title: "Demo", mount: "generation.model",
  settings: { steps: { type: "int", label: "Steps", default: 20 } },
  ...over,
});

test("a module whose setup throws leaves nothing on screen", async () => {
  const host = region();
  const { mounted, hidden } = await mountAll(
    { modules: [spec({ ui: "./boom.js" })] },
    { load: async () => ({ setup: () => { throw new Error("no"); } }) });

  assert.deepEqual(mounted, []);
  assert.equal(hidden.length, 1);
  assert.equal(host.childElementCount, 0,
    "a hidden module left its panel in the document");
});

test("a module whose ui.js will not import leaves nothing on screen", async () => {
  const host = region();
  const { hidden } = await mountAll(
    { modules: [spec({ ui: "./missing.js" })] },
    { load: async () => { throw new Error("404"); } });

  assert.equal(hidden.length, 1);
  assert.equal(host.childElementCount, 0);
});

test("a module that mounts is on screen, so the checks above can fail", async () => {
  // Without this the two tests above pass on a mountAll that mounts nothing.
  const host = region();
  const { mounted, hidden } = await mountAll({ modules: [spec()] });
  assert.deepEqual(hidden, []);
  assert.equal(mounted.length, 1);
  assert.equal(host.childElementCount, 1);
});

test("a module's setup still runs, and its teardown is kept", async () => {
  const host = region();
  let torn = 0;
  const { mounted } = await mountAll(
    { modules: [spec({ ui: "./ok.js" })] },
    { load: async () => ({ setup: () => () => { torn += 1; } }) });

  assert.equal(host.childElementCount, 1);
  mounted[0].destroy();
  assert.equal(torn, 1, "the unsubscribe setup() returned was dropped");
});

test("a region's stand-in comes down once something is in it", () => {
  // A line reading "modules appear here" left above the modules that appeared
  // is a region explaining itself to nobody.
  resetMounts();
  const host = document.createElement("div");
  const standIn = document.createElement("p");
  host.append(standIn);
  document.body.replaceChildren(host);
  offer("generation.model", host, standIn);

  settle();
  assert.ok(standIn.parentNode, "a region nothing mounted into lost its stand-in");

  host.append(document.createElement("div"));
  settle();
  assert.equal(standIn.parentNode, null, "the stand-in stayed above what mounted");
});

test("two names on one host share its stand-in", () => {
  // Five mount points are offered on the Generation panel. Taking the stand-in
  // down as each one mounted would take down a stand-in belonging to all five,
  // on behalf of the first.
  resetMounts();
  const host = document.createElement("div");
  const standIn = document.createElement("p");
  host.append(standIn);
  document.body.replaceChildren(host);
  offer("generation.model", host, standIn);
  offer("generation.sampling", host, standIn);

  settle();
  assert.ok(standIn.parentNode);
  host.append(document.createElement("div"));
  settle();
  assert.equal(standIn.parentNode, null);
});
