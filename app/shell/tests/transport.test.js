// The transport row, driven by run states.
//
// The reason this is one component reading one state object: a button here, a
// progress bar there and a status somewhere else each get their own idea of
// what is happening, and the user believes whichever spoke last.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { IDLE, QUEUED, RUNNING, DONE, FAILED, CANCELLED } from "../run.js";

let createTransport, describe;
test.before(async () => {
  setupDom();
  await import("../../composer/composer.js");
  ({ createTransport, describe } = await import("../transport.js"));
});
test.after(() => teardownDom());

const state = (over = {}) => ({
  phase: IDLE, promptId: null, progress: null, images: [], audio: [],
  error: null, node: null, ...over,
});

const built = (handlers) => {
  const t = createTransport(handlers);
  document.body.appendChild(t.node);
  return t;
};

test("Generate is disabled while a run is in flight and comes back after", () => {
  const t = built({});
  t.draw(state({ phase: RUNNING }));
  assert.equal(t.generate.node.disabled, true);

  t.draw(state({ phase: DONE, images: [{ filename: "a.png" }] }));
  assert.equal(t.generate.node.disabled, false, "a finished run left Generate dead");
});

test("Cancel is only offered while there is something to cancel", () => {
  const t = built({});
  assert.equal(t.cancel.node.hasAttribute("hidden"), true, "idle offered a cancel");

  t.draw(state({ phase: QUEUED }));
  assert.equal(t.cancel.node.hasAttribute("hidden"), false);

  t.draw(state({ phase: CANCELLED }));
  assert.equal(t.cancel.node.hasAttribute("hidden"), true);
});

test("the progress bar appears only when there is a real measure", () => {
  const t = built({});
  // Running with nothing to measure yet: a bar at zero reads as "stuck", which
  // is worse than no bar.
  t.draw(state({ phase: RUNNING }));
  assert.equal(t.progress.node.hasAttribute("hidden"), true);

  t.draw(state({ phase: RUNNING, progress: { value: 3, max: 20 } }));
  assert.equal(t.progress.node.hasAttribute("hidden"), false);
  assert.equal(t.progress.node.getAttribute("aria-valuenow"), "3");
  assert.equal(t.progress.node.getAttribute("aria-valuemax"), "20");
});

test("the buttons report to the handlers they were given", () => {
  let generated = 0, cancelled = 0;
  const t = built({ onGenerate: () => { generated += 1; }, onCancel: () => { cancelled += 1; } });
  t.generate.node.click();
  t.cancel.node.click();
  assert.equal(generated, 1);
  assert.equal(cancelled, 1);
});

test("a run that saved nothing does not read the same as one that saved something", () => {
  // ComfyUI reports execution_success for a graph with no save node too. "Done"
  // over an empty preview is the kind of quiet wrong this project keeps finding.
  assert.match(describe(state({ phase: DONE, images: [] })), /nothing was saved/);
  assert.match(describe(state({ phase: DONE, images: [{}] })), /1 result/);
  assert.match(describe(state({ phase: DONE, images: [{}, {}] })), /2 results/);
});

test("a failure says which node and what it said", () => {
  const text = describe(state({
    phase: FAILED,
    error: { node: "KSampler", message: "expected 4 channels" },
  }));
  assert.match(text, /KSampler/);
  assert.match(text, /4 channels/);
});

test("every phase says something, including ones added later", () => {
  for (const phase of [IDLE, QUEUED, RUNNING, DONE, FAILED, CANCELLED, "something-new"]) {
    const text = describe(state({ phase }));
    assert.ok(text && text.trim(), `${phase} draws an empty status`);
  }
});
