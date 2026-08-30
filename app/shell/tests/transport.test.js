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

// --- a refusal from the queue, which names the nodes ------------------------

test("a graph the queue refused says which node and why", () => {
  // run.js captured ComfyUI's node_errors from the start; describe() only ever
  // read the singular `node`, which is set by the WebSocket crash path and
  // never by this one. So every value refusal showed ComfyUI's top-level
  // string -- "Prompt outputs failed validation" -- over a pipeline that can
  // hold a dozen loaders, with nothing saying which to fix.
  const said = describe({
    phase: FAILED, images: [],
    error: {
      message: "Prompt outputs failed validation",
      nodes: {
        vae: {
          class_type: "FunPackVAELoader",
          errors: [{ type: "value_not_in_list", message: "Value not in list",
                     details: "vae_name: 'gone.safetensors' not in []" }],
        },
      },
    },
  });
  assert.match(said, /FunPackVAELoader/);
  assert.match(said, /vae/);
  assert.match(said, /Value not in list/);
  assert.match(said, /gone\.safetensors/);
});

test("a crash during execution still names its one node", () => {
  const said = describe({
    phase: FAILED, images: [],
    error: { node: "sampler", message: "shape mismatch" },
  });
  assert.equal(said, "sampler: shape mismatch");
});

test("a refusal with no node errors falls back to what it did say", () => {
  const said = describe({
    phase: FAILED, images: [],
    error: { message: "the queue could not be reached: fetch failed", nodes: {} },
  });
  assert.match(said, /could not be reached/);
});

test("a node error carrying nothing readable does not swallow the message", () => {
  // An entry with an empty errors array would otherwise return "" and read as
  // a run that failed for no reason at all.
  const said = describe({
    phase: FAILED, images: [],
    error: { message: "refused", nodes: { "4": { class_type: "X", errors: [] } } },
  });
  assert.equal(said, "refused");
});
