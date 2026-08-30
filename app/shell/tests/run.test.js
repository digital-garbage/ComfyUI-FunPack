// The run lifecycle, driven without a server.
//
// Everything the outside provides is injected, so these tests exercise the real
// file rather than a mock of it: a fake fetch returns what ComfyUI returns, and
// messages are the ones ComfyUI actually sends (checked against execution.py
// and comfy_execution/progress.py, not from memory).

import test from "node:test";
import assert from "node:assert/strict";

import { createRun, viewUrl, IDLE, QUEUED, RUNNING, DONE, FAILED, CANCELLED }
  from "../run.js";

const ok = (body) => async () => ({ ok: true, status: 200, json: async () => body });
const refuse = (status, body) => async () => ({ ok: false, status, json: async () => body });

const message = (type, data) => ({ data: JSON.stringify({ type, data }) });

function runner(fetchImpl, { clientId = "c1" } = {}) {
  const sent = [];
  const record = async (url, init) => {
    sent.push({ url, body: init && init.body ? JSON.parse(init.body) : null });
    return fetchImpl(url, init);
  };
  return { run: createRun({ fetch: record, clientId }), sent };
}

test("queueing a prompt reports it and keeps the id the server gave", async () => {
  const { run, sent } = runner(ok({ prompt_id: "p-1", number: 3 }));
  const id = await run.start({ "1": { class_type: "X", inputs: {} } });

  assert.equal(id, "p-1");
  assert.equal(run.state.phase, QUEUED);
  assert.equal(run.state.promptId, "p-1");
  assert.equal(sent[0].url, "/prompt");
  assert.equal(sent[0].body.client_id, "c1", "the socket would never hear about this run");
});

test("a graph the queue refuses fails with ComfyUI's own reason", async () => {
  const { run } = runner(refuse(400, {
    error: { type: "prompt_outputs_failed_validation", message: "Prompt outputs failed validation" },
    node_errors: { "4": { errors: [{ message: "Value not in list" }] } },
  }));

  await assert.rejects(() => run.start({}), /failed validation/);
  assert.equal(run.state.phase, FAILED);
  assert.match(run.state.error.message, /failed validation/);
  assert.ok(run.state.error.nodes["4"], "the node that was wrong was dropped");
});

test("a queue that cannot be reached says so rather than looking idle", async () => {
  // The dev server serves the app and has no queue behind it, so this is the
  // ordinary case while working on the UI, not an exotic one.
  const { run } = runner(async () => { throw new TypeError("Failed to fetch"); });
  await assert.rejects(() => run.start({}), /Failed to fetch/);
  assert.equal(run.state.phase, FAILED);
  assert.match(run.state.error.message, /could not be reached/);
});

test("progress follows the node that is working, not the sum of them", async () => {
  const { run } = runner(ok({ prompt_id: "p-1" }));
  await run.start({});

  run.handle(JSON.parse(message("execution_start", { prompt_id: "p-1" }).data));
  assert.equal(run.state.phase, RUNNING);

  run.handle(JSON.parse(message("progress_state", {
    prompt_id: "p-1",
    nodes: {
      "3": { state: "finished", value: 7, max: 7, node_id: "3", display_node_id: "3" },
      "5": { state: "running", value: 2, max: 20, node_id: "5", display_node_id: "5" },
    },
  }).data));

  assert.deepEqual(run.state.progress, { value: 2, max: 20 });
  assert.equal(run.state.node, "5");
});

test("images are collected from every save, not replaced by the last", async () => {
  const { run } = runner(ok({ prompt_id: "p-1" }));
  await run.start({});

  for (const name of ["a.png", "b.png"]) {
    run.handle(JSON.parse(message("executed", {
      prompt_id: "p-1",
      output: { images: [{ filename: name, subfolder: "", type: "output" }] },
    }).data));
  }
  run.handle(JSON.parse(message("execution_success", { prompt_id: "p-1" }).data));

  assert.equal(run.state.phase, DONE);
  assert.equal(run.state.images.length, 2);
  assert.deepEqual(run.images(), [
    "/view?filename=a.png&subfolder=&type=output",
    "/view?filename=b.png&subfolder=&type=output",
  ]);
});

test("another client's run does not drive this one", async () => {
  // execution_interrupted is BROADCAST, so a second tab cancelling its own run
  // would otherwise mark this one cancelled while it is still generating.
  const { run } = runner(ok({ prompt_id: "mine" }));
  await run.start({});
  run.handle(JSON.parse(message("execution_start", { prompt_id: "mine" }).data));

  run.handle(JSON.parse(message("execution_interrupted", { prompt_id: "theirs" }).data));
  assert.equal(run.state.phase, RUNNING, "somebody else's cancel stopped this run");

  run.handle(JSON.parse(message("execution_success", { prompt_id: "theirs" }).data));
  assert.equal(run.state.phase, RUNNING, "somebody else's result finished this run");

  run.handle(JSON.parse(message("execution_success", { prompt_id: "mine" }).data));
  assert.equal(run.state.phase, DONE);
});

test("a failure carries the node and the message ComfyUI gave", async () => {
  const { run } = runner(ok({ prompt_id: "p-1" }));
  await run.start({});
  run.handle(JSON.parse(message("execution_error", {
    prompt_id: "p-1", node_id: "9", node_type: "KSampler",
    exception_message: "Given groups=1, expected input to have 4 channels",
    traceback: ["line one", "line two"],
  }).data));

  assert.equal(run.state.phase, FAILED);
  assert.equal(run.state.error.node, "KSampler");
  assert.match(run.state.error.message, /4 channels/);
  assert.ok(run.state.error.traceback, "the traceback was thrown away");
});

test("cancelling asks the server and waits for it to say it stopped", async () => {
  const { run, sent } = runner(ok({ prompt_id: "p-1" }));
  await run.start({});
  run.handle(JSON.parse(message("execution_start", { prompt_id: "p-1" }).data));

  await run.cancel();
  assert.equal(sent[1].url, "/interrupt");
  assert.equal(sent[1].body.prompt_id, "p-1");
  // Claiming it early is how a UI shows "cancelled" over a run still burning
  // GPU time. The server says when it actually stopped.
  assert.equal(run.state.phase, RUNNING);

  run.handle(JSON.parse(message("execution_interrupted", { prompt_id: "p-1" }).data));
  assert.equal(run.state.phase, CANCELLED);
});

test("cancelling when nothing is running asks the server nothing", async () => {
  const { run, sent } = runner(ok({}));
  assert.equal(await run.cancel(), false);
  assert.equal(sent.length, 0, "an idle cancel interrupted whatever else was queued");
});

test("subscribers hear the current state immediately and on every change", async () => {
  const { run } = runner(ok({ prompt_id: "p-1" }));
  const seen = [];
  const stop = run.subscribe((s) => seen.push(s.phase));
  assert.deepEqual(seen, [IDLE], "a subscriber had to wait for a change to learn anything");

  await run.start({});
  assert.ok(seen.includes(QUEUED));
  stop();
  const before = seen.length;
  run.handle(JSON.parse(message("execution_success", { prompt_id: "p-1" }).data));
  assert.equal(seen.length, before, "unsubscribing did not stop the updates");
});

test("a listener that throws does not stop the others", async () => {
  const { run } = runner(ok({ prompt_id: "p-1" }));
  const seen = [];
  run.subscribe(() => { throw new Error("bad listener"); });
  run.subscribe((s) => seen.push(s.phase));
  await run.start({});
  assert.ok(seen.includes(QUEUED));
});

test("a binary frame and a malformed one leave the socket alone", async () => {
  const events = {};
  const socket = { addEventListener: (type, fn) => { events[type] = fn; } };
  const run = createRun({ fetch: ok({}), clientId: "c1", connect: () => socket });
  run.listen();

  events.message({ data: new Uint8Array([1, 2, 3]) });   // a preview image
  events.message({ data: "{not json" });
  assert.equal(run.state.phase, IDLE);
});

test("viewUrl escapes what goes into it", () => {
  const url = viewUrl({ filename: "a b&c.png", subfolder: "s/1", type: "temp" });
  assert.match(url, /filename=a\+b%26c\.png/);
  assert.match(url, /subfolder=s%2F1/);
  assert.match(url, /type=temp/);
});
