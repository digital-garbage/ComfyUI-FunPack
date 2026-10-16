// The pipeline client: what actually reaches /api/pipeline on the wire.
//
// pipeline_window.test.js injects a FAKE check(), so it can never catch a
// field this module forgets to forward -- which is exactly what happened:
// check()'s destructured signature had no `input`/`from_slot`/`from_output`,
// so a real wire()/unwire() call silently dropped them and the server
// answered "which input to unwire is named by a string, not a NoneType" to a
// button click that looked correct in every jsdom test. Caught live, in a
// browser, not by any suite -- this is the suite that would have caught it.

import test from "node:test";
import assert from "node:assert/strict";

import { check, load, describe, search } from "../pipeline.js";

function fetchRecording(response = { slots: [], refused: [], incomplete: [], queueable: true }) {
  const calls = [];
  const fetch = async (url, opts) => {
    calls.push({ url, opts, body: opts && opts.body ? JSON.parse(opts.body) : undefined });
    return { ok: true, json: async () => response };
  };
  return { fetch, calls };
}

test("wire's own fields all reach the request body", async () => {
  const { fetch, calls } = fetchRecording();
  await check({
    fetch, slots: [{ id: "a", node: "X", inputs: {} }],
    action: "wire", slot: "sampler", input: "model", from_slot: "loader", from_output: 0,
  });
  assert.deepEqual(calls[0].body, {
    action: "wire", slots: [{ id: "a", node: "X", inputs: {} }],
    slot: "sampler", input: "model", from_slot: "loader", from_output: 0,
  });
});

test("unwire's fields reach the request body, including a zero from_output on wire", async () => {
  const { fetch, calls } = fetchRecording();
  await check({ fetch, action: "unwire", slot: "sampler", input: "model" });
  assert.deepEqual(calls[0].body, { action: "unwire", slot: "sampler", input: "model" });

  const second = fetchRecording();
  await check({
    fetch: second.fetch, action: "wire", slot: "a", input: "x",
    from_slot: "b", from_output: 0,
  });
  // 0 is a legal output index and must not be dropped the way `undefined` is.
  assert.equal(second.calls[0].body.from_output, 0);
});

test("fields that were not passed are not sent as null or undefined", async () => {
  const { fetch, calls } = fetchRecording();
  await check({ fetch, action: "check" });
  assert.equal("input" in calls[0].body, false);
  assert.equal("from_slot" in calls[0].body, false);
  assert.equal("from_output" in calls[0].body, false);
  assert.equal("slot" in calls[0].body, false);
  assert.equal("node" in calls[0].body, false);
});

test("a refused check comes back with the server's reasons, not a thrown error", async () => {
  const fetch = async () => ({
    ok: false, status: 400,
    json: async () => ({ problems: ["which input to unwire is named by a string, not a NoneType"] }),
  });
  const result = await check({ fetch, action: "unwire", slot: "a" });
  assert.equal(result.queueable, false);
  assert.deepEqual(result.refused, ["which input to unwire is named by a string, not a NoneType"]);
});

test("load reads the pipeline as it stands", async () => {
  const fetch = async () => ({
    ok: true, json: async () => ({ slots: [{ id: "a" }], incomplete: ["x needs y"], queueable: false }),
  });
  const result = await load({ fetch });
  assert.deepEqual(result, { slots: [{ id: "a" }], incomplete: ["x needs y"], queueable: false });
});

test("load throws with the status when the server refuses to answer at all", async () => {
  const fetch = async () => ({ ok: false, status: 500 });
  await assert.rejects(() => load({ fetch }), /500/);
});

test("describe asks nothing for an empty list, and answers by class name", async () => {
  let asked = false;
  const fetch = async () => { asked = true; return { ok: true, json: async () => ({ nodes: {} }) }; };
  assert.deepEqual(await describe([], { fetch }), {});
  assert.equal(asked, false);

  const real = async (url) => ({
    ok: true, json: async () => ({ nodes: { A: { node: "A" }, B: null } }),
  });
  assert.deepEqual(await describe(["A", "B"], { fetch: real }), { A: { node: "A" }, B: null });
});

test("search answers with nodes and a total even when the server sends neither", async () => {
  const fetch = async () => ({ ok: true, json: async () => ({}) });
  assert.deepEqual(await search("x", { fetch }), { nodes: [], total: 0 });
});
