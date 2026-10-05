import test from "node:test";
import assert from "node:assert/strict";
import { linkPipeline } from "../pipeline_link.js";

const slots = [{ id: "model", node: "N", inputs: { ckpt_name: "a" } }];

function rig(models) {
  const writes = [];
  let opened, heard;
  const project = { project: { models }, setField: (k, v) => { writes.push(v); project.project[k] = v; } };
  const pipeline = { adopt: async () => true, slots: () => slots, removedIds: () => [], unwiredMap: () => ({}), whole: () => false, subscribe: (fn) => { heard = fn; } };
  linkPipeline({ project, pipeline, onOpen: (fn) => { opened = fn; } });
  return { writes, open: () => opened(), edit: (s) => heard(s) };
}

test("opening a project whose pipeline is unchanged does not rewrite it; an edit does", async () => {
  const r = rig({ slots, removed: [], unwired: {}, whole: false });
  await r.open();
  assert.equal(r.writes.length, 0);
  r.edit([{ ...slots[0], inputs: { ckpt_name: "b" } }]);
  assert.equal(r.writes.length, 1);
});

test("an older project with no saved pipeline gets the default and is not rewritten; a new one keeps the live pipeline", async () => {
  for (const fresh of [false, true]) {
    const writes = [], asked = [];
    let opened;
    const project = { fresh, project: { models: { slots: [] } }, setField: (k, v) => { writes.push(v); project.project[k] = v; } };
    const pipeline = { adopt: async (...a) => { asked.push(a[4]); return true; }, slots: () => slots, removedIds: () => [], unwiredMap: () => ({}), whole: () => false, subscribe: () => {} };
    linkPipeline({ project, pipeline, onOpen: (fn) => { opened = fn; } });
    await opened();
    assert.deepStrictEqual([asked[0], writes.length], [fresh, fresh ? 1 : 0]);
  }
});

test("an older file with no whole flag is not rewritten just to add one", async () => {
  const r = rig({ slots, removed: [], unwired: {} });
  await r.open();
  assert.equal(r.writes.length, 0);
});
