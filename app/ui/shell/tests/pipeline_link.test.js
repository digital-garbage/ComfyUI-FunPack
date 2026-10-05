import test from "node:test";
import assert from "node:assert/strict";
import { linkPipeline } from "../pipeline_link.js";

const slots = [{ id: "model", node: "N", inputs: { ckpt_name: "a" } }];

function rig(models) {
  const writes = [];
  let opened, heard;
  const project = { project: { models }, setField: (k, v) => { writes.push(v); project.project[k] = v; } };
  const pipeline = { adopt: async () => true, slots: () => slots, removedIds: () => [], unwiredMap: () => ({}), subscribe: (fn) => { heard = fn; } };
  linkPipeline({ project, pipeline, onOpen: (fn) => { opened = fn; } });
  return { writes, open: () => opened(), edit: (s) => heard(s) };
}

test("opening a project whose pipeline is unchanged does not rewrite it; an edit does", async () => {
  const r = rig({ slots, removed: [], unwired: {} });
  await r.open();
  assert.equal(r.writes.length, 0);
  r.edit([{ ...slots[0], inputs: { ckpt_name: "b" } }]);
  assert.equal(r.writes.length, 1);
});

test("a project with no saved pipeline gets the live one on open", async () => {
  const r = rig(undefined);
  await r.open();
  assert.deepEqual(r.writes[0].slots, slots);
});
