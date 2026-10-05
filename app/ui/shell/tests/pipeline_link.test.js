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

test("an older file guessed whole is not rewritten to record the guess", async () => {
  const writes = [];
  let opened;
  const project = { project: { models: { slots, removed: [], unwired: {} } }, setField: (k, v) => { writes.push(v); } };
  const pipeline = { adopt: async () => true, slots: () => slots, removedIds: () => [], unwiredMap: () => ({}), whole: () => true, subscribe: () => {} };
  linkPipeline({ project, pipeline, onOpen: (fn) => { opened = fn; } });
  await opened();
  assert.equal(writes.length, 0);
});

test("switching project while the first is still loading never writes into either the wrong pipeline", async () => {
  const writes = [];
  let opened, heard, release;
  const A = { id: "a", models: { slots: [{ ...slots[0], inputs: { ckpt_name: "A" } }], removed: [], unwired: {}, whole: false } };
  const B = { id: "b", models: { slots: [{ ...slots[0], inputs: { ckpt_name: "B" } }], removed: [], unwired: {}, whole: false } };
  let live = A.models.slots;
  const project = { project: A, setField: (k, v) => { writes.push([project.project.id, v.slots[0].inputs.ckpt_name]); project.project[k] = v; } };
  const pipeline = {
    adopt: async (saved) => { if (saved[0].inputs.ckpt_name === "A") await new Promise((r) => { release = r; }); live = saved; return true; },
    slots: () => live, removedIds: () => [], unwiredMap: () => ({}), whole: () => false, subscribe: (fn) => { heard = fn; },
  };
  linkPipeline({ project, pipeline, onOpen: (fn) => { opened = fn; } });
  const first = opened();
  heard(live);                              // a pipeline change while A is still loading
  project.project = B;
  await opened();
  release();
  await first;                              // A's late finish must not touch B
  heard([{ ...slots[0], inputs: { ckpt_name: "B2" } }]);
  assert.deepStrictEqual(writes, [["b", "B2"]]);
});

test("an Undo copy of the same project keeps receiving edits", async () => {
  const writes = [];
  let opened, heard;
  const project = { project: { id: "a", models: { slots, removed: [], unwired: {}, whole: false } }, setField: (k, v) => { writes.push(v); project.project[k] = v; } };
  const pipeline = { adopt: async () => true, slots: () => slots, removedIds: () => [], unwiredMap: () => ({}), whole: () => false, subscribe: (fn) => { heard = fn; } };
  linkPipeline({ project, pipeline, onOpen: (fn) => { opened = fn; } });
  await opened();
  project.project = JSON.parse(JSON.stringify(project.project));
  heard([{ ...slots[0], inputs: { ckpt_name: "b" } }]);
  assert.equal(writes.length, 1);
});

test("a new project keeps the preset pipeline it inherits as whole, so it reopens exactly", async () => {
  const writes = [];
  let opened;
  const project = { fresh: true, project: { id: "n", models: { slots: [] } }, setField: (k, v) => { writes.push(v); project.project[k] = v; } };
  const pipeline = { adopt: async () => true, slots: () => slots, removedIds: () => ["clip"], unwiredMap: () => ({}), whole: () => true, subscribe: () => {} };
  linkPipeline({ project, pipeline, onOpen: (fn) => { opened = fn; } });
  await opened();
  assert.equal(writes[0].whole, true);
});
