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

test("a retry left over from a project that failed to load never reloads the one opened after it", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let opened, calls = 0, fail = true;
  const project = { project: { id: "a", models: { slots, removed: [], unwired: {}, whole: false } }, setField: () => {} };
  const pipeline = { adopt: async () => { calls += 1; return !fail; }, slots: () => slots, removedIds: () => [], unwiredMap: () => ({}), whole: () => false, subscribe: () => {} };
  linkPipeline({ project, pipeline, onOpen: (fn) => { opened = fn; } });
  await opened();                          // A fails: a retry is waiting
  fail = false;
  project.project = { id: "b", models: { slots, removed: [], unwired: {}, whole: false } };
  await opened();                          // B opens and loads
  const before = calls;
  t.mock.timers.tick(6000);
  await new Promise((r) => setImmediate(r));
  assert.equal(calls, before);
});

test("Undo of something else, with a pipeline edit still landing, keeps the edit and puts nothing back in", async () => {
  const writes = [];
  let opened, heard, land, adopts = 0;
  let live = slots;
  const models = { slots, removed: [], unwired: {}, whole: false };
  const project = { project: { id: "a", models }, setField: (k, v) => { writes.push(v.slots[0].inputs.ckpt_name); project.project[k] = v; } };
  const pipeline = { adopt: async () => { adopts += 1; return true; }, slots: () => live, removedIds: () => [], unwiredMap: () => ({}), whole: () => false,
    subscribe: (fn) => { heard = fn; }, settled: () => new Promise((r) => { land = r; }) };
  linkPipeline({ project, pipeline, onOpen: (fn) => { opened = fn; } });
  const first = opened(); land(); await first;
  const before = adopts;
  project.project = JSON.parse(JSON.stringify(project.project));          // Undo puts a copy of the project in place
  const undo = opened();
  live = [{ ...slots[0], inputs: { ckpt_name: "b" } }];                    // the edit's answer arrives
  heard(live);
  land();
  await undo;
  assert.deepStrictEqual([writes, adopts - before], [["b"], 0]);
});

test("opening an older project whose pipeline an update's defaults fill in does not rewrite it", async () => {
  const writes = [];
  let opened;
  const project = { project: { id: "o", models: { slots, removed: [], unwired: {} } }, setField: (k, v) => { writes.push(v); } };
  const filled = [{ ...slots[0], inputs: { ...slots[0].inputs, denoise: 1 } }];
  const pipeline = { adopt: async () => true, slots: () => filled, removedIds: () => [], unwiredMap: () => ({}), whole: () => false, subscribe: () => {} };
  linkPipeline({ project, pipeline, onOpen: (fn) => { opened = fn; } });
  await opened();
  assert.equal(writes.length, 0);
});

test("a project opened while ComfyUI is down is owned once it answers, so later edits are saved, and the toast is said once", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const writes = [], said = [];
  let opened, heard, up = false;
  const project = { project: { id: "a", models: { slots, removed: [], unwired: {}, whole: false } }, setField: (k, v) => { writes.push(v.slots[0].inputs.ckpt_name); project.project[k] = v; } };
  const pipeline = { adopt: async () => up, slots: () => (up ? slots : null), removedIds: () => [], unwiredMap: () => ({}), whole: () => false, subscribe: (fn) => { heard = fn; } };
  linkPipeline({ project, pipeline, onOpen: (fn) => { opened = fn; }, say: (x) => said.push(x) });
  const flush = async () => { for (let i = 0; i < 5; i++) await new Promise((r) => setImmediate(r)); };
  await opened();
  for (let i = 0; i < 5; i++) { t.mock.timers.tick(16000); await flush(); }
  assert.equal(said.length, 1);
  up = true;
  heard(slots);                              // a panel's own load got the pipeline in and announced it
  await flush();
  heard([{ ...slots[0], inputs: { ckpt_name: "edited" } }]);
  assert.deepStrictEqual(writes, ["edited"]);
});

test("a slow pipeline load is waited for, never retried over while it can still land", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let opened, calls = 0, finish;
  const project = { project: { id: "a", models: { slots, removed: [], unwired: {}, whole: false } }, setField: () => {} };
  const pipeline = { adopt: () => { calls += 1; return new Promise((r) => { finish = r; }); }, slots: () => slots, removedIds: () => [], unwiredMap: () => ({}), whole: () => false, subscribe: () => {} };
  linkPipeline({ project, pipeline, onOpen: (fn) => { opened = fn; } });
  const opening = opened();
  for (let i = 0; i < 4; i++) { t.mock.timers.tick(10000); await new Promise((r) => setImmediate(r)); }
  finish(true);
  await opening;
  assert.equal(calls, 1);
});

test("a project whose pipeline could not get through mid-session keeps being retried, and is owned once ComfyUI answers", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const writes = [];
  let opened, heard, up = false;
  const project = { project: { id: "a", models: { slots, removed: [], unwired: {}, whole: false } }, setField: (k, v) => { writes.push(v.slots[0].inputs.ckpt_name); project.project[k] = v; } };
  const pipeline = { adopt: async () => up, slots: () => slots, unreachable: () => !up, removedIds: () => [], unwiredMap: () => ({}), whole: () => false, subscribe: (fn) => { heard = fn; } };
  const link = linkPipeline({ project, pipeline, onOpen: (fn) => { opened = fn; } });
  const flush = async () => { for (let i = 0; i < 5; i++) await new Promise((r) => setImmediate(r)); };
  await opened();
  for (let i = 0; i < 4; i++) { t.mock.timers.tick(6000); await flush(); }
  assert.match(link.why(), /ComfyUI is not answering/);
  up = true;
  t.mock.timers.tick(16000); await flush();
  assert.equal(link.owns(), true);
  heard([{ ...slots[0], inputs: { ckpt_name: "edited" } }]);
  assert.deepStrictEqual(writes, ["edited"]);
});

test("a pipeline ComfyUI refuses says the server's reason, and the person's next edit becomes the project's pipeline", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const writes = [], said = [];
  let opened, heard;
  const project = { project: { id: "a", models: { slots, removed: [], unwired: {}, whole: false } }, setField: (k, v) => { writes.push(v.slots[0].inputs.ckpt_name); project.project[k] = v; } };
  const pipeline = { adopt: async () => false, slots: () => slots, unreachable: () => false, refusal: () => "slot 0 role 0 has no input",
    removedIds: () => [], unwiredMap: () => ({}), whole: () => false, subscribe: (fn) => { heard = fn; } };
  const link = linkPipeline({ project, pipeline, onOpen: (fn) => { opened = fn; }, say: (x) => said.push(x) });
  const flush = async () => { for (let i = 0; i < 5; i++) await new Promise((r) => setImmediate(r)); };
  await opened();
  for (let i = 0; i < 4; i++) { t.mock.timers.tick(6000); await flush(); }
  assert.match(said.join(" "), /refused .*slot 0 role 0 has no input/);
  assert.match(link.why(), /refused/);
  heard([{ ...slots[0], inputs: { ckpt_name: "fixed" } }]);
  assert.deepStrictEqual(writes, ["fixed"]);
});
