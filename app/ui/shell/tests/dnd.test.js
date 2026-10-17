import test from "node:test";
import assert from "node:assert/strict";
import { onMediaDrop, MEDIA_DRAG } from "../dnd.js";

const fake = () => { const l = {}; return { addEventListener: (n, f) => { l[n] = f; }, removeEventListener: (n) => { delete l[n]; }, querySelector: () => null, l }; };
const ev = (payload, hit) => ({ dataTransfer: { types: payload ? [MEDIA_DRAG] : ["Files"], getData: () => JSON.stringify(payload), set dropEffect(v) {} }, target: { closest: () => hit }, preventDefault() { this.stopped = true; } });

test("a bin tile dropped on a matching target reaches the handler; files and other targets are left alone", () => {
  const doc = fake(), got = [];
  const off = onMediaDrop(".lane", (item, el) => got.push([item, el]), doc);
  const hit = {}, a = ev({ id: "m", kind: "image" }, hit);
  doc.l.dragover(a); doc.l.drop(a);
  assert.deepEqual(got, [[{ id: "m", kind: "image" }, hit]]);
  assert.equal(a.stopped, true);
  const b = ev(null, hit), c = ev({ id: "x" }, null);
  doc.l.drop(b); doc.l.drop(c);
  assert.equal(got.length, 1);
  off();
  assert.deepEqual(Object.keys(doc.l), []);
});

test("a target can be named, and the handler is told which clip is under the drop and which row the target is", () => {
  const lanes = [{}, {}, {}], clip = { dataset: { id: "s3" } };
  const doc = { ...fake(), querySelectorAll: () => lanes };
  const got = [];
  onMediaDrop("lane:video", (item, el, e, where) => got.push([el === lanes[1], where.clipId, where.row]), doc);
  const e = ev({ id: "m", kind: "image" }, lanes[1]);
  e.target.closest = (sel) => (sel.includes("cx-nle-clip") ? clip : lanes[1]);
  doc.l.drop(e);
  assert.deepEqual(got, [[true, "s3", 1]]);
});
