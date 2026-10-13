import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let hub;
test.before(async () => { setupDom(); hub = (await import("./index.js")).default; });
test.after(() => teardownDom());

test("a scene's text changed elsewhere (the Story box) shows in the Prompt box, but not under the caret", async () => {
  let fire;
  const doc = { id: "P", frame_rate: 25, num_frames_per_scene: 49, scenes: [{ id: "a", text: "old" }], scene_renders: {} };
  const p = { project: doc, get scenes() { return doc.scenes; }, get selected() { return doc.scenes[0]; }, selectedId: "a",
    setText: () => {}, setScene: () => {}, edit: (fn) => fn(doc) };
  const host = document.createElement("div");
  document.body.append(host);
  hub.setup({ host, app: { project: p, on: (fn) => { fire = fn; return () => {}; }, api: {}, say: () => {} } });
  const box = () => [...host.querySelectorAll("textarea")][0];
  assert.equal(box().value, "old");
  doc.scenes[0].text = "from story";
  fire("change");
  assert.equal(box().value, "from story");
  box().focus();
  doc.scenes[0].text = "while typing";
  fire("change");
  assert.equal(box().value, "from story");          // not rebuilt under the caret
  box().blur();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(box().value, "while typing");        // caught up once the focus left
});

test("typing in the Prompt box, then pressing a button in the panel, does not rebuild the button out from under the click", async () => {
  let fire;
  const doc = { id: "P", frame_rate: 25, num_frames_per_scene: 49, scenes: [{ id: "a", text: "old" }], scene_renders: {} };
  const p = { project: doc, get scenes() { return doc.scenes; }, get selected() { return doc.scenes[0]; }, selectedId: "a",
    setText: () => {}, setScene: () => {}, edit: (fn) => fn(doc) };
  const host = document.createElement("div");
  document.body.append(host);
  hub.setup({ host, app: { project: p, on: (fn) => { fire = fn; return () => {}; }, api: {}, say: () => {} } });
  const box = host.querySelector("textarea");
  box.focus();
  doc.scenes[0].text = "typed here"; fire("change");                 // the box's own edit, as setText would announce it
  const button = box.closest("[aria-label=Properties]") ? [...box.closest("[aria-label=Properties]").querySelectorAll("button, input[type=checkbox]")].find((b) => b !== box) : null;
  assert.ok(button, "a control inside the panel body");
  button.focus();                                                     // the mouse-down moves focus to it
  await new Promise((r) => setTimeout(r, 0));
  assert.ok(button.isConnected, "the pressed button is still there for its mouse-up");
  host.remove();
});

test("each reference has a 'Pin reference to slot RN' row; ticking one pins it at its number, and the rows follow", () => {
  let fire;
  const doc = { id: "P", frame_rate: 25, num_frames_per_scene: 49, fixed_references: {}, scene_renders: {}, scenes: [{ id: "a", text: "t", references: ["m1", "m2"] }] };
  const p = { project: doc, get scenes() { return doc.scenes; }, get selected() { return doc.scenes[0]; }, selectedId: "a",
    setText: () => {}, setScene: () => {}, edit: (fn) => fn(doc) };
  const host = document.createElement("div");
  document.body.append(host);
  hub.setup({ host, app: { project: p, on: (fn) => { fire = fn; return () => {}; }, api: {}, say: () => {} } });
  const rows = () => [...host.querySelectorAll("label")].filter((l) => l.textContent.includes("Pin reference to slot"));
  const box = (i) => rows()[i].previousElementSibling;       // the checkbox is the label's sibling
  const names = () => rows().map((l) => l.textContent.match(/Pin reference to slot R\d+/)[0]);
  assert.deepEqual(names(), ["Pin reference to slot R1", "Pin reference to slot R2"]);
  box(1).click();
  assert.deepEqual(doc.fixed_references, { m2: 2 }, "pinned at the number it had");
  fire("change");
  assert.deepEqual(names(), ["Pin reference to slot R2", "Pin reference to slot R1"], "the pinned one keeps R2, the other takes R1");
  assert.ok(box(0).checked && !box(1).checked);
});
