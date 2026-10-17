// The stage drawn over a picture, the cover over a monitor, the picture's look, and the other pieces added for features.
import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "./_dom.js";

let c;
test.before(async () => { setupDom(); ({ composer: c } = await import("../composer.js")); });
test.after(() => teardownDom());

const pointer = (node, type, { x = 0, y = 0 } = {}) =>
  node.dispatchEvent(new window.PointerEvent(type, { clientX: x, clientY: y, pointerId: 1, button: 0, bubbles: true, cancelable: true }));
const rect = (node, r) => { node.getBoundingClientRect = () => ({ left: 0, top: 0, right: r.width, bottom: r.height, ...r }); };

// A parent holding the picture area (400 x 200 on screen, a 4:2 picture) and the stage.
function monitor(handlers = {}) {
  const parent = document.createElement("div"), area = document.createElement("div");
  document.body.append(parent);
  parent.append(area);
  rect(parent, { width: 400, height: 200 }); rect(area, { width: 400, height: 200 });
  const stage = c.stage.overlay(handlers);
  parent.append(stage.node);
  return { stage, parent, area };
}
const text = (over = {}) => ({ id: "t1", kind: "text", text: " Hi ", fontSize: 20, x: 0.5, y: 0.5, ...over });

test("the stage fits the picture's aspect into the room it has and draws sizes at that scale", () => {
  const { stage } = monitor();
  assert.equal(stage.fit(200, 100), true);
  assert.deepEqual(stage.size, { w: 400, h: 200 });
  stage.setItems([text()]);
  const item = stage.node.querySelector(".cx-stage-item");
  assert.equal(item.textContent, "Hi", "text is trimmed, as the render trims it");
  assert.equal(item.style.fontSize, "40px", "20 picture pixels at a scale of 2");
  stage.destroy();
});

test("with no visible area there is nothing to fit, so the stage says so", () => {
  const parent = document.createElement("div");
  document.body.append(parent);
  const stage = c.stage.overlay();
  parent.append(stage.node);
  assert.equal(stage.fit(200, 100), false);
});

test("dragging an item reports its new place as fractions of the picture, clamped, and busy is true meanwhile", () => {
  const moved = [];
  const { stage } = monitor({ onMove: (id, to) => moved.push([id, to]) });
  stage.fit(200, 100);
  stage.setItems([text()]);
  const item = stage.node.querySelector(".cx-stage-item");
  pointer(item, "pointerdown", { x: 100, y: 100 });
  assert.equal(stage.busy, true);
  pointer(item, "pointermove", { x: 200, y: 100 });         // 100 px over a 400 px wide picture: a quarter of it
  assert.equal(item.style.left, "75%", "shown live while dragging");
  pointer(item, "pointermove", { x: 5000, y: 100 });
  pointer(item, "pointerup", { x: 5000, y: 100 });
  assert.equal(stage.busy, false);
  assert.deepEqual(moved, [["t1", { x: 1, y: 0.5 }]], "far past the edge stops at the edge");
});

test("the corner handle reports a growth factor, not a style; a cancelled gesture reports nothing but onCancel", () => {
  const grown = [], cancelled = [];
  const { stage } = monitor({ onResize: (id, f) => grown.push([id, f]), onCancel: () => cancelled.push(1) });
  stage.fit(200, 100);
  stage.setItems([text()]);
  const item = stage.node.querySelector(".cx-stage-item"), grip = item.querySelector(".cx-stage-grip");
  rect(item, { width: 100, height: 40 });
  pointer(grip, "pointerdown", { x: 0, y: 0 });
  pointer(grip, "pointermove", { x: 50, y: 0 });             // both sides grow: 100 -> 200 wide
  assert.equal(item.style.fontSize, "80px", "shown live while dragging");
  pointer(grip, "pointerup", { x: 50, y: 0 });
  assert.deepEqual(grown, [["t1", 2]]);
  pointer(grip, "pointerdown", { x: 0, y: 0 });
  pointer(grip, "pointermove", { x: 10, y: 0 });
  pointer(grip, "pointercancel", { x: 10, y: 0 });
  assert.equal(grown.length, 1);
  assert.equal(cancelled.length, 1);
});

test("a picture item has its width in picture pixels and keeps its aspect unless a height is given", () => {
  const { stage } = monitor();
  stage.fit(200, 100);
  stage.setItems([{ id: "i", kind: "image", src: "/x.png", widthPx: 50, heightPx: null }, { id: "j", kind: "image", src: "/y.png", widthPx: 50, heightPx: 20 }]);
  const [a, b] = [...stage.node.querySelectorAll("img")];
  assert.equal(a.style.width, "100px"); assert.equal(a.style.height, "");
  assert.equal(b.style.height, "40px");
});

test("the cover shows a file over its parent, closes on ✕ or close(), and says so once", () => {
  let closed = 0;
  const cover = c.viewer.cover({ onClose: () => { closed += 1; } });
  document.body.append(cover.node);
  assert.equal(cover.isOpen, false);
  cover.show({ src: "/a.mp4", kind: "video" });
  assert.equal(cover.isOpen, true);
  assert.equal(cover.node.querySelector("video").hasAttribute("controls"), true);
  cover.node.querySelector("button").click();
  assert.equal(cover.isOpen, false); assert.equal(closed, 1);
  cover.close();
  assert.equal(closed, 1, "closing what is shut is not a close");
  cover.show({ src: "/s.wav", kind: "audio" });
  assert.ok(cover.node.querySelector("audio"));
  cover.show({});
  assert.equal(cover.isOpen, false, "nothing to show, nothing shown");
});

test("the viewer keeps its look across a change of source", () => {
  const v = c.viewer.media({ kind: "video" });
  v.setSource("/a.mp4", "video");
  v.setLook({ transform: "scale(1.5)", filter: "blur(2px)", fit: "cover", opacity: 0.5 });
  const style = () => v.element.style;
  assert.deepEqual([style().transform, style().filter, style().objectFit, style().opacity], ["scale(1.5)", "blur(2px)", "cover", "0.5"]);
  v.setSource("/b.mp4", "video");
  assert.equal(style().transform, "scale(1.5)", "the new picture is drawn the same way");
  v.setLook(null);
  assert.equal(style().transform, "");
});

test("a slider whose top changes draws its fill against the new top", () => {
  const s = c.slider.sm({ label: "Position", min: 0, max: 1, step: 0.01 });
  s.setMax(10); s.setValue(5);
  assert.equal(s.input.style.getPropertyValue("--fill"), "50%");
});

test("a corner button follows its box, and goes (once) when the box does", () => {
  const box = document.createElement("textarea");
  document.body.append(box);
  rect(box, { left: 100, right: 400, top: 200, bottom: 300, width: 300, height: 100 });
  let gone = 0;
  const corner = c.floating.corner({ anchor: box, label: "💡", onGone: () => { gone += 1; } });
  rect(corner.node, { width: 24, height: 20 });
  return new Promise((resolve) => setTimeout(() => {
    assert.deepEqual([corner.node.style.left, corner.node.style.top], ["360px", "276px"]);
    box.remove();
    setTimeout(() => {
      assert.equal(gone, 1);
      assert.equal(corner.node.isConnected, false);
      resolve();
    }, 60);
  }, 60));
});

test("tiles carry a state, refuse an unknown one, and say what the dot means", () => {
  const ready = c.tile.node({ title: "Load", sub: "m.safetensors" });
  assert.ok(ready.node.querySelector(".cx-tile-state"));
  assert.equal(ready.node.querySelector(".cx-tile-state").title, "Ready");
  const needs = c.tile.node({ title: "Load", sub: "", state: "needs" });
  assert.equal(needs.node.querySelector(".cx-tile-state").title, "Needs attention");
  assert.throws(() => c.tile.node({ title: "x", state: "bogus" }), /Unknown tile state/);
  let hit = 0;
  c.tile.action({ title: "Go", sub: "now", onClick: () => { hit += 1; } }).node.click();
  assert.equal(hit, 1);
});

test("a hero, an inline progress and a settings row's hint can be changed after the fact", () => {
  const row = c.settingsRow.default({ label: "Update", hint: "Checking…", control: c.button.sm({ label: "Update" }) });
  row.setHint("Up to date");
  assert.match(row.node.textContent, /Up to date/);
  const bar = c.progress.inline({ label: "Generating" });
  bar.setText("Step 1/2"); bar.setValue(50); bar.setHidden(true);
  assert.equal(bar.node.hidden, true); assert.match(bar.node.textContent, /Step 1\/2/);
  assert.match(c.hero.default({ title: "FunPack 5", hint: "Cutting Room" }).node.textContent, /FunPack 5.*Cutting Room/);
});
