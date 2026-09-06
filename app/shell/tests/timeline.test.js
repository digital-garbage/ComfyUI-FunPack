// The timeline: reorder by drag, and the zoom that changes clip width.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

/** jsdom has no real DataTransfer -- a minimal stand-in, shared across the
 *  events of ONE drag gesture, the same way the browser shares one. */
function fakeDataTransfer() {
  const store = {};
  return {
    effectAllowed: null,
    setData(type, value) { store[type] = String(value); },
    getData(type) { return store[type] || ""; },
    get types() { return Object.keys(store); },
  };
}
const dragFire = (node, type, dataTransfer) => {
  const evt = new window.Event(type, { bubbles: true, cancelable: true });
  evt.dataTransfer = dataTransfer;
  node.dispatchEvent(evt);
};

let createTimeline;
test.before(async () => {
  setupDom();
  await import("../../composer/composer.js");
  ({ createTimeline } = await import("../timeline.js"));
});
test.after(() => teardownDom());

/** Just enough of the store for the timeline to read and act on. */
function fakeProject(scenes) {
  let selected = scenes[0] ? scenes[0].id : null;
  const moved = [];
  return {
    get scenes() { return scenes; },
    get selectedId() { return selected; },
    get selected() { return scenes.find((s) => s.id === selected) || null; },
    video: {},
    moved,
    addScene() { const s = { id: `s${scenes.length}`, text: "", result: null, length: null }; scenes.push(s); selected = s.id; },
    removeScene(id) { const at = scenes.findIndex((s) => s.id === id); if (at >= 0) scenes.splice(at, 1); },
    move(id, by) {
      moved.push({ id, by });
      const at = scenes.findIndex((s) => s.id === id);
      const to = at + by;
      if (at < 0 || to < 0 || to >= scenes.length) return;
      const [scene] = scenes.splice(at, 1);
      scenes.splice(to, 0, scene);
    },
    select(id) { selected = id; },
  };
}

const scenes = (n) => Array.from({ length: n }, (_, i) => ({ id: `s${i}`, text: "", result: null, length: null }));

test("dragging a clip onto another reorders by the distance dropped, not just one step", () => {
  const project = fakeProject(scenes(4));
  const t = createTimeline({ project });
  document.body.append(t.node);

  const cells = t.node.querySelectorAll(".cx-strip-cell");
  assert.equal(cells.length, 4);

  // Drag the first clip (s0) onto the fourth cell's position.
  const dt = fakeDataTransfer();
  dragFire(cells[0], "dragstart", dt);
  dragFire(cells[3], "dragover", dt);
  dragFire(cells[3], "drop", dt);

  assert.deepEqual(project.moved, [{ id: "s0", by: 3 }]);
  assert.deepEqual(project.scenes.map((s) => s.id), ["s1", "s2", "s3", "s0"]);
  t.destroy();
});

test("a redraw mid-drag (a remove fired from the keyboard, say) still resolves to the clip actually picked up", () => {
  // The mouse held down and a keyboard shortcut are two different input
  // channels -- nothing stops project.removeScene firing, and the strip
  // redrawing, while a drag is still in flight. Tracking BY POSITION would
  // have this act on whatever now sits where the drag started, not on what
  // was actually grabbed.
  const project = fakeProject(scenes(4));           // s0 s1 s2 s3
  const t = createTimeline({ project });
  document.body.append(t.node);

  const dt = fakeDataTransfer();
  dragFire(t.node.querySelectorAll(".cx-strip-cell")[1], "dragstart", dt);   // grabs s1

  project.removeScene("s0");                        // s1 s2 s3 -- s1 is now cell 0
  t.draw();                                          // the redraw a real remove triggers

  const after = t.node.querySelectorAll(".cx-strip-cell");
  dragFire(after[2], "dragover", dt);                // drop on whatever cell now shows s3
  dragFire(after[2], "drop", dt);

  assert.deepEqual(project.moved, [{ id: "s1", by: 2 }],
    "moved the wrong clip -- position at dragstart, not the one actually dragged");
  assert.deepEqual(project.scenes.map((s) => s.id), ["s2", "s3", "s1"]);
  t.destroy();
});

test("dropping a clip on itself does nothing", () => {
  const project = fakeProject(scenes(3));
  const t = createTimeline({ project });
  document.body.append(t.node);

  const cells = t.node.querySelectorAll(".cx-strip-cell");
  const dt = fakeDataTransfer();
  dragFire(cells[1], "dragstart", dt);
  dragFire(cells[1], "drop", dt);

  assert.deepEqual(project.moved, []);
  t.destroy();
});

test("a drop outside any cell is not a reorder", () => {
  const project = fakeProject(scenes(3));
  const t = createTimeline({ project });
  document.body.append(t.node);

  const strip = t.node.querySelector(".cx-strip");
  const cells = t.node.querySelectorAll(".cx-strip-cell");
  const dt = fakeDataTransfer();
  dragFire(cells[0], "dragstart", dt);
  dragFire(strip, "drop", dt);     // the strip's own background, not a cell

  assert.deepEqual(project.moved, []);
  t.destroy();
});

test("dragend firing on a node a redraw already detached does not leave anything for an unrelated drop to pick up", () => {
  // dragend fires on the ORIGINAL source element, wherever it now is -- a
  // redraw mid-drag can detach it from the tree, and an event on a detached
  // node does not bubble to a listener on an ancestor it no longer has. A
  // closure variable set at dragstart and only ever cleared by that dragend
  // would be left stuck; the fix does not depend on dragend firing at all, so
  // there is nothing to leave stuck. This drop's dataTransfer belongs to a
  // DIFFERENT, unrelated drag session (its own fresh, empty one) -- exactly
  // what a later, disconnected drag looks like.
  const project = fakeProject(scenes(4));            // s0 s1 s2 s3
  const t = createTimeline({ project });
  document.body.append(t.node);

  const started = fakeDataTransfer();
  dragFire(t.node.querySelectorAll(".cx-strip-cell")[1], "dragstart", started);   // grabs s1

  project.addScene();                                 // unrelated change, still redraws
  t.draw();
  // No drop ever reached the strip for `started` -- dropped outside the
  // window, Escape, anything. Its dragend, even if it fires, cannot reach
  // this delegated listener once its source node is gone.

  const unrelated = fakeDataTransfer();                // a completely separate drag
  const after = t.node.querySelectorAll(".cx-strip-cell");
  dragFire(after[3], "drop", unrelated);

  assert.deepEqual(project.moved, [], "an unrelated drop moved a scene from an abandoned drag");
  t.destroy();
});

test("zoom sets the clip width and remembers it for next time", () => {
  window.localStorage.clear();
  const project = fakeProject(scenes(2));
  const t = createTimeline({ project });
  document.body.append(t.node);

  const strip = t.node.querySelector(".cx-strip");
  t.node.querySelector('[role="radio"][aria-checked="false"]').click(); // any non-default level
  assert.ok(strip.style.getPropertyValue("--strip-w"), "zoom did not touch the strip");
  assert.ok(window.localStorage.getItem("funpack.timeline.zoom"));
  t.destroy();
});

test("zoom picked earlier is applied when the timeline is rebuilt", () => {
  window.localStorage.setItem("funpack.timeline.zoom", "lg");
  const project = fakeProject(scenes(2));
  const t = createTimeline({ project });
  document.body.append(t.node);

  const strip = t.node.querySelector(".cx-strip");
  assert.equal(strip.style.getPropertyValue("--strip-w"), "128px");
  t.destroy();
  window.localStorage.clear();
});
