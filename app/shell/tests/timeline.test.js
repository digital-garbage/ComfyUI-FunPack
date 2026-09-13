// The timeline: real seconds, drag reorder, and the zoom that changes clip
// width.

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

/** Just enough of the store for the timeline to read and act on. Video length
 *  is 24 frames at the 24fps fallback -- one second per scene by default,
 *  which keeps the maths in each test readable. */
function fakeProject(scenes, video = { length: 24 }) {
  let selected = scenes[0] ? scenes[0].id : null;
  const moved = [];
  return {
    get scenes() { return scenes; },
    get selectedId() { return selected; },
    get selected() { return scenes.find((s) => s.id === selected) || null; },
    video,
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
    select(id) { if (!scenes.some((s) => s.id === id) || id === selected) return; selected = id; },
  };
}

const scenes = (n) => Array.from({ length: n }, (_, i) => ({ id: `s${i}`, text: "", result: null, length: null }));

test("dragging a clip onto another reorders by the distance dropped, not just one step", () => {
  const project = fakeProject(scenes(4));
  const t = createTimeline({ project });
  document.body.append(t.node);

  const cells = t.node.querySelectorAll(".cx-track-clip");
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
  // channels -- nothing stops project.removeScene firing, and the track
  // redrawing, while a drag is still in flight. Tracking BY POSITION would
  // have this act on whatever now sits where the drag started, not on what
  // was actually grabbed.
  const project = fakeProject(scenes(4));           // s0 s1 s2 s3
  const t = createTimeline({ project });
  document.body.append(t.node);

  const dt = fakeDataTransfer();
  dragFire(t.node.querySelectorAll(".cx-track-clip")[1], "dragstart", dt);   // grabs s1

  project.removeScene("s0");                        // s1 s2 s3 -- s1 is now cell 0
  t.draw();                                          // the redraw a real remove triggers

  const after = t.node.querySelectorAll(".cx-track-clip");
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

  const cells = t.node.querySelectorAll(".cx-track-clip");
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

  const track = t.node.querySelector(".cx-track");
  const cells = t.node.querySelectorAll(".cx-track-clip");
  const dt = fakeDataTransfer();
  dragFire(cells[0], "dragstart", dt);
  dragFire(track, "drop", dt);     // the track's own background, not a cell

  assert.deepEqual(project.moved, []);
  t.destroy();
});

test("dragend firing on a node a redraw already detached does not leave anything for an unrelated drop to pick up", () => {
  const project = fakeProject(scenes(4));            // s0 s1 s2 s3
  const t = createTimeline({ project });
  document.body.append(t.node);

  const started = fakeDataTransfer();
  dragFire(t.node.querySelectorAll(".cx-track-clip")[1], "dragstart", started);   // grabs s1

  project.addScene();                                 // unrelated change, still redraws
  t.draw();

  const unrelated = fakeDataTransfer();                // a completely separate drag
  const after = t.node.querySelectorAll(".cx-track-clip");
  dragFire(after[3], "drop", unrelated);

  assert.deepEqual(project.moved, [], "an unrelated drop moved a scene from an abandoned drag");
  t.destroy();
});

test("zoom sets pixels-per-second and remembers it for next time", () => {
  window.localStorage.clear();
  const project = fakeProject(scenes(2));
  const t = createTimeline({ project });
  document.body.append(t.node);

  const before = parseFloat(t.node.querySelector(".cx-track-clip").style.width);
  t.node.querySelector('[role="radio"][aria-checked="false"]').click(); // any non-default level
  const after = parseFloat(t.node.querySelector(".cx-track-clip").style.width);
  assert.notEqual(after, before, "zoom did not change a clip's width");
  assert.ok(window.localStorage.getItem("funpack.timeline.zoom"));
  t.destroy();
});

test("zoom picked earlier is applied when the timeline is rebuilt", () => {
  window.localStorage.setItem("funpack.timeline.zoom", "lg");
  const project = fakeProject(scenes(2));             // 1s scenes, lg = 80px/s
  const t = createTimeline({ project });
  document.body.append(t.node);

  assert.equal(t.node.querySelector(".cx-track-clip").style.width, "80px");
  t.destroy();
  window.localStorage.clear();
});

test("an excluded scene is dimmed in the track, an included one is not", () => {
  const project = fakeProject(scenes(2));
  project.scenes[1].excluded = true;
  const t = createTimeline({ project });
  document.body.append(t.node);

  const cells = t.node.querySelectorAll(".cx-track-clip");
  assert.equal(cells[0].classList.contains("cx-excluded"), false);
  assert.equal(cells[1].classList.contains("cx-excluded"), true);
  t.destroy();
});

// --- real time ---------------------------------------------------------

test("a clip's width comes from its own length in frames, not a fixed size", () => {
  const s = scenes(2);
  s[0].length = 48;   // 2s at the 24fps fallback
  s[1].length = 24;   // 1s
  const project = fakeProject(s);
  const t = createTimeline({ project });
  document.body.append(t.node);

  const widths = [...t.node.querySelectorAll(".cx-track-clip")].map((c) => parseFloat(c.style.width));
  // md zoom = 40px/s.
  assert.deepEqual(widths, [80, 40]);
  t.destroy();
});

test("a scene with no length of its own uses the project's", () => {
  const s = scenes(1);
  const project = fakeProject(s, { length: 48 });   // 2s at 24fps
  const t = createTimeline({ project });
  document.body.append(t.node);

  assert.equal(parseFloat(t.node.querySelector(".cx-track-clip").style.width), 80);
  t.destroy();
});

test("clicking a scene's clip selects it and reports the change", () => {
  const seen = [];
  const project = fakeProject(scenes(3));            // 1s scenes at 40px/s: s0 0-40, s1 40-80, s2 80-120
  const t = createTimeline({ project, onSelect: (scene) => seen.push(scene && scene.id) });
  document.body.append(t.node);

  t.node.querySelector(".cx-track-inner")
    .dispatchEvent(new window.MouseEvent("click", { bubbles: true, clientX: 100 }));
  assert.equal(project.selectedId, "s2");
  assert.deepEqual(seen, ["s2"]);
  t.destroy();
});

test("the playhead follows a selection made from outside the track", () => {
  // Add/Remove/Move and a pick made elsewhere in the app all just change
  // project.selectedId -- the same path this exercises directly, rather than
  // depending on the fake store's own reselection fallback.
  const project = fakeProject(scenes(3));            // 1s scenes: s0 0-1, s1 1-2, s2 2-3
  const t = createTimeline({ project });
  document.body.append(t.node);

  t.node.querySelector(".cx-track-inner")                 // s2 starts at 80px (2s)
    .dispatchEvent(new window.MouseEvent("click", { bubbles: true, clientX: 80 }));
  assert.equal(t.node.querySelector(".cx-track-playhead").style.insetInlineStart, "80px");

  project.select("s0");
  t.draw();
  assert.equal(t.node.querySelector(".cx-track-playhead").style.insetInlineStart, "0px",
    "the playhead did not follow a selection made outside the track");
  t.destroy();
});

test("clicking mid-clip moves the playhead to that exact second without changing the selection", () => {
  const project = fakeProject(scenes(2));             // s0 0-1s, s1 1-2s
  const t = createTimeline({ project });               // starts selected on s0
  document.body.append(t.node);

  const inner = t.node.querySelector(".cx-track-inner");
  // 20px at 40px/s = 0.5s, inside s0's own span -- a pure seek, no reselect.
  inner.dispatchEvent(new window.MouseEvent("click", { bubbles: true, clientX: 20 }));

  assert.equal(project.selectedId, "s0", "a click within the selected clip changed the selection");
  assert.equal(t.node.querySelector(".cx-track-playhead").style.insetInlineStart, "20px");
  t.destroy();
});
