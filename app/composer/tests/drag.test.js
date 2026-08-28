import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom } from "./_dom.js";
import { drag } from "../internals/drag.js";

test.beforeEach(() => setupDom());
test.after(() => teardownDom());

function handle() {
  const n = document.createElement("div");
  document.body.appendChild(n);
  // jsdom has PointerEvent but not the capture API; the kit guards for both, so
  // record the calls to prove the guards do the right thing.
  n.captured = [];
  n.setPointerCapture = (id) => n.captured.push(id);
  n.releasePointerCapture = (id) => { n.captured = n.captured.filter((c) => c !== id); };
  n.hasPointerCapture = (id) => n.captured.includes(id);
  return n;
}

const pointer = (node, type, { x = 0, y = 0, id = 1, button = 0 } = {}) =>
  node.dispatchEvent(new window.PointerEvent(type, {
    clientX: x, clientY: y, pointerId: id, button, bubbles: true, cancelable: true,
  }));

test("reports movement relative to where the drag began", () => {
  const h = handle();
  const moves = [];
  drag(h, { onMove: (m) => moves.push([m.dx, m.dy]) });

  pointer(h, "pointerdown", { x: 100, y: 100 });
  pointer(h, "pointermove", { x: 130, y: 90 });
  pointer(h, "pointermove", { x: 100, y: 100 });

  assert.deepEqual(moves, [[30, -10], [0, 0]]);
});

test("start, move and end all fire, with the final delta on end", () => {
  const h = handle();
  const seen = [];
  drag(h, {
    onStart: () => seen.push("start"),
    onMove: () => seen.push("move"),
    onEnd: (e) => seen.push(`end:${e.dx},${e.dy}`),
  });
  pointer(h, "pointerdown", { x: 10, y: 10 });
  pointer(h, "pointermove", { x: 15, y: 20 });
  pointer(h, "pointerup", { x: 25, y: 30 });
  assert.deepEqual(seen, ["start", "move", "end:15,20"]);
});

test("the pointer is captured for the drag and released after", () => {
  // Capture is what guarantees a pointerup even when the cursor leaves the
  // window; without it a drag can be left running forever.
  const h = handle();
  drag(h);
  pointer(h, "pointerdown", { x: 0, y: 0, id: 7 });
  assert.deepEqual(h.captured, [7]);
  pointer(h, "pointerup", { x: 0, y: 0, id: 7 });
  assert.deepEqual(h.captured, []);
});

test("moves before a pointerdown are ignored", () => {
  const h = handle();
  let moved = false;
  drag(h, { onMove: () => { moved = true; } });
  pointer(h, "pointermove", { x: 50, y: 50 });
  assert.equal(moved, false);
});

test("moves after the drag ends are ignored", () => {
  const h = handle();
  let moves = 0;
  drag(h, { onMove: () => { moves += 1; } });
  pointer(h, "pointerdown", { x: 0, y: 0 });
  pointer(h, "pointermove", { x: 5, y: 5 });
  pointer(h, "pointerup", { x: 5, y: 5 });
  pointer(h, "pointermove", { x: 99, y: 99 });
  assert.equal(moves, 1);
});

test("a second pointer does not hijack a drag in progress", () => {
  // Two fingers on a trim handle would otherwise interleave deltas from both.
  const h = handle();
  const moves = [];
  drag(h, { onMove: (m) => moves.push(m.dx) });
  pointer(h, "pointerdown", { x: 0, y: 0, id: 1 });
  pointer(h, "pointermove", { x: 10, y: 0, id: 2 });
  pointer(h, "pointermove", { x: 20, y: 0, id: 1 });
  assert.deepEqual(moves, [20]);
});

test("a second pointerdown while dragging is ignored", () => {
  const h = handle();
  let starts = 0;
  drag(h, { onStart: () => { starts += 1; } });
  pointer(h, "pointerdown", { x: 0, y: 0, id: 1 });
  pointer(h, "pointerdown", { x: 5, y: 5, id: 2 });
  assert.equal(starts, 1);
});

test("only the configured button starts a drag", () => {
  const h = handle();
  let starts = 0;
  drag(h, { onStart: () => { starts += 1; } });
  pointer(h, "pointerdown", { x: 0, y: 0, button: 2 });   // right-click
  assert.equal(starts, 0);
  pointer(h, "pointerdown", { x: 0, y: 0, button: 0 });
  assert.equal(starts, 1);
});

test("pointercancel ends the drag and says it was cancelled", () => {
  const h = handle();
  let end = null;
  drag(h, { onEnd: (e) => { end = e; } });
  pointer(h, "pointerdown", { x: 0, y: 0 });
  pointer(h, "pointercancel", { x: 9, y: 9 });
  assert.equal(end.cancelled, true);
  assert.deepEqual(h.captured, [], "capture is released on cancel too");
});

test("pointerdown is prevented, so a drag does not also select text", () => {
  const h = handle();
  drag(h);
  const event = new window.PointerEvent("pointerdown", {
    clientX: 0, clientY: 0, pointerId: 1, button: 0, bubbles: true, cancelable: true,
  });
  h.dispatchEvent(event);
  assert.equal(event.defaultPrevented, true);
});

test("dispose stops listening, mid-drag included", () => {
  const h = handle();
  let moves = 0;
  const dispose = drag(h, { onMove: () => { moves += 1; } });
  pointer(h, "pointerdown", { x: 0, y: 0 });
  dispose();
  pointer(h, "pointermove", { x: 10, y: 10 });
  assert.equal(moves, 0);
  assert.deepEqual(h.captured, [], "a disposed drag does not keep the pointer");
});

test("a drag with no callbacks does not throw", () => {
  const h = handle();
  drag(h);
  assert.doesNotThrow(() => {
    pointer(h, "pointerdown", { x: 0, y: 0 });
    pointer(h, "pointermove", { x: 1, y: 1 });
    pointer(h, "pointerup", { x: 1, y: 1 });
  });
});
