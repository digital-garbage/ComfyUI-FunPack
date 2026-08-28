import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom, fire, key } from "./_dom.js";
import { push, depth, _resetDismiss } from "../internals/dismiss.js";

test.beforeEach(() => { setupDom(); _resetDismiss(); });
test.after(() => teardownDom());

const panel = (id) => {
  const n = document.createElement("div");
  n.id = id;
  document.body.appendChild(n);
  return n;
};

test("Escape dismisses the top entry only", () => {
  // v4's bug: a modal and an autocomplete inside it each had their own Escape
  // handler, so one press closed both.
  const closed = [];
  const modal = panel("modal");
  const menu = panel("menu");
  push({ nodes: modal, onDismiss: () => closed.push("modal") });
  push({ nodes: menu, onDismiss: () => closed.push("menu") });

  key(document.body, "Escape");
  assert.deepEqual(closed, ["menu"]);
  assert.equal(depth(), 1);

  key(document.body, "Escape");
  assert.deepEqual(closed, ["menu", "modal"]);
  assert.equal(depth(), 0);
});

test("a click outside dismisses the top entry", () => {
  let closed = false;
  const menu = panel("menu");
  push({ nodes: menu, onDismiss: () => { closed = true; } });
  fire(document.body, "pointerdown");
  assert.equal(closed, true);
});

test("a click inside does not dismiss", () => {
  let closed = false;
  const menu = panel("menu");
  const child = document.createElement("button");
  menu.appendChild(child);
  push({ nodes: menu, onDismiss: () => { closed = true; } });
  fire(child, "pointerdown");
  assert.equal(closed, false);
});

test("the trigger counts as inside, so clicking it again does not close-then-reopen", () => {
  let closed = false;
  const menu = panel("menu");
  const trigger = panel("trigger");
  push({ nodes: [menu, trigger], onDismiss: () => { closed = true; } });
  fire(trigger, "pointerdown");
  assert.equal(closed, false);
});

test("a click inside the top entry leaves the one below alone", () => {
  const closed = [];
  const modal = panel("modal");
  const menu = panel("menu");
  push({ nodes: modal, onDismiss: () => closed.push("modal") });
  push({ nodes: menu, onDismiss: () => closed.push("menu") });
  fire(menu, "pointerdown");
  assert.deepEqual(closed, []);
});

test("closeOnEsc false ignores Escape but still takes outside clicks", () => {
  let closed = false;
  const modal = panel("modal");
  push({ nodes: modal, closeOnEsc: false, onDismiss: () => { closed = true; } });
  key(document.body, "Escape");
  assert.equal(closed, false);
  fire(document.body, "pointerdown");
  assert.equal(closed, true);
});

test("closeOnOutside false ignores clicks but still takes Escape", () => {
  let closed = false;
  const modal = panel("modal");
  push({ nodes: modal, closeOnOutside: false, onDismiss: () => { closed = true; } });
  fire(document.body, "pointerdown");
  assert.equal(closed, false);
  key(document.body, "Escape");
  assert.equal(closed, true);
});

test("a non-dismissable top entry blocks Escape reaching the one below", () => {
  // Otherwise a modal that declined Escape would still close when a child
  // overlay above it declined too.
  const closed = [];
  push({ nodes: panel("modal"), onDismiss: () => closed.push("modal") });
  push({ nodes: panel("sticky"), closeOnEsc: false, onDismiss: () => closed.push("sticky") });
  key(document.body, "Escape");
  assert.deepEqual(closed, []);
});

test("dismissing twice runs onDismiss once", () => {
  let n = 0;
  const h = push({ nodes: panel("menu"), onDismiss: () => { n += 1; } });
  h.dismiss();
  h.dismiss();
  assert.equal(n, 1);
});

test("release removes the entry without running onDismiss", () => {
  let n = 0;
  const h = push({ nodes: panel("menu"), onDismiss: () => { n += 1; } });
  h.release();
  assert.equal(n, 0);
  assert.equal(depth(), 0);
  key(document.body, "Escape");
  assert.equal(n, 0);
});

test("dismissing out of order keeps the rest of the stack intact", () => {
  const closed = [];
  const a = push({ nodes: panel("a"), onDismiss: () => closed.push("a") });
  push({ nodes: panel("b"), onDismiss: () => closed.push("b") });
  a.dismiss();
  assert.deepEqual(closed, ["a"]);
  assert.equal(depth(), 1);
  key(document.body, "Escape");
  assert.deepEqual(closed, ["a", "b"]);
});

test("isTop and depth describe the stack", () => {
  const a = push({ nodes: panel("a") });
  const b = push({ nodes: panel("b") });
  assert.equal(a.isTop, false);
  assert.equal(b.isTop, true);
  assert.equal(a.depth, 1);
  assert.equal(b.depth, 2);
});

test("listeners are removed once the stack empties", () => {
  // A stack that empties but keeps listening means every later key press walks
  // dead handlers, and the leak grows with every overlay ever opened.
  let added = 0;
  const realAdd = document.addEventListener.bind(document);
  const realRemove = document.removeEventListener.bind(document);
  document.addEventListener = (...args) => { added += 1; return realAdd(...args); };
  document.removeEventListener = (...args) => { added -= 1; return realRemove(...args); };

  const h = push({ nodes: panel("a") });
  assert.equal(added, 2);
  h.dismiss();
  assert.equal(added, 0);

  push({ nodes: panel("b") });
  assert.equal(added, 2, "listening resumes for a later overlay");
});

test("keys other than Escape are ignored", () => {
  let closed = false;
  push({ nodes: panel("a"), onDismiss: () => { closed = true; } });
  key(document.body, "Enter");
  key(document.body, "a");
  assert.equal(closed, false);
});

test("a virtual anchor never reaches the outside-click check", () => {
  // A context menu is anchored to coordinates, not an element. That stand-in
  // has no .contains, and passing it through threw on the first click anywhere.
  let closed = false;
  const point = { getBoundingClientRect: () => ({ x: 10, y: 10, width: 0, height: 0 }) };
  push({ nodes: [panel("menu"), point], onDismiss: () => { closed = true; } });
  assert.doesNotThrow(() => fire(document.body, "pointerdown"));
  assert.equal(closed, true);
});
