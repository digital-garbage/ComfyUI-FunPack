import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom } from "./_dom.js";
import { portal, mount, unmount, ROOT_ID, _resetPortal } from "../internals/portal.js";

test.beforeEach(() => { setupDom(); _resetPortal(); });
test.after(() => teardownDom());

test("there is exactly one overlay root, however often it is asked for", () => {
  const a = portal();
  const b = portal();
  assert.equal(a, b);
  assert.equal(document.querySelectorAll(`#${ROOT_ID}`).length, 1);
});

test("the root is a direct child of body", () => {
  // Anywhere deeper and it inherits a stacking context from app layout, which
  // is what makes z-index ladders stop working.
  assert.equal(portal().parentNode, document.body);
});

test("the root creates no stacking context of its own", () => {
  const root = portal();
  for (const prop of ["transform", "filter", "backdropFilter", "contain", "willChange"]) {
    assert.ok(!root.style[prop], `${prop} would trap every overlay inside the root`);
  }
});

test("the root does not swallow clicks, but its children take them", () => {
  const root = portal();
  assert.equal(root.style.pointerEvents, "none");
  const node = mount(document.createElement("div"));
  assert.equal(node.style.pointerEvents, "auto");
});

test("mounted nodes land on the root and can be removed", () => {
  const node = mount(document.createElement("div"));
  assert.equal(node.parentNode, portal());
  unmount(node);
  assert.equal(node.parentNode, null);
});

test("unmounting something that was never mounted is harmless", () => {
  assert.doesNotThrow(() => unmount(document.createElement("div")));
  assert.doesNotThrow(() => unmount(null));
});

test("the root is recreated if something removes it", () => {
  portal().remove();
  assert.equal(portal().isConnected, true);
});

test("an existing root in the document is adopted, not duplicated", () => {
  const existing = document.createElement("div");
  existing.id = ROOT_ID;
  document.body.appendChild(existing);
  _resetPortal();
  assert.equal(portal(), existing);
  assert.equal(document.querySelectorAll(`#${ROOT_ID}`).length, 1);
});
