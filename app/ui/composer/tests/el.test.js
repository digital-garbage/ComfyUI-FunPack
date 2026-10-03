import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom } from "./_dom.js";
import { el, frag, clear } from "../internals/el.js";

test.beforeEach(() => setupDom());
test.after(() => teardownDom());

test("builds an element with class and text", () => {
  const n = el("button", { cls: "cx-btn", text: "Generate" });
  assert.equal(n.tagName, "BUTTON");
  assert.equal(n.className, "cx-btn");
  assert.equal(n.textContent, "Generate");
});

test("class accepts an array and drops empty entries", () => {
  assert.equal(el("div", { cls: ["a", null, "b", false] }).className, "a b");
});

test("text goes through asText, so markup stays text", () => {
  const n = el("div", { text: "<b>no</b>" });
  assert.equal(n.querySelector("b"), null);
  assert.equal(n.textContent, "<b>no</b>");
});

test("style and class attributes are refused", () => {
  // Presentation is the kit's. Allowing either through attrs would reopen the
  // exact hole the no-styling rule closes.
  assert.throws(() => el("div", { attrs: { style: "color:red" } }), TypeError);
  assert.throws(() => el("div", { attrs: { class: "sneaky" } }), TypeError);
});

test("inline event handler attributes are refused", () => {
  for (const name of ["onclick", "onLoad", "onerror"]) {
    assert.throws(() => el("div", { attrs: { [name]: "boom()" } }), TypeError);
  }
});

test("ordinary attributes are set, booleans become empty strings", () => {
  const n = el("input", { attrs: { type: "checkbox", disabled: true, "aria-label": "Pick" } });
  assert.equal(n.getAttribute("type"), "checkbox");
  assert.equal(n.getAttribute("disabled"), "");
  assert.equal(n.getAttribute("aria-label"), "Pick");
});

test("false and null attributes are omitted entirely", () => {
  const n = el("input", { attrs: { disabled: false, title: null } });
  assert.equal(n.hasAttribute("disabled"), false);
  assert.equal(n.hasAttribute("title"), false);
});

test("listeners and children are attached", () => {
  let clicks = 0;
  const child = el("span", { text: "x" });
  const n = el("div", { on: { click: () => { clicks += 1; } }, children: child });
  n.dispatchEvent(new window.Event("click"));
  assert.equal(clicks, 1);
  assert.equal(n.firstChild, child);
});

test("falsy children are skipped, so conditionals read naturally", () => {
  const n = el("div", { children: [el("i"), null, false, undefined] });
  assert.equal(n.childNodes.length, 1);
});

test("frag is detached, so a half-built tree never reaches the document", () => {
  const f = frag();
  f.appendChild(el("div"));
  assert.equal(f.nodeType, 11);
  assert.equal(document.body.childNodes.length, 0);
});

test("clear empties a node", () => {
  const n = el("div", { children: [el("i"), el("b")] });
  assert.equal(clear(n).childNodes.length, 0);
});
