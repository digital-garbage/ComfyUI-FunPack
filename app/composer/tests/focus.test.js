import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom } from "./_dom.js";
import { focusables, roving, trap } from "../internals/focus.js";

test.beforeEach(() => setupDom());
test.after(() => teardownDom());

function box(html) {
  const n = document.createElement("div");
  n.innerHTML = html;                 // test fixture, not kit code
  document.body.appendChild(n);
  return n;
}

const tab = (node, shift = false) =>
  node.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Tab", shiftKey: shift, bubbles: true, cancelable: true }));

const arrow = (node, key) =>
  node.dispatchEvent(new window.KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }));

test("focusables skips disabled and hidden controls", () => {
  const n = box(`<button>a</button><button disabled>b</button>
    <input hidden><a href="#">c</a><span tabindex="-1">d</span><div aria-hidden="true"><button>e</button></div>`);
  assert.deepEqual(focusables(n).map((x) => x.textContent || x.tagName), ["a", "c"]);
});

test("focus moves into the container on trap", () => {
  const n = box("<button>a</button><button>b</button>");
  trap(n);
  assert.equal(document.activeElement.textContent, "a");
});

test("Tab wraps from last back to first", () => {
  const n = box("<button>a</button><button>b</button>");
  trap(n);
  n.querySelectorAll("button")[1].focus();
  tab(n);
  assert.equal(document.activeElement.textContent, "a");
});

test("Shift+Tab wraps from first back to last", () => {
  const n = box("<button>a</button><button>b</button>");
  trap(n);
  tab(n, true);
  assert.equal(document.activeElement.textContent, "b");
});

test("focus that escaped the container is pulled back", () => {
  const outside = box("<button>outside</button>");
  const n = box("<button>a</button><button>b</button>");
  trap(n);
  outside.querySelector("button").focus();
  tab(n);
  assert.ok(n.contains(document.activeElement));
});

test("release puts focus back where it came from", () => {
  // Dropping focus to <body> on close silently ends keyboard navigation.
  const opener = box("<button>opener</button>").querySelector("button");
  opener.focus();
  const n = box("<button>a</button>");
  const release = trap(n);
  assert.equal(document.activeElement.textContent, "a");
  release();
  assert.equal(document.activeElement, opener);
});

test("release can decline to restore", () => {
  const opener = box("<button>opener</button>").querySelector("button");
  opener.focus();
  const n = box("<button>a</button>");
  trap(n)({ restore: false });
  assert.notEqual(document.activeElement, opener);
});

test("an empty container traps without throwing", () => {
  const n = box("<p>nothing focusable</p>");
  assert.doesNotThrow(() => trap(n));
  assert.doesNotThrow(() => tab(n));
});

test("roving gives the group one tab stop", () => {
  const n = box("<button>a</button><button>b</button><button>c</button>");
  roving(n);
  const tabindexes = [...n.querySelectorAll("button")].map((b) => b.getAttribute("tabindex"));
  assert.deepEqual(tabindexes, ["0", "-1", "-1"]);
});

test("arrows move within the group and wrap", () => {
  const n = box("<button>a</button><button>b</button><button>c</button>");
  roving(n);
  n.querySelector("button").focus();
  arrow(n, "ArrowRight");
  assert.equal(document.activeElement.textContent, "b");
  arrow(n, "ArrowRight");
  arrow(n, "ArrowRight");
  assert.equal(document.activeElement.textContent, "a", "wraps past the end");
  arrow(n, "ArrowLeft");
  assert.equal(document.activeElement.textContent, "c", "wraps before the start");
});

test("Home and End jump to the edges", () => {
  const n = box("<button>a</button><button>b</button><button>c</button>");
  roving(n);
  n.querySelector("button").focus();
  arrow(n, "End");
  assert.equal(document.activeElement.textContent, "c");
  arrow(n, "Home");
  assert.equal(document.activeElement.textContent, "a");
});

test("a vertical group uses up and down", () => {
  const n = box("<button>a</button><button>b</button>");
  roving(n, { orientation: "vertical" });
  n.querySelector("button").focus();
  arrow(n, "ArrowDown");
  assert.equal(document.activeElement.textContent, "b");
  arrow(n, "ArrowRight");
  assert.equal(document.activeElement.textContent, "b", "the cross axis is ignored");
});

test("arrow keys outside the group are left alone", () => {
  const n = box("<button>a</button><button>b</button>");
  roving(n);
  document.body.focus();
  assert.doesNotThrow(() => arrow(n, "ArrowRight"));
});
