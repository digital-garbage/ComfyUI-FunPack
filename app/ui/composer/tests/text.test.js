import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom } from "./_dom.js";
import { asText, setText } from "../internals/text.js";

test.beforeEach(() => setupDom());
test.after(() => teardownDom());

test("scalars become text nodes", () => {
  assert.equal(asText("hi").nodeType, 3);
  assert.equal(asText(42).textContent, "42");
  assert.equal(asText(true).textContent, "true");
  assert.equal(asText(0).textContent, "0");
  assert.equal(asText("").textContent, "");
});

test("null and undefined render as nothing, not as the word", () => {
  assert.equal(asText(null).textContent, "");
  assert.equal(asText(undefined).textContent, "");
});

test("markup arrives as literal text, never as elements", () => {
  const host = document.createElement("div");
  host.appendChild(asText('<img src=x onerror="boom()">'));
  assert.equal(host.querySelector("img"), null);
  assert.equal(host.textContent, '<img src=x onerror="boom()">');
});

test("a DOM node is refused, and the message says why", () => {
  const node = document.createElement("span");
  assert.throws(() => asText(node), (err) => {
    assert.ok(err instanceof TypeError);
    assert.match(err.message, /DOM node/);
    return true;
  });
});

test("objects, arrays and functions are refused", () => {
  // Refusing beats escaping: there is no path where markup exists and someone
  // forgot to escape it.
  for (const bad of [{}, [], () => {}, Symbol("x")]) {
    assert.throws(() => asText(bad), TypeError);
  }
});

test("non-finite numbers are refused rather than printed", () => {
  for (const bad of [NaN, Infinity, -Infinity]) assert.throws(() => asText(bad), TypeError);
});

test("setText replaces content instead of appending", () => {
  const host = document.createElement("div");
  setText(host, "one");
  setText(host, "two");
  assert.equal(host.textContent, "two");
  assert.equal(host.childNodes.length, 1);
});
