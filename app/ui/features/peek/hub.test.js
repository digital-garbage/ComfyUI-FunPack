import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let peek;
test.before(async () => { setupDom(); globalThis.localStorage = window.localStorage; peek = (await import("./index.js")).default; });
test.after(() => { delete globalThis.localStorage; teardownDom(); });

test("the button turns the frame's auto-hide on and off, remembers it, and switches it off when removed", () => {
  localStorage.clear();
  const calls = [], app = { peek: { set: (on) => calls.push(on) } };
  const host = document.createElement("div");
  const off = peek.setup({ host, app });
  const button = host.querySelector("button");
  assert.deepEqual(calls, [false]);
  button.click();
  assert.deepEqual(calls, [false, true]);
  assert.equal(localStorage.getItem("funpack_timeline_peek"), "1");
  assert.equal(button.getAttribute("aria-pressed"), "true");
  button.click();
  assert.equal(localStorage.getItem("funpack_timeline_peek"), null);
  off();
  assert.equal(calls.at(-1), false);
  assert.equal(host.children.length, 0);
});
