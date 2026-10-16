import test from "node:test";
import assert from "node:assert/strict";
import { createKeys } from "../keys.js";

const fakeDoc = (dialog = false) => { const d = { on: null, addEventListener(_, fn) { d.on = fn; }, querySelector: () => dialog }; return d; };
const press = (d, key, extra = {}, target = {}) => { let stopped = false; d.on({ key, target, preventDefault: () => { stopped = true; }, ...extra }); return stopped; };

test("a bound combo runs and is swallowed; false lets the key through", () => {
  const d = fakeDoc(), k = createKeys(d);
  let n = 0;
  k.bind("mod+z", () => { n++; });
  k.bind("delete", () => false);
  assert.equal(press(d, "z", { metaKey: true }), true);
  assert.equal(n, 1);
  assert.equal(press(d, "Delete"), false);
  assert.equal(press(d, "q"), false);
});

test("shift is part of the combo, typing in a field is never intercepted, unbind works", () => {
  const d = fakeDoc(), k = createKeys(d);
  let n = 0;
  const off = k.bind("shift+mod+z", () => { n++; });
  press(d, "Z", { metaKey: true, shiftKey: true });
  press(d, "Z", { metaKey: true, shiftKey: true }, { tagName: "TEXTAREA" });
  assert.equal(n, 1);
  off();
  press(d, "Z", { metaKey: true, shiftKey: true });
  assert.equal(n, 1);
});

test("+ works with Shift held, and nothing fires on repeat, with Alt, or behind a dialog", () => {
  const d = fakeDoc(), k = createKeys(d);
  let n = 0;
  k.bind("+", () => { n++; });
  press(d, "+", { shiftKey: true });
  assert.equal(n, 1);
  press(d, "+", { repeat: true });
  press(d, "+", { altKey: true });
  assert.equal(n, 1);
  const behind = fakeDoc(true), k2 = createKeys(behind);
  k2.bind("+", () => { n++; });
  press(behind, "+");
  assert.equal(n, 1);
});
