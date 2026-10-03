import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom } from "./_dom.js";
import { AUTO, MAX_COLS, applyDensity, cellExpression, cellWidth, normaliseCols, recallCols } from "../internals/density.js";

test.beforeEach(() => setupDom());
test.after(() => teardownDom());

test("cell width accounts for the gaps between columns", () => {
  // v4 hand-tuned these (92/44/29/21) and they drifted from the gap they were
  // meant to allow for. Derived, they add up.
  for (const cols of [1, 2, 3, 4]) {
    const cell = cellWidth(cols);
    const total = cell * cols + 2.5 * (cols - 1);
    assert.ok(Math.abs(total - 100) < 0.05, `${cols} columns totalled ${total}`);
  }
});

test("one column is the full container", () => {
  assert.equal(cellWidth(1), 100);
});

test("auto falls back to the clamp expression", () => {
  assert.match(cellExpression(null), /clamp\(/);
  assert.match(cellExpression(2), /cqw$/);
});

test("column counts are clamped into range", () => {
  assert.equal(normaliseCols(-3), AUTO);
  assert.equal(normaliseCols(99), MAX_COLS);
  assert.equal(normaliseCols("2"), 2);
  assert.equal(normaliseCols(2.7), 2);
  assert.equal(normaliseCols("nonsense"), AUTO);
  assert.equal(normaliseCols(undefined), AUTO);
});

test("applying density sets both the attribute and the cell variable", () => {
  const host = document.createElement("div");
  assert.equal(applyDensity(host, 3), 3);
  assert.equal(host.dataset.cols, "3");
  assert.match(host.style.getPropertyValue("--cell"), /cqw/);
});

test("recall survives storage being unavailable", () => {
  // Private windows and blocked site data throw on access; a grid must still
  // render rather than taking the page down with it.
  const original = Object.getOwnPropertyDescriptor(window, "localStorage");
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    get() { throw new Error("blocked"); },
  });
  globalThis.localStorage = undefined;
  assert.equal(recallCols("bin", 2), 2);
  if (original) Object.defineProperty(window, "localStorage", original);
});
