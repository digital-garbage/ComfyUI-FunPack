import test from "node:test";
import assert from "node:assert/strict";

import { place } from "../internals/anchor.js";

const VIEW = { width: 1000, height: 800 };
const mid = { x: 400, y: 400, width: 100, height: 24 };
const float = { width: 200, height: 120 };

const at = (anchor, opts = {}) => place({ anchor, float, viewport: VIEW, ...opts });

test("below and start-aligned by default", () => {
  const r = at(mid);
  assert.equal(r.side, "bottom");
  assert.equal(r.y, mid.y + mid.height + 6);
  assert.equal(r.x, mid.x);
  assert.equal(r.flipped, false);
});

test("gap is respected on every side", () => {
  assert.equal(at(mid, { side: "bottom", gap: 20 }).y, mid.y + mid.height + 20);
  assert.equal(at(mid, { side: "top", gap: 20 }).y, mid.y - float.height - 20);
  assert.equal(at(mid, { side: "right", gap: 20 }).x, mid.x + mid.width + 20);
  assert.equal(at(mid, { side: "left", gap: 20 }).x, mid.x - float.width - 20);
});

test("alignment moves along the cross axis only", () => {
  assert.equal(at(mid, { align: "start" }).x, 400);
  assert.equal(at(mid, { align: "center" }).x, 400 + 50 - 100);
  assert.equal(at(mid, { align: "end" }).x, 400 + 100 - 200);
  // the main axis is untouched by alignment
  const ys = ["start", "center", "end"].map((align) => at(mid, { align }).y);
  assert.equal(new Set(ys).size, 1);
});

// --- flipping ---------------------------------------------------------------

test("flips up when there is no room below", () => {
  const low = { x: 400, y: 760, width: 100, height: 24 };
  const r = at(low, { side: "bottom" });
  assert.equal(r.side, "top");
  assert.equal(r.flipped, true);
});

test("flips left when there is no room right", () => {
  const right = { x: 940, y: 400, width: 40, height: 24 };
  const r = at(right, { side: "right" });
  assert.equal(r.side, "left");
});

test("does not flip into an even tighter gap", () => {
  // Squeezed both ways: 106px below, 20px above, and the float needs 134. It
  // fits neither, so it stays where it was asked to go and clamps -- flipping
  // into LESS room just moves the clipping somewhere the caller did not expect,
  // which is what several of v4's seven copies did.
  const nearTop = { x: 400, y: 20, width: 100, height: 24 };
  const shallow = { width: 1000, height: 150 };
  const r = place({ anchor: nearTop, float, viewport: shallow, side: "bottom" });
  assert.equal(r.side, "bottom");
  assert.equal(r.flipped, false);
  assert.equal(r.clamped, true);
});

test("flip can be switched off", () => {
  const low = { x: 400, y: 760, width: 100, height: 24 };
  assert.equal(at(low, { side: "bottom", flip: false }).side, "bottom");
});

// --- clamping ---------------------------------------------------------------

test("clamps back inside the left edge", () => {
  const r = at({ x: 2, y: 400, width: 40, height: 24 }, { align: "end" });
  assert.ok(r.x >= 8);
  assert.equal(r.clamped, true);
});

test("clamps back inside the right edge", () => {
  const r = at({ x: 960, y: 400, width: 40, height: 24 });
  assert.ok(r.x + float.width <= VIEW.width - 8);
});

test("clamping never pushes the float off the opposite edge", () => {
  // A float wider than the viewport must sit at the padding, not at a negative x.
  const huge = { width: 1200, height: 120 };
  const r = place({ anchor: mid, float: huge, viewport: VIEW });
  assert.equal(r.x, 8);
});

test("clamp can be switched off", () => {
  const r = at({ x: 2, y: 400, width: 40, height: 24 }, { align: "end", clamp: false });
  assert.ok(r.x < 0);
  assert.equal(r.clamped, false);
});

// --- edges ------------------------------------------------------------------

test("an unknown side is refused rather than guessed", () => {
  assert.throws(() => at(mid, { side: "sideways" }), RangeError);
});

test("a zero-size anchor still places sensibly", () => {
  // Context menus anchor to a point, not an element.
  const r = at({ x: 300, y: 300, width: 0, height: 0 });
  assert.equal(r.x, 300);
  assert.equal(r.y, 306);
});

test("results are integers", () => {
  const r = at(mid, { align: "center" });
  assert.equal(Number.isInteger(r.x), true);
  assert.equal(Number.isInteger(r.y), true);
});
