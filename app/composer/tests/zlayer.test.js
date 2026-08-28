import test from "node:test";
import assert from "node:assert/strict";

import { LADDER, SLOTS, baseOf, claim, _resetLayers } from "../internals/zlayer.js";

test.beforeEach(() => _resetLayers());

test("the ladder is ordered, and every rung clears the one below", () => {
  for (let i = 1; i < LADDER.length; i += 1) {
    assert.ok(baseOf(LADDER[i]) > baseOf(LADDER[i - 1]) + SLOTS - 1,
      `${LADDER[i]} must clear every peer of ${LADDER[i - 1]}`);
  }
});

test("autocomplete outranks modal", () => {
  // It opens from a field that is usually inside a modal; v4 got this wrong.
  assert.ok(baseOf("autocomplete") > baseOf("modal"));
});

test("an unknown rung is refused rather than guessed", () => {
  assert.throws(() => baseOf("somewhere"), RangeError);
});

test("peers stack within their rung and never reach the next one", () => {
  const a = claim("modal");
  const b = claim("modal");
  assert.ok(b.z > a.z);
  assert.ok(b.z < baseOf("floatingWindow"));
});

test("releasing frees the slot for reuse", () => {
  const a = claim("modal");
  const first = a.z;
  a.release();
  assert.equal(claim("modal").z, first);
});

test("releasing twice is harmless", () => {
  const a = claim("modal");
  a.release();
  a.release();
  assert.equal(a.live, false);
});

test("raise() puts a claim above its peers", () => {
  const a = claim("floatingWindow");
  const b = claim("floatingWindow");
  assert.ok(b.z > a.z);
  a.raise();
  assert.ok(a.z > b.z, "click-to-front");
  assert.ok(a.z < baseOf("toast"));
});

test("raise() on the top claim is a no-op", () => {
  const a = claim("modal");
  const b = claim("modal");
  const before = b.z;
  b.raise();
  assert.equal(b.z, before);
});

test("raise() after release does nothing", () => {
  const a = claim("modal");
  a.release();
  const z = a.z;
  a.raise();
  assert.equal(a.z, z);
});

test("exhausting a rung is loud, not silently wrapped", () => {
  // A rung running dry means overlays are being opened and never released.
  // Reusing a slot would hide the leak and stack two things at one z.
  for (let i = 0; i < SLOTS; i += 1) claim("popover");
  assert.throws(() => claim("popover"), /is full/);
});
