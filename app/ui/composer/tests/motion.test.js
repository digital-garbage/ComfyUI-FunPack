import test from "node:test";
import assert from "node:assert/strict";

import { parseDuration } from "../internals/motion.js";

test("ms and s both parse to milliseconds", () => {
  assert.equal(parseDuration("120ms"), 120);
  assert.equal(parseDuration("0.12s"), 120);
  assert.equal(parseDuration("1s"), 1000);
});

test("whitespace from getPropertyValue is tolerated", () => {
  // getComputedStyle returns custom properties with their leading space intact.
  assert.equal(parseDuration(" 180ms "), 180);
});

test("a bare number is treated as milliseconds", () => {
  assert.equal(parseDuration("0"), 0);
  assert.equal(parseDuration("250"), 250);
});

test("missing or unreadable values are zero, never NaN", () => {
  // A NaN duration becomes a setTimeout that fires immediately or never, and
  // the teardown it was guarding is stranded either way.
  for (const bad of ["", null, undefined, "auto", "abc", {}]) {
    assert.equal(parseDuration(bad), 0);
  }
});

test("negative durations clamp to zero", () => {
  assert.equal(parseDuration("-100ms"), 0);
});
