import test from "node:test";
import assert from "node:assert/strict";
import { elapsed } from "./progress.js";

test("elapsed reads as minutes and seconds", () => {
  assert.equal(elapsed(0), "00m 00s");
  assert.equal(elapsed(72_900), "01m 12s");
  assert.equal(elapsed(-5), "00m 00s");
});
