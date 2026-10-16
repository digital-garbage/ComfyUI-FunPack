import test from "node:test";
import assert from "node:assert/strict";
import { FORGET, nameOf, tasteOf } from "./choices.js";

test("a saved rating reads as words; no rating or a forgotten one reads as nothing", () => {
  assert.equal(nameOf("10"), "Liked");
  assert.equal(nameOf("Disliked: bad image"), "Disliked — bad image");
  assert.equal(nameOf("7|loved"), "7/10");
  assert.equal(nameOf(""), "");
  assert.equal(nameOf(FORGET), "");
});

test("what a rating teaches: liked or disliked, which way a dislike went, forget clears, unknown words teach nothing", () => {
  assert.deepEqual(tasteOf("10"), { rating: "liked", axis: null });
  assert.deepEqual(tasteOf("1"), { rating: "disliked", axis: null });
  assert.deepEqual(tasteOf("Disliked: bad composition"), { rating: "disliked", axis: "composition" });
  assert.equal(tasteOf(FORGET), "clear");
  assert.equal(tasteOf(""), "clear");
  assert.equal(tasteOf("Perfect"), null);
});
