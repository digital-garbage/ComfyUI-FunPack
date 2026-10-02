// H3's rating scale sends a bare number instead of a category label; formatLabel/buttonLabel
// must render it sensibly since normalize_refiner_v2_rating (conditioning.py) already accepts
// it verbatim. Run: node --test movie_editor/frontend/rating_picker.test.js
const test = require("node:test");
const assert = require("node:assert");

global.window = {};
require("./rating_picker.js");
const Picker = global.window.MovieRatingPicker;

test("a bare scale number renders as N/10", () => {
  assert.strictEqual(Picker.formatLabel("7"), "7/10");
  assert.strictEqual(Picker.buttonLabel("7"), "★ 7/10");
});

test("a loved scale number keeps the heart", () => {
  assert.strictEqual(Picker.formatLabel("9|loved"), "9/10 ♥");
});

test("the H3 picker's binary endpoints render as Liked/Disliked, not N/10", () => {
  assert.strictEqual(Picker.formatLabel("10"), "Liked");
  assert.strictEqual(Picker.formatLabel("1"), "Disliked");
  assert.strictEqual(Picker.buttonLabel("10"), "★ Liked");
});

test("an existing category label is untouched", () => {
  assert.strictEqual(Picker.formatLabel("Perfect"), "Perfect");
});

test("forget and empty still render as nothing / the prompt", () => {
  assert.strictEqual(Picker.formatLabel(Picker.FORGET_LABEL), "");
  assert.strictEqual(Picker.buttonLabel(""), "Rate scene…");
});

test("what a label teaches the taste key: sign decides, a dislike may name its axis", () => {
  assert.deepStrictEqual(Picker.tasteOf("10"), { rating: "liked", axis: null });
  assert.deepStrictEqual(Picker.tasteOf("1"), { rating: "disliked", axis: null });
  assert.deepStrictEqual(Picker.tasteOf("Disliked: bad image"), { rating: "disliked", axis: "image" });
  assert.deepStrictEqual(Picker.tasteOf("Disliked: bad composition"), { rating: "disliked", axis: "composition" });
  assert.deepStrictEqual(Picker.tasteOf("Perfect|loved"), { rating: "liked", axis: null });
  assert.deepStrictEqual(Picker.tasteOf("Awful"), { rating: "disliked", axis: null });
  assert.strictEqual(Picker.tasteOf("Wrong appearance"), null);
  assert.strictEqual(Picker.tasteOf(""), "clear");
  assert.strictEqual(Picker.tasteOf(Picker.FORGET_LABEL), "clear");
});
