// A knob left at its default is not a setting — it is the absence of one.
//
// Reported from a rental: the H3 gain dials could not be cleared. Once touched, a value was
// written into the project file and stayed there; typing the default back in stored the
// default rather than removing the key, so the value outlived the refinement key it came
// from and looked like learned state that would not clear.
//
// The learned values themselves have always lived only in the refinement key. What leaked
// into the project was the MANUAL side of the same dials, which from the user's chair is the
// same knob.

const test = require("node:test");
const assert = require("node:assert");

// The rule under test, extracted so it can run without the DOM the renderer needs.
function writeSamplerInput(store, k, value, immediate) {
  if (value === k.default) {
    store.unsetSamplerInput(k.name);
    if (immediate) store.flushSave?.();
    return;
  }
  (immediate ? store.setSamplerInputNow : store.setSamplerInput)(k.name, value);
}

function fakeStore() {
  const calls = [];
  return {
    calls,
    unsetSamplerInput: (n) => calls.push(["unset", n]),
    setSamplerInput: (n, v) => calls.push(["set", n, v]),
    setSamplerInputNow: (n, v) => calls.push(["setNow", n, v]),
    flushSave: () => calls.push(["flush"]),
  };
}

const GAIN = { name: "h3_gain_video", default: 1.0 };
const BIAS = { name: "h3_taste_bias", default: 0.0 };

test("returning a dial to its default removes it from the project", () => {
  const s = fakeStore();
  writeSamplerInput(s, GAIN, 1.0, true);
  assert.deepStrictEqual(s.calls[0], ["unset", "h3_gain_video"]);
});

test("a default is never stored as a value", () => {
  const s = fakeStore();
  writeSamplerInput(s, GAIN, 1.0, true);
  assert.ok(!s.calls.some((c) => c[0] === "set" || c[0] === "setNow"));
});

test("a real change is still stored", () => {
  const s = fakeStore();
  writeSamplerInput(s, GAIN, 0.85, true);
  assert.deepStrictEqual(s.calls[0], ["setNow", "h3_gain_video", 0.85]);
});

test("a zero default clears too", () => {
  // h3_taste_bias is centred on 0, not 1 — a rule keyed on truthiness would never clear it.
  const s = fakeStore();
  writeSamplerInput(s, BIAS, 0.0, true);
  assert.deepStrictEqual(s.calls[0], ["unset", "h3_taste_bias"]);
});

test("a non-zero value on a zero-default knob is stored", () => {
  const s = fakeStore();
  writeSamplerInput(s, BIAS, -0.2, true);
  assert.deepStrictEqual(s.calls[0], ["setNow", "h3_taste_bias", -0.2]);
});

test("clearing on commit flushes, so the removal reaches disk", () => {
  const s = fakeStore();
  writeSamplerInput(s, GAIN, 1.0, true);
  assert.ok(s.calls.some((c) => c[0] === "flush"));
});

test("clearing while typing does not flush", () => {
  // Quiet while the caret is in the field, same rule the text knobs already follow.
  const s = fakeStore();
  writeSamplerInput(s, GAIN, 1.0, false);
  assert.ok(!s.calls.some((c) => c[0] === "flush"));
});

test("a store without flushSave does not throw", () => {
  const s = fakeStore();
  delete s.flushSave;
  assert.doesNotThrow(() => writeSamplerInput(s, GAIN, 1.0, true));
});

test("booleans clear when set back to their default", () => {
  const s = fakeStore();
  writeSamplerInput(s, { name: "segmented_detailing", default: false }, false, true);
  assert.deepStrictEqual(s.calls[0], ["unset", "segmented_detailing"]);
});
