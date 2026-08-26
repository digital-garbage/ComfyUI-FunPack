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
    (k.clearsOnDefault || []).forEach((name) => store.unsetSamplerInput(name));
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


// ── handing ownership back ──────────────────────────────────────────────────
//
// Reported from a rental as a question, which is the best kind: "save as manual, load it,
// switch to learned — and nothing gets cleared. Is it so?" It was. The six values stayed in
// the project, inert (learned mode reads the key, never the dials) and invisible (the dials
// hide in learned mode), until the next time the project was opened. Correct output, state
// nobody could see. That is the failure this whole rule exists to prevent.

const fs = require("node:fs");
const path = require("node:path");

const MODE = {
  name: "h3_gain_mode",
  default: "learned",
  clearsOnDefault: ["h3_gain_video", "h3_gain_prompt", "h3_gain_audio",
                    "h3_prompt_scale", "h3_taste_bias", "h3_video_detail"],
};

test("switching H3 render gains back to learned clears the manual values with it", () => {
  const s = fakeStore();
  writeSamplerInput(s, MODE, "learned", true);
  const cleared = s.calls.filter((c) => c[0] === "unset").map((c) => c[1]);
  assert.deepStrictEqual(cleared, ["h3_gain_mode", ...MODE.clearsOnDefault]);
});

test("choosing manual clears nothing — those values are about to be used", () => {
  const s = fakeStore();
  writeSamplerInput(s, MODE, "manual", true);
  assert.ok(!s.calls.some((c) => c[0] === "unset"));
  assert.deepStrictEqual(s.calls[0], ["setNow", "h3_gain_mode", "manual"]);
});

test("an ordinary knob returning to default clears only itself", () => {
  const s = fakeStore();
  writeSamplerInput(s, GAIN, 1.0, true);
  assert.deepStrictEqual(s.calls.filter((c) => c[0] === "unset"), [["unset", "h3_gain_video"]]);
});

test("the rule under test is the one the renderer actually runs", () => {
  // This file re-implements writeSamplerInput so it can run without a DOM. That copy is
  // free to drift from the real one, and a green suite would say nothing.
  const src = fs.readFileSync(path.join(__dirname, "engine_settings.js"), "utf8");
  assert.match(src, /\(k\.clearsOnDefault \|\| \[\]\)\.forEach/);
  assert.match(src, /clearsOnDefault: \["h3_gain_video"/);
});

test("the mode clears exactly the values the backend treats as key-scoped", () => {
  // Two halves of one rule in two languages: the browser clears them on the switch, the
  // server clears them on load. A value in one list and not the other is a leak.
  const src = fs.readFileSync(
    path.join(__dirname, "..", "backend", "timeline.py"), "utf8");
  const block = src.split("KEY_SCOPED_SAMPLER_INPUTS = frozenset({")[1].split("})")[0];
  const backend = [...block.matchAll(/"([a-z0-9_]+)"/g)].map((m) => m[1]).sort();
  assert.deepStrictEqual(backend, [...MODE.clearsOnDefault].sort());
});
