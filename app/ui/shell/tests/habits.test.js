import test from "node:test";
import assert from "node:assert/strict";
import { analyze, byHabit, compose, presentIn } from "../habits.js";

const lib = [
  { name: "rain", triggers: ["rain"], category: "weather" }, { name: "fog", triggers: ["fog"], category: "weather" },
  { name: "dolly", triggers: ["dolly in"], category: "camera" }, { name: "off", triggers: ["zoom"], category: "camera", enabled: false }, { name: "x", triggers: [], category: "none" },
];

test("a trigger counts only as a whole token", () => {
  assert.deepEqual(presentIn(lib, "a train in the rain").map((s) => s.name), ["rain"]);
  assert.deepEqual(presentIn(lib, "constraint").map((s) => s.name), []);
  assert.deepEqual(presentIn(lib, "a DOLLY IN shot").map((s) => s.name), ["dolly"]);
});

test("a category is used when any of its triggers is in the text; disabled and trigger-less shortcuts are ignored", () => {
  const { used, missing } = analyze(lib, "rain");
  assert.deepEqual(used.map((g) => g.cat), ["weather"]);
  assert.deepEqual(missing.map((g) => g.cat), ["camera"]);
});

test("habits: partners of what is here, strongest first, one-offs ignored, nothing already used", () => {
  const edges = [["rain", "dolly", 3], ["rain", "fog", 1], ["fog", "dolly", 2]];
  const got = byHabit(lib, edges, new Set(["rain"]), new Set(["rain"]), { both: true });
  assert.deepEqual(got.map((g) => [g.sc.name, g.n, g.with.name]), [["dolly", 3, "rain"]]);
});

const pool = [
  { name: "rain", triggers: ["rain"], replacements: ["r"], category: "weather" }, { name: "fog", triggers: ["fog"], replacements: ["f"], category: "weather" },
  { name: "dolly", triggers: ["dolly in"], replacements: ["d"], category: "camera" }, { name: "pan", triggers: ["pan"], replacements: ["p"], category: "camera" },
  { name: "empty", triggers: ["nil"], replacements: [], category: "style" },
];

test("compose keeps the idea, adds one shortcut per unused category, never one with nothing to put in", () => {
  const { text, added } = compose(pool, {}, "a fox in the rain", { random: () => 0 });
  assert.equal(text, "a fox in the rain dolly in");
  assert.deepEqual(added.map((s) => s.name), ["dolly"]);
  assert.equal(compose(pool, {}, "", { random: () => 0 }).added.length, 2, "weather + camera; the empty style is skipped");
});

test("compose favours what was liked and never draws what was rated down, alone or beside the idea", () => {
  assert.equal(compose(pool, { scores: { fog: 5 } }, "", { random: () => 0.5, add: 1 }).added[0].name, "fog", "11 against 1, either order");
  assert.equal(compose(pool, { scores: { rain: -1 } }, "", { random: () => 0, add: 1 }).added[0].name, "fog");
  assert.deepEqual(compose(pool, { rated_pairs: [["dolly", "rain", -1]] }, "rain", { random: () => 0 }).added.map((s) => s.name), ["pan"]);
  assert.deepEqual(compose(pool, { scores: { dolly: -1, pan: -2 } }, "rain").added, [], "nothing left to add");
});
