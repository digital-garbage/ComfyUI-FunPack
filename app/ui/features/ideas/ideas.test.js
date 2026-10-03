import test from "node:test";
import assert from "node:assert/strict";
import { analyze, byHabit, presentIn } from "./ideas.js";

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
