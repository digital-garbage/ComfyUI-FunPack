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
  assert.equal(text, "a fox in the rain, dolly in");
  assert.deepEqual(added.map((s) => s.name), ["dolly"]);
  assert.equal(compose(pool, {}, "", { random: () => 0 }).added.length, 2, "weather + camera; the empty style is skipped");
});

test("compose favours what was liked and never draws what was rated down, alone or beside the idea", () => {
  assert.equal(compose(pool, { scores: { fog: 5 } }, "", { random: () => 0.5, add: 1 }).added[0].name, "fog", "11 against 1, either order");
  assert.equal(compose(pool, { scores: { rain: -1 } }, "", { random: () => 0, add: 1 }).added[0].name, "fog");
  assert.deepEqual(compose(pool, { rated_pairs: [["dolly", "rain", -1]] }, "rain", { random: () => 0 }).added.map((s) => s.name), ["pan"]);
  assert.deepEqual(compose(pool, { scores: { dolly: -1, pan: -2 } }, "rain").added, [], "nothing left to add");
});

test("compose: only ratings weigh, never how often a shortcut is typed; uncategorised shortcuts are each their own", () => {
  const two = [{ name: "liked", triggers: ["liked"], replacements: ["l"] }, { name: "habit", triggers: ["habit"], replacements: ["h"] }];
  const one = (r) => compose(two.map((s) => ({ ...s, category: "same" })), { scores: { liked: 1 }, counts: { habit: 500 } }, "", { add: 1, random: () => r }).added[0].name;
  assert.equal(one(0.7), "liked", "3 against 1: typing a shortcut 500 times teaches nothing");
  assert.equal(compose(two, {}, "").added.length, 2);
});

test("a trigger is matched as the expander matches it", () => {
  const lib = [{ name: "fox", triggers: ["fox"] }, { name: "gh", triggers: ["golden hour"] }];
  assert.deepEqual(presentIn(lib, "a fox-like fox's walk").map((s) => s.name), []);
  assert.deepEqual(presentIn(lib, "GOLDEN\n  hour").map((s) => s.name), ["gh"]);
  assert.deepEqual(presentIn(lib, "(fox)").map((s) => s.name), ["fox"]);
});

test("compose never builds what was rated down: not by two triggers side by side, not as a pair it makes itself, not via a shared trigger", () => {
  const lib = [{ name: "G", triggers: ["golden"], replacements: ["g"], category: "a" }, { name: "H", triggers: ["hour"], replacements: ["h"], category: "b" },
    { name: "GH", triggers: ["golden hour"], replacements: ["gh"], category: "c" }];
  const { text } = compose(lib, { scores: { GH: -3 } }, "", { random: () => 0, add: 2 });
  assert.doesNotMatch(text, /golden\s+hour/i, "joined with a comma, so the rated-down one cannot fire");
  const pq = [{ name: "P", triggers: ["p"], replacements: ["p"], category: "x" }, { name: "Q", triggers: ["q"], replacements: ["q"], category: "y" }];
  assert.equal(compose(pq, { rated_pairs: [["P", "Q", -4]] }, "").added.length, 1, "the second would make the disliked pair");
  const twins = [{ name: "bad", triggers: ["x"], replacements: ["b"], category: "u" }, { name: "good", triggers: ["x"], replacements: ["g"], category: "v" }];
  assert.deepEqual(compose(twins, { scores: { bad: -1 } }, "").added, [], "its trigger would fire the other one");
  assert.deepEqual(compose([{ name: "cut", triggers: ["cut"], replacements: [""], category: "w" }], {}, "").added, [], "removes itself: adds nothing");
});
