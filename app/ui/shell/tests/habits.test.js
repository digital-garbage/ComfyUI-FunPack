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

test("ratings set the odds: liked comes up more, disliked sits out as often as it was disliked, even with no rival, but is never banned", () => {
  assert.equal(compose(pool, { scores: { fog: 4 } }, "", { random: () => 0.3, add: 1 }).added[0].name, "fog", "4 against 1 each");
  assert.deepEqual(compose(pool, { scores: { dolly: 0.25 } }, "rain", { random: () => 0.5 }).added.map((s) => s.name), ["pan"]);
  assert.deepEqual(compose(pool, { scores: { dolly: 0.25 } }, "rain", { random: () => 0.1 }).added.map((s) => s.name), ["dolly"], "still possible");
  assert.deepEqual(compose(pool, { rated_pairs: [["dolly", "rain", 0.25]] }, "rain", { random: () => 0.5 }).added.map((s) => s.name), ["pan"], "disliked beside the idea");
  const lone = [{ name: "x", triggers: ["x"], replacements: ["x"], category: "c" }];
  assert.deepEqual(compose(lone, { scores: { x: 0.25 } }, "", { random: () => 0.5 }).added, [], "alone in its category, still sits out");
});

test("compose: only ratings weigh, never how often a shortcut is typed; uncategorised shortcuts are each their own", () => {
  const two = [{ name: "liked", triggers: ["liked"], replacements: ["l"] }, { name: "habit", triggers: ["habit"], replacements: ["h"] }];
  const one = (r) => compose(two.map((s) => ({ ...s, category: "same" })), { scores: { liked: 1.5 }, counts: { habit: 500 } }, "", { add: 1, random: () => r }).added[0].name;
  assert.equal(one(0.55), "liked", "typing a shortcut 500 times teaches nothing");
  assert.equal(compose(two, {}, "").added.length, 2);
});

test("a trigger is matched as the expander matches it", () => {
  const lib = [{ name: "fox", triggers: ["fox"] }, { name: "gh", triggers: ["golden hour"] }];
  assert.deepEqual(presentIn(lib, "a fox-like fox's walk").map((s) => s.name), []);
  assert.deepEqual(presentIn(lib, "GOLDEN\n  hour").map((s) => s.name), ["gh"]);
  assert.deepEqual(presentIn(lib, "(fox)").map((s) => s.name), ["fox"]);
});

test("compose never assembles a third shortcut from two side by side, weighs the pairs it makes, never offers a shared trigger", () => {
  const lib = [{ name: "G", triggers: ["golden"], replacements: ["g"], category: "a" }, { name: "H", triggers: ["hour"], replacements: ["h"], category: "b" },
    { name: "GH", triggers: ["golden hour"], replacements: ["gh"], category: "c" }];
  const { text } = compose(lib, { scores: { GH: 1e-12 } }, "", { random: () => 0, add: 2 });
  assert.doesNotMatch(text, /golden\s+hour/i, "joined with a comma, so the disliked one cannot fire");
  const pq = [{ name: "P", triggers: ["p"], replacements: ["p"], category: "x" }, { name: "Q", triggers: ["q"], replacements: ["q"], category: "y" }];
  assert.equal(compose(pq, { rated_pairs: [["P", "Q", 0.25]] }, "", { random: () => 0.5 }).added.length, 1, "the second would make the disliked pair");
  const twins = [{ name: "bad", triggers: ["x"], replacements: ["b"], category: "u" }, { name: "good", triggers: ["x"], replacements: ["g"], category: "v" }];
  assert.deepEqual(compose(twins, {}, "").added, [], "its trigger could fire the other one");
  assert.deepEqual(compose([{ name: "cut", triggers: ["cut"], replacements: [""], category: "w" }], {}, "").added, [], "removes itself: adds nothing");
});

test("the like-guess picks likelier drafts more often, never always, and only once it is ready", async () => {
  const { likelier } = await import("../habits.js");
  const drafts = [{ text: "a fox, fog" }, { text: "a fox, Neon" }];
  const guess = { ready: true, weights: { neon: Math.log(3), fog: 0 } };
  let seed = 7; const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const picks = Array.from({ length: 4000 }, () => likelier(drafts, guess, random).text);
  const neon = picks.filter((t) => t.includes("Neon")).length / picks.length;
  assert.ok(neon > 0.7 && neon < 0.8, `3 to 1 odds: ${neon}`);
  assert.equal(likelier(drafts, { ...guess, ready: false }), drafts[0]);
  assert.equal(likelier(drafts, undefined), drafts[0]);
});
