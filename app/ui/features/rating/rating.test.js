import test from "node:test";
import assert from "node:assert/strict";
import { notTaught } from "./index.js";

const ps = (key, learning = true) => { const taste = { id: "taste" }; return { modulesById: () => ({ taste }), useful: () => learning, currentValues: () => ({ taste: { key } }) }; };

test("the not-taught reason blames a missing key only when the run captured nothing, a learner is on, and no key is set", () => {
  const nothing = { why: "this run captured nothing", reason: "nothing" }, forgotten = { why: "a new Generate started", reason: "forgotten" };
  assert.match(notTaught(nothing, ps("")), /no Taste key is set/);
  assert.match(notTaught(forgotten, ps("")), /new Generate started/, "the key cleared after the run is not why");
  assert.match(notTaught(nothing, ps("", false)), /captured nothing/, "nothing learns anyway: a key would not have helped");
  assert.match(notTaught(nothing, ps("portraits")), /captured nothing/);
  assert.equal(notTaught({ recorded: ["vel"], why: null }, ps("")), "");
});
