import test from "node:test";
import assert from "node:assert/strict";
import { accept, matchTriggers, suggestionsAt } from "./match.js";

const lib = [{ name: "a", triggers: ["golden hour", "goldfish"], replacements: ["warm light"] }, { name: "b", triggers: ["rain"], enabled: false }, { name: "c", triggers: ["old gold"] }];

test("prefix matches come first, a disabled shortcut never shows", () => {
  assert.deepEqual(matchTriggers(lib, "gold").map((e) => e.trigger), ["goldfish", "golden hour", "old gold"]);
  assert.deepEqual(matchTriggers(lib, "rain"), []);
});

test("the longest trailing run of words that matches wins, within the current token", () => {
  const text = "a man, walking in golden ho";
  const r = suggestionsAt(lib, text, text.length);
  assert.equal(text.slice(r.span.start, r.span.end), "golden ho");
  assert.equal(r.items[0].trigger, "golden hour");
  assert.equal(suggestionsAt(lib, "x", 1), null);                 // under two letters: nothing
  assert.equal(suggestionsAt(lib, "(zz", 3), null);
});

test("accepting replaces only the span and leaves the caret ready for the next word", () => {
  const out = accept("a golden ho", { start: 2, end: 11 }, "golden hour");
  assert.deepEqual(out, { text: "a golden hour ", caret: 14 });
  assert.equal(accept("a gold, b", { start: 2, end: 6 }, "goldfish").text, "a goldfish, b");      // a separator already follows
});
