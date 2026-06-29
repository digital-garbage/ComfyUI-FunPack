// Regression tests for the global-prompt scene→root matching (scene_align.js).
// Run: node --test movie_editor/frontend/scene_align.test.js
//
// These guard the "stale scene anchor after reorder/insert/delete" bug: a scene's render +
// anchor live under its root id, so the matcher must bind a parsed slot to the root that
// owns its CONTENT, never to whatever root sits at the same array index.
const test = require("node:test");
const assert = require("node:assert");
const { matchParsedToRoots } = require("./scene_align.js");

// Helper: resolve each parsed slot to the root's anchor label it would inherit ("new" = -1).
function anchors(roots, parsedTexts) {
  const rootTexts = roots.map((r) => r.text);
  const idx = matchParsedToRoots(parsedTexts, rootTexts);
  return idx.map((ri) => (ri >= 0 ? roots[ri].anchor : "new"));
}

const A = { text: "a cat", anchor: "anchorA" };
const B = { text: "a dog", anchor: "anchorB" };
const C = { text: "a bird", anchor: "anchorC" };

test("reorder only — anchors follow content, not slot", () => {
  assert.deepStrictEqual(anchors([A, B], ["a dog", "a cat"]), ["anchorB", "anchorA"]);
});

test("edit in place — render/anchor preserved", () => {
  assert.deepStrictEqual(anchors([A], ["a black cat"]), ["anchorA"]);
});

test("insert at front — existing anchors stay, new slot is fresh", () => {
  assert.deepStrictEqual(
    anchors([A, B], ["new scene", "a cat", "a dog"]),
    ["new", "anchorA", "anchorB"],
  );
});

test("delete middle — survivors keep anchors", () => {
  assert.deepStrictEqual(anchors([A, B, C], ["a cat", "a bird"]), ["anchorA", "anchorC"]);
});

test("reorder + edit one — unchanged text pins it, edited one follows", () => {
  assert.deepStrictEqual(anchors([A, B], ["a dog", "a big cat"]), ["anchorB", "anchorA"]);
});

test("edit + insert + keep — leftover edit takes leftover root, new is fresh", () => {
  assert.deepStrictEqual(
    anchors([A, B], ["a cat v2", "NEW", "a dog"]),
    ["anchorA", "new", "anchorB"],
  );
});

test("irreducible (both reordered AND fully rewritten) — positional fallback", () => {
  // No unchanged text to anchor on → genuinely ambiguous; positional guess is acceptable.
  assert.deepStrictEqual(anchors([A, B], ["a big dog", "a cute cat"]), ["anchorA", "anchorB"]);
});

test("duplicate texts — nearest position breaks the tie", () => {
  const X0 = { text: "x", anchor: "x0" };
  const X2 = { text: "x", anchor: "x2" };
  // roots at index 0 and 2; parsed at index 0 and 3 → 0↔x0, 3↔x2.
  assert.deepStrictEqual(
    anchors([X0, { text: "y", anchor: "y1" }, X2], ["x", "y", "z", "x"]),
    ["x0", "y1", "new", "x2"],
  );
});

test("empty parsed text is positional only (never exact-matches a root)", () => {
  // Blank slot must not steal a root by 'exact text'; it takes a leftover root positionally.
  assert.deepStrictEqual(anchors([A, B], ["", "a cat"]), ["anchorB", "anchorA"]);
});

test("no old roots — every slot is fresh", () => {
  assert.deepStrictEqual(anchors([], ["a", "b"]), ["new", "new"]);
});

test("each root used at most once", () => {
  const idx = matchParsedToRoots(["a cat", "a cat"], ["a cat"]);
  assert.deepStrictEqual(idx, [0, -1]); // second duplicate slot finds no root left
});
