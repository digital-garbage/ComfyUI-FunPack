const test = require("node:test");
const assert = require("node:assert");
const { pending } = require("./reactive_focus.js");

const shot = (key, extra) => ({ shot: 1, key, already: false, auto_lemma: "a", candidates: [{ lemma: "a", text: "a" }], ...extra });

test("asks only about shots not decided yet, and skips shots with a camera move already", () => {
  const scenes = [{ index: 0, preview: "p", shots: [shot("k1"), shot("k2"), shot("k3", { already: true }), shot("k4", { candidates: [] })] }];
  const out = pending(scenes, { k1: { mode: "auto" } });
  assert.deepStrictEqual(out[0].shots.map((s) => s.key), ["k2"]);
  assert.deepStrictEqual(pending(scenes, { k1: 1, k2: 1 }), []);
});

test("views are asked about by their own list, and a decided view is not asked again", () => {
  const v = (key, extra) => ({ shot: 2, key, already: false, auto: "Side view", candidates: [{ view: "Side view" }], ...extra });
  const scenes = [{ index: 0, preview: "p", views: [v("a"), v("b"), v("c", { already: true })] }];
  const out = pending(scenes, { a: { mode: "auto" } }, "views", (s) => !s.already && s.candidates.length);
  assert.deepStrictEqual(out[0].views.map((s) => s.key), ["b"]);
});
