// A gallery keeps the tiles it has: only a tile whose item changed is rebuilt, and a selection change rebuilds nothing.
import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "./_dom.js";

let composer;
test.before(async () => { setupDom(); ({ composer } = await import("../composer.js")); });
test.after(() => teardownDom());

const item = (id, extra = {}) => ({ id, label: id, thumb: `/t/${id}`, badge: "image", ...extra });
const cells = (g) => [...g.node.querySelectorAll(".cx-cell")];

for (const view of ["adaptive", "icons", "list"]) {
  test(`${view}: setting the same items again keeps every tile, and a selection change rebuilds none`, () => {
    const g = composer.gallery[view]({ items: [item("a"), item("b"), item("c")] });
    document.body.append(g.node);
    const first = [...g.node.querySelectorAll("button")];
    g.setItems([item("a"), item("b"), item("c")]);
    g.setValue(["b"]);
    const again = [...g.node.querySelectorAll("button")];
    assert.equal(again.length, 3);
    again.forEach((n, i) => assert.equal(n, first[i], "the very same tile, so its picture is not fetched again"));
    assert.deepEqual(again.map((n) => n.getAttribute("aria-selected")), ["false", "true", "false"]);
    assert.ok(again[1].classList.contains("cx-on") && !again[0].classList.contains("cx-on"));
    g.setValue([]);
    assert.deepEqual([...g.node.querySelectorAll("button")].map((n) => n.getAttribute("aria-selected")), ["false", "false", "false"]);
    g.node.remove();
  });
}

test("a changed item rebuilds only its own tile, and a new order moves tiles instead of making new ones", () => {
  const g = composer.gallery.adaptive({ items: [item("a"), item("b"), item("c")] });
  document.body.append(g.node);
  const [a, b, c] = cells(g);
  g.setItems([item("a"), item("b", { flags: [{ text: "R2", title: "slot 2" }] }), item("c")]);
  const [a2, b2, c2] = cells(g);
  assert.equal(a2, a); assert.equal(c2, c);
  assert.notEqual(b2, b, "the one whose flags changed");
  assert.equal(b2.querySelector(".cx-cell-flag").textContent, "R2");
  g.setItems([item("c"), item("a"), item("b", { flags: [{ text: "R2", title: "slot 2" }] })]);
  assert.deepEqual(cells(g), [c, a, b2], "reordered, not rebuilt");
  g.setItems([item("a")]);
  assert.deepEqual(cells(g), [a], "a tile that is gone is gone");
  g.setItems([]);
  assert.equal(g.node.querySelectorAll(".cx-cell").length, 0);
  assert.ok(g.node.querySelector(".cx-list-empty"));
  g.node.remove();
});

test("a click still reaches the item after tiles were kept across redraws", () => {
  const seen = [];
  const g = composer.gallery.adaptive({ items: [item("a"), item("b")], onActivate: (it) => seen.push(it.id) });
  document.body.append(g.node);
  g.setItems([item("a"), item("b")]);
  g.setValue(["a"]);
  cells(g)[1].click();
  cells(g)[0].click();
  assert.deepEqual(seen, ["b", "a"]);
  g.node.remove();
});
