import test from "node:test";
import assert from "node:assert/strict";
import { createSelection } from "../selection.js";

function fake(ids) {
  let focus = ids[0];
  return { scenes: ids.map((id) => ({ id })), get selectedId() { return focus; }, select(id) { focus = id; } };
}
const order = ["a", "b", "c", "d"];

test("a click picks one; cmd toggles; shift extends from where you started", () => {
  const sel = createSelection({ project: fake(order) });
  sel.pick("b", {}, order);
  assert.deepEqual(sel.ids, ["b"]);
  sel.pick("d", { additive: true }, order);
  assert.deepEqual(sel.ids, ["b", "d"]);
  sel.pick("d", { additive: true }, order);
  assert.deepEqual(sel.ids, ["b"]);
  sel.pick("a", { range: true }, order);
  assert.deepEqual(sel.ids, ["a", "b"]);
});

test("a clip that is gone drops out of the selection, and nothing selected falls back to the focus", () => {
  const project = fake(order);
  const sel = createSelection({ project });
  sel.pick("c", {}, order);
  project.scenes = project.scenes.filter((s) => s.id !== "c");
  sel.heal();
  assert.deepEqual(sel.ids, [project.selectedId]);
});
