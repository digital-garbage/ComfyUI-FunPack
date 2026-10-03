import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let hub;
test.before(async () => { setupDom(); hub = (await import("./index.js")).default; });
test.after(() => teardownDom());

const rig = (over = {}) => {
  const calls = [];
  const project = { project: { id: "P", name: "Demo" }, canUndo: true, canRedo: false, undo: () => calls.push("undo"), redo: () => calls.push("redo"), ...over };
  const host = document.createElement("div");
  hub.setup({ host, app: { project } });
  return { host, calls };
};
const pick = (host, menu, item) => {
  [...host.querySelectorAll("button")].find((b) => b.textContent === menu).click();
  [...document.querySelectorAll(".cx-menu-item")].find((b) => b.textContent.includes(item)).click();
};

test("Edit > Undo undoes; Redo is greyed when there is nothing to redo", () => {
  const { host, calls } = rig();
  pick(host, "Edit", "Undo");
  assert.deepEqual(calls, ["undo"]);
  [...host.querySelectorAll("button")].find((b) => b.textContent === "Edit").click();
  assert.equal([...document.querySelectorAll(".cx-menu-item")].find((b) => b.textContent.includes("Redo")).disabled, true);
});
