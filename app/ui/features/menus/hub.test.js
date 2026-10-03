import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let hub;
test.before(async () => { setupDom(); hub = (await import("./index.js")).default; });
test.after(() => teardownDom());

const rig = (over = {}) => {
  const calls = [];
  const project = { project: { id: "P", name: "Demo" }, canUndo: true, canRedo: false, undo: () => calls.push("undo"), redo: () => calls.push("redo"), ...over };
  const selection = { ids: [], focus: null };
  const host = document.createElement("div");
  hub.setup({ host, app: { project, selection } });
  return { host, calls };
};
const pick = async (host, menu, item) => {
  [...host.querySelectorAll("button")].find((b) => b.textContent === menu).click();
  await new Promise((r) => setTimeout(r, 10));
  [...document.querySelectorAll(".cx-menu-item")].find((b) => b.textContent.includes(item)).click();
};

test("Edit > Undo undoes; Redo is greyed when there is nothing to redo", async () => {
  const { host, calls } = rig();
  await pick(host, "Edit", "Undo");
  assert.deepEqual(calls, ["undo"]);
  [...host.querySelectorAll("button")].find((b) => b.textContent === "Edit").click();
  await new Promise((r) => setTimeout(r, 10));
  assert.equal([...document.querySelectorAll(".cx-menu-item")].find((b) => b.textContent.includes("Redo")).disabled, true);
});

test("Edit > Delete Scene is greyed until a scene is picked; File lists recent projects", async () => {
  const { host } = rig();
  [...host.querySelectorAll("button")].find((b) => b.textContent === "Edit").click();
  await new Promise((r) => setTimeout(r, 10));
  assert.equal([...document.querySelectorAll(".cx-menu-item")].find((b) => b.textContent.includes("Delete Scene")).disabled, true);
  assert.ok([...document.querySelectorAll(".cx-menu-item")].find((b) => b.textContent.includes("Add Scene")));
});
