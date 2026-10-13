import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let shortcuts;
test.before(async () => { setupDom(); shortcuts = (await import("./shortcuts.js")).shortcuts; });
test.after(() => teardownDom());

const one = (name, extra = {}) => ({ name, triggers: [name.toLowerCase()], replacements: ["x"], enabled: true, category: "", sub_category: "", ...extra });
const wait = () => new Promise((r) => setTimeout(r, 10));
const mount = async (api) => { const page = shortcuts({ api }, () => {}); document.body.append(page.node); await wait(); return page; };

test("shortcuts are in real alphabetical order: case ignored, and a paused one is placed by its name, not by its pause mark", async () => {
  const lib = { shortcuts: [one("banana"), one("Apple"), one("cherry", { enabled: false }), one("apple2"), one("Zed")] };
  const page = await mount({ shortcuts: async () => lib });
  assert.deepEqual([...page.node.querySelectorAll(".cx-filter-label")].map((n) => n.textContent), ["Apple", "apple2", "banana", "⏸ cherry", "Zed"]);
  page.node.remove();
});

test("what was typed in the search box is still there after a shortcut is saved", async () => {
  const lib = { shortcuts: [one("banana"), one("Apple"), one("bandana")] };
  const page = await mount({ shortcuts: async () => lib, saveShortcut: async () => lib });
  const search = () => page.node.querySelector(".cx-search");
  search().value = "ban";
  search().dispatchEvent(new window.Event("input", { bubbles: true }));
  assert.deepEqual([...page.node.querySelectorAll(".cx-filter-label")].map((n) => n.textContent), ["banana", "bandana"]);
  page.node.querySelector(".cx-filter-row").click();                                     // opens the editor
  [...document.querySelectorAll("button")].find((b) => b.textContent === "Save").click();
  await wait();
  assert.equal(search().value, "ban", "the search box was rebuilt but kept its text");
  assert.equal(page.node.querySelectorAll(".cx-filter-label").length, 2, "and the list is still filtered");
  page.node.remove();
});
