// The unified Settings window: one modal, a searchable list of sections, and
// two kinds of section -- mounted in place (About), or a deep link that
// closes this window and opens the real one.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let createSettingsWindow;
test.before(async () => {
  setupDom();
  await import("../../composer/composer.js");
  ({ createSettingsWindow } = await import("../settings_window.js"));
});
test.after(() => teardownDom());

function fakeSection(id, title) {
  const destroyed = [];
  return {
    id, title, subtitle: `${title} subtitle`, keywords: title,
    calls: destroyed,
    mount() {
      const stack = { node: document.createElement("div"), destroy: () => destroyed.push("destroyed") };
      stack.node.textContent = `${title} content`;
      return stack;
    },
  };
}

test("opening draws the first section by default", () => {
  const w = createSettingsWindow({ sections: [fakeSection("a", "Alpha"), fakeSection("b", "Beta")] });
  w.open();
  const modal = document.querySelector(".cx-modal");
  assert.ok(modal, "no modal was mounted");
  assert.match(modal.textContent, /About FunPack/);
  w.close();
});

test("picking a section in the nav swaps the content and tears down the last one", () => {
  const alpha = fakeSection("a", "Alpha");
  const w = createSettingsWindow({ sections: [alpha] });
  w.open("a");
  assert.match(document.querySelector(".cx-modal").textContent, /Alpha content/);

  document.querySelector('.cx-filter-row[aria-selected="false"]').click();  // About
  assert.match(document.querySelector(".cx-modal").textContent, /About FunPack/);
  assert.deepEqual(alpha.calls, ["destroyed"]);
  w.close();
});

test("a deep-link section closes this window and calls its own action instead of drawing anything", () => {
  let called = 0;
  const w = createSettingsWindow({
    sections: [{ id: "packs", title: "Node packs", keywords: "packs", action: () => { called += 1; } }],
  });
  w.open("packs");
  assert.equal(called, 1);
  assert.equal(document.querySelector(".cx-modal"), null, "the settings window is still open over the real one");
});

test("opening again while already open jumps to the requested section instead of stacking a second modal", () => {
  const w = createSettingsWindow({ sections: [fakeSection("a", "Alpha"), fakeSection("b", "Beta")] });
  w.open("a");
  w.open("b");
  assert.equal(document.querySelectorAll(".cx-modal").length, 1);
  assert.match(document.querySelector(".cx-modal").textContent, /Beta content/);
  w.close();
});

test("the search box narrows the list to matching sections", () => {
  const w = createSettingsWindow({ sections: [fakeSection("a", "Alpha"), fakeSection("b", "Beta")] });
  w.open();
  const search = document.querySelector(".cx-search");
  search.value = "Beta";
  search.dispatchEvent(new window.Event("input", { bubbles: true }));
  const labels = [...document.querySelectorAll(".cx-filter-label")].map((n) => n.textContent);
  assert.deepEqual(labels, ["Beta"]);
  w.close();
});

test("a gitStatus rejection is shown as a hint rather than left blank or thrown", async () => {
  const w = createSettingsWindow({ sections: [], gitStatus: () => Promise.reject(new Error("offline")) });
  w.open("about");
  await new Promise((r) => setTimeout(r, 0));
  assert.match(document.querySelector(".cx-modal").textContent, /Could not read version information/);
  w.close();
});

test("closing tears down the active section and a later open starts fresh", () => {
  const alpha = fakeSection("a", "Alpha");
  const w = createSettingsWindow({ sections: [alpha] });
  w.open("a");
  w.close();
  assert.equal(document.querySelector(".cx-modal"), null);
  assert.deepEqual(alpha.calls, ["destroyed"]);
});
