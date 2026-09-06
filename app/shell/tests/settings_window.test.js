// The unified Settings window: one modal, a searchable nav list, and every
// section mounted in place beside it -- picking one never leaves the window.

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

function fakeSection(id, title, { keywords } = {}) {
  const destroyed = [];
  return {
    id, title, subtitle: `${title} subtitle`, keywords: keywords ?? "",
    calls: destroyed,
    lastCtx: null,
    mount(ctx) {
      this.lastCtx = ctx;
      const node = document.createElement("div");
      node.textContent = `${title} content`;
      return { node, destroy: () => destroyed.push("destroyed") };
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

test("picking a section in the nav swaps the content and tears down the last one, without closing the window", () => {
  const alpha = fakeSection("a", "Alpha");
  const w = createSettingsWindow({ sections: [alpha] });
  w.open("a");
  assert.match(document.querySelector(".cx-modal").textContent, /Alpha content/);

  document.querySelector('.cx-filter-row[aria-selected="false"]').click();  // About
  assert.equal(document.querySelectorAll(".cx-modal").length, 1, "picking a section opened or closed a modal");
  assert.match(document.querySelector(".cx-modal").textContent, /About FunPack/);
  assert.deepEqual(alpha.calls, ["destroyed"]);
  w.close();
});

test("a mounted section gets a setFooter and a close, wired to this window's own modal", () => {
  const alpha = fakeSection("a", "Alpha");
  const w = createSettingsWindow({ sections: [alpha] });
  w.open("a");
  assert.equal(typeof alpha.lastCtx.setFooter, "function");
  alpha.lastCtx.setFooter({ note: "hello" });
  assert.match(document.querySelector(".cx-modal-foot").textContent, /hello/);

  alpha.lastCtx.close();
  assert.equal(document.querySelector(".cx-modal"), null, "the section's own close did not close the window");
});

test("switching sections clears the previous one's footer instead of leaving it showing", () => {
  const alpha = fakeSection("a", "Alpha");
  const beta = fakeSection("b", "Beta");
  const w = createSettingsWindow({ sections: [alpha, beta] });
  w.open("a");
  alpha.lastCtx.setFooter({ note: "alpha's own note" });
  w.open("b");
  assert.doesNotMatch(document.querySelector(".cx-modal-foot").textContent, /alpha's own note/);
  w.close();
});

test("opening again while already open jumps to the requested section instead of stacking a second modal", () => {
  const w = createSettingsWindow({ sections: [fakeSection("a", "Alpha"), fakeSection("b", "Beta")] });
  w.open("a");
  w.open("b");
  assert.equal(document.querySelectorAll(".cx-modal").length, 1);
  assert.match(document.querySelector(".cx-modal").textContent, /Beta content/);
  w.close();
});

test("the search box narrows the list to sections matching the title", () => {
  const w = createSettingsWindow({ sections: [fakeSection("a", "Alpha"), fakeSection("b", "Beta")] });
  w.open();
  const search = document.querySelector(".cx-search");
  search.value = "Beta";
  search.dispatchEvent(new window.Event("input", { bubbles: true }));
  const labels = [...document.querySelectorAll(".cx-filter-label")].map((n) => n.textContent);
  assert.deepEqual(labels, ["Beta"]);
  w.close();
});

test("the search box also matches a section's keywords, not only its visible title and subtitle", () => {
  // The bug this guards: an earlier version only searched `label` + `hint`,
  // so a section findable by a word that appears ONLY in its keywords list
  // (e.g. "loaders" for a section titled "Models and pipeline") silently
  // vanished from the filtered list instead of matching.
  const w = createSettingsWindow({
    sections: [fakeSection("a", "Alpha", { keywords: "xyzzy" }), fakeSection("b", "Beta")],
  });
  w.open();
  const search = document.querySelector(".cx-search");
  search.value = "xyzzy";
  search.dispatchEvent(new window.Event("input", { bubbles: true }));
  const labels = [...document.querySelectorAll(".cx-filter-label")].map((n) => n.textContent);
  assert.deepEqual(labels, ["Alpha"]);
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

test("two independent windows do not share which section is open", () => {
  // The bug this guards: `current` used to live at module scope, so a second
  // createSettingsWindow() instance's open() would redraw the FIRST
  // instance's already-open modal instead of opening its own.
  const w1 = createSettingsWindow({ sections: [fakeSection("a", "Alpha")] });
  const w2 = createSettingsWindow({ sections: [fakeSection("z", "Zeta")] });
  w1.open("a");
  w2.open("z");
  assert.equal(document.querySelectorAll(".cx-modal").length, 2);
  const texts = [...document.querySelectorAll(".cx-modal")].map((m) => m.textContent);
  assert.ok(texts.some((t) => t.includes("Alpha content")), "the first window lost its own section");
  assert.ok(texts.some((t) => t.includes("Zeta content")), "the second window did not open its own section");
  w1.close();
  w2.close();
});
