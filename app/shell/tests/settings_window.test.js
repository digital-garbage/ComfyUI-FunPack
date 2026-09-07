// The unified Settings window: one modal, a searchable nav list, and every
// section mounted in place beside it -- picking one never leaves the window.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom, fire } from "../../composer/tests/_dom.js";

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

test("a gitStatus rejection degrades to unknown rather than throwing or hanging on ellipsis", async () => {
  const w = createSettingsWindow({ sections: [], gitStatus: () => Promise.reject(new Error("offline")) });
  w.open("about");
  await new Promise((r) => setTimeout(r, 0));
  assert.match(document.querySelector(".cx-modal").textContent, /unknown/);
  w.close();
});

test("a systemInfo rejection leaves the identity facts showing instead of losing them too", async () => {
  const w = createSettingsWindow({
    sections: [],
    gitStatus: () => Promise.resolve({ ok: true, version: "5.0.0", branch: "v5" }),
    systemInfo: () => Promise.reject(new Error("offline")),
  });
  w.open("about");
  await new Promise((r) => setTimeout(r, 0));
  assert.match(document.querySelector(".cx-modal").textContent, /5\.0\.0/);
  assert.match(document.querySelector(".cx-modal").textContent, /v5/);
  w.close();
});

test("system facts arrive independently of git status -- one being slow does not hold the other back", async () => {
  let resolveGit;
  const w = createSettingsWindow({
    sections: [],
    gitStatus: () => new Promise((r) => { resolveGit = r; }),
    systemInfo: () => Promise.resolve({ cpu: { threads: 8 }, python: "3.12.0", gpus: [] }),
  });
  w.open("about");
  await new Promise((r) => setTimeout(r, 0));
  assert.match(document.querySelector(".cx-modal").textContent, /3\.12\.0/, "system facts waited on git");
  resolveGit({ ok: true, version: "5.0.0", branch: "v5" });
  w.close();
});

test("a click outside the window does not close it -- a half-finished edit in any section is not a click away from gone", () => {
  // The bug this guards: pipeline_window.js's own standalone modal has
  // ALWAYS refused to close on an outside click (a group edit's draft is
  // the only copy of it), but Settings' own modal was built without
  // carrying that over, so hosting Models and pipeline in here silently
  // lost the protection its own file was written to guarantee.
  const alpha = fakeSection("a", "Alpha");
  const w = createSettingsWindow({ sections: [alpha] });
  w.open("a");
  fire(document.body, "pointerdown");
  assert.ok(document.querySelector(".cx-modal"), "an outside click closed the window");
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
