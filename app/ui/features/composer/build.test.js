import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let build;
test.before(async () => { setupDom(); ({ build } = await import("./build.js")); });
test.after(() => teardownDom());
const tick = () => new Promise((r) => setTimeout(r, 0));
const LIB = [{ name: "neon", triggers: ["neon"], replacements: ["neon lights"], category: "light" }];

function rig({ expand } = {}) {
  const scenes = [{ id: "a", text: "a fox" }, { id: "b", text: "a cat" }, { id: "c", text: "", gen_unit_id: "b", cut_offset_frames: 8 }, { id: "v", text: "", source: { type: "video" } }];
  const heard = new Set(), writes = [];
  const p = { project: { scenes }, scenes, selectedId: "a", get selected() { return scenes.find((s) => s.id === this.selectedId); },
    anchor: "", postfix: "", postfixEnabled: true, variables: [], flush: async () => {},
    setText: (id, t) => { writes.push([id, t]); scenes.find((s) => s.id === id).text = t; heard.forEach((f) => f()); } };
  const api = { shortcuts: async () => ({ shortcuts: LIB }), suggestionStats: async () => ({ scores: { neon: 2 } }),
    expandPrompt: expand || (async (body) => ({ text: body.text.replace("neon", "neon lights") })) };
  const page = build({ project: p, api, on: (f) => { heard.add(f); return () => heard.delete(f); } }, () => {});
  document.body.replaceChildren(page.node);
  const q = (label) => page.node.querySelector(`[aria-label="${label}"]`);
  const press = (label) => [...page.node.querySelectorAll("button")].find((b) => b.textContent === label).click();
  const select = (id) => { p.selectedId = id; heard.forEach((f) => f()); };
  return { page, q, press, select, writes };
}

test("Build drafts from the selected scene, shows the full prompt, and Use puts the draft in that scene", async () => {
  const { page, q, press, writes } = rig();
  assert.equal(q("Idea").value, "a fox");
  press("Build"); await tick(); await tick();
  assert.equal(q("Draft").value, "a fox, neon");
  assert.match(page.node.textContent, /For scene 1\. Added: neon/);
  assert.match(page.node.textContent, /a fox, neon lights/);
  press("Use in scene");
  assert.deepEqual(writes, [["a", "a fox, neon"]]);
});

test("Use before Build, or with an emptied draft, never wipes the scene", async () => {
  const { q, press, writes } = rig();
  press("Use in scene");
  press("Build"); await tick(); await tick();
  q("Draft").value = "  ";
  press("Use in scene");
  assert.deepEqual(writes, []);
});

test("the idea follows the selection until typed in; the draft goes to the scene it was built for, a cut's to its root", async () => {
  const { q, press, select, writes } = rig();
  select("b");
  assert.equal(q("Idea").value, "a cat");
  q("Idea").value = "my words"; q("Idea").dispatchEvent(new window.Event("input"));
  select("b");
  assert.equal(q("Idea").value, "my words", "typed: kept while the same scene stays selected");
  select("c");                                  // a cut of b: same root, so the typing stays
  assert.equal(q("Idea").value, "my words");
  press("Build"); await tick(); await tick();
  select("a");
  assert.equal(q("Idea").value, "a fox", "another scene starts the idea again");
  press("Use in scene");
  assert.deepEqual(writes, [["b", "my words, neon"]], "the scene it was built for, not the one selected since");
});

test("an imported video clip is refused: it has no prompt", async () => {
  const { page, press, select } = rig();
  select("v");
  press("Build"); await tick();
  assert.match(page.node.textContent, /imported video/);
});

test("a slow full-prompt answer never replaces a newer one", async () => {
  const waits = [];
  const { page, q } = rig({ expand: (body) => new Promise((r) => waits.push(() => r({ text: `X(${body.text})` }))) });
  for (const t of ["first", "second"]) { q("Draft").value = t; q("Draft").dispatchEvent(new window.Event("change")); }
  waits[1](); await tick(); waits[0](); await tick();
  assert.match(page.node.textContent, /X\(second\)/);
  assert.doesNotMatch(page.node.textContent, /X\(first\)/);
});
