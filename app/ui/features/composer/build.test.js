import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let build;
test.before(async () => { setupDom(); ({ build } = await import("./build.js")); });
test.after(() => teardownDom());
const tick = () => new Promise((r) => setTimeout(r, 0));

test("Build drafts from the idea, shows the full prompt, and Use puts the draft in the selected scene", async () => {
  const scene = { id: "s1", text: "a fox" }, sent = [];
  const p = { selected: scene, selectedId: "s1", anchor: "", postfix: "", postfixEnabled: true, variables: [], setText: (id, t) => { scene.text = t; } };
  const api = {
    shortcuts: async () => ({ shortcuts: [{ name: "neon", triggers: ["neon"], replacements: ["neon lights"], category: "light" }] }),
    suggestionStats: async () => ({ scores: { neon: 2 } }),
    expandPrompt: async (body) => { sent.push(body); return { text: body.text.replace("neon", "neon lights") }; },
  };
  const page = build({ project: p, api }, () => {});
  document.body.append(page.node);
  assert.equal(page.node.querySelector('[aria-label="Idea"]').value, "a fox", "starts from the selected scene");
  [...page.node.querySelectorAll("button")].find((b) => b.textContent === "Build").click();
  await tick(); await tick();
  assert.equal(page.node.querySelector('[aria-label="Draft"]').value, "a fox neon");
  assert.match(page.node.textContent, /Added: neon/);
  assert.match(page.node.textContent, /a fox neon lights/);
  [...page.node.querySelectorAll("button")].find((b) => b.textContent === "Use in selected scene").click();
  assert.equal(scene.text, "a fox neon");
});
