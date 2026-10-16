import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let templatesBar;
test.before(async () => { setupDom(); ({ templatesBar } = await import("./storytools.js")); });
test.after(() => teardownDom());

test("applying a template swaps the story and leaves the project's variables alone", async () => {
  const project = { id: "p", anchor: "", scenes: [{ id: "a", text: "old" }], variables: [{ name: "mine", value: "kept" }],
    prompt_templates: [{ name: "T", scenes: ["new story"], variables: [{ name: "stale", value: "from an old template" }] }], active_prompt_template: "" };
  const p = { project, edit: (fn) => fn(project), scenes: project.scenes };
  const bar = templatesBar({ project: p, on: () => () => {} }, () => {});
  const select = bar.node.querySelector("select");
  select.value = "T";
  select.dispatchEvent(new window.Event("change", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(project.scenes[0].text, "new story", "the template was applied");
  assert.deepEqual(project.variables, [{ name: "mine", value: "kept" }], "variables are the project's own");
});
