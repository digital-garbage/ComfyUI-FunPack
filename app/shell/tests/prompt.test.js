// The inputs a pipeline puts on the app's own surface.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom, fire } from "../../composer/tests/_dom.js";

let createPrompts, offer, resetMounts;
test.before(async () => {
  setupDom();
  await import("../../composer/composer.js");
  ({ createPrompts } = await import("../prompt.js"));
  ({ offer, _reset: resetMounts } = await import("../mounts.js"));
});
test.after(() => teardownDom());

const SLOTS = [
  { id: "positive", node: "CLIPTextEncode",
    roles: [{ at: "generation.prompt", input: "text", label: "Prompt" }],
    inputs: { clip: ["clip", 0], text: "a cat" } },
  { id: "negative", node: "CLIPTextEncode",
    roles: [{ at: "generation.prompt", input: "text", label: "Negative" }],
    inputs: { clip: ["clip", 0], text: "" } },
  { id: "latent", node: "EmptyLatent", inputs: { width: 512 } },
];

function host(mount = "generation.prompt") {
  resetMounts();
  const node = document.createElement("div");
  document.body.replaceChildren(node);
  offer(mount, node);
  return node;
}

const boxes = (node) => [...node.querySelectorAll("textarea")];

test("a slot that names a place gets a box there", () => {
  const node = host();
  const p = createPrompts(SLOTS);

  assert.equal(boxes(node).length, 2);
  assert.deepEqual([...node.querySelectorAll(".cx-label")].map((l) => l.textContent),
    ["Prompt", "Negative"]);
  assert.equal(boxes(node)[0].value, "a cat", "the box did not show what the slot holds");
  assert.deepEqual(p.fields.map((f) => f.slot), ["positive", "negative"]);
});

test("what is typed comes back addressed by slot and input", () => {
  const node = host();
  const p = createPrompts(SLOTS);

  boxes(node)[0].value = "a dog on a roof";
  fire(boxes(node)[0], "input");

  assert.deepEqual(p.overrides(), {
    positive: { text: "a dog on a roof" },
    negative: { text: "" },
  });
});

test("a numeric input gets a number control, not a prompt box", () => {
  // The same mechanism carries the prompt and the size the project generates
  // at. What kind of control appears is decided by what is IN the input, so a
  // pipeline that puts a number on the surface does not get a textarea for it.
  const node = host("project.video");
  const p = createPrompts([{ id: "latent", node: "EmptyLatent",
    roles: [{ at: "project.video", input: "width", label: "Width" },
            { at: "project.video", input: "length", label: "Length" }],
    inputs: { width: 832, length: 97 } }]);

  assert.equal(node.querySelectorAll("textarea").length, 0);
  const numbers = [...node.querySelectorAll('input[type="number"]')];
  assert.equal(numbers.length, 2);
  assert.equal(numbers[0].value, "832");
  assert.deepEqual(p.overrides(), { latent: { width: 832, length: 97 } });
});

test("a changed control says which input it was", () => {
  // Two places take a value now -- a scene's text and the project's settings --
  // and "something changed, go and read it all" does not say which one.
  const node = host("project.video");
  const told = [];
  createPrompts([{ id: "latent", node: "EmptyLatent",
    roles: [{ at: "project.video", input: "width", label: "Width" }],
    inputs: { width: 512 } }], { onChange: (field) => told.push(field) });

  const box = node.querySelector('input[type="number"]');
  box.value = "832";
  fire(box, "blur");

  assert.equal(told.length, 1);
  assert.equal(told[0].at, "project.video");
  assert.equal(told[0].input, "width");
  assert.equal(told[0].slot, "latent");
  assert.equal(told[0].control.value, 832);
});

test("the controls at a place come back with the input each one writes", () => {
  const node = host("project.video");
  const p = createPrompts([{ id: "latent", node: "EmptyLatent",
    roles: [{ at: "project.video", input: "width", label: "Width" },
            { at: "project.video", input: "height", label: "Height" }],
    inputs: { width: 512, height: 512 } }]);
  void node;

  const found = p.controlsAt("project.video");
  assert.deepEqual(found.map((f) => f.input), ["width", "height"]);
  found[1].control.setValue(480);
  assert.deepEqual(p.overrides(), { latent: { width: 512, height: 480 } });
  assert.deepEqual(p.controlsAt("nowhere.at.all"), []);
});

test("a role naming a place nobody offers is not shown", () => {
  // The same rule a module's panel lives by: absent, not broken, and not a
  // warning either -- this shell may simply not have that region yet.
  const node = host("somewhere.else");
  const p = createPrompts(SLOTS);
  assert.equal(boxes(node).length, 0);
  assert.deepEqual(p.overrides(), {});
});

test("an input fed by another node gets no box", () => {
  // A value written over a link unwires a node, and the server refuses it. A
  // box offering to do that is a box whose every use is an error.
  const node = host();
  createPrompts([{ id: "positive", node: "CLIPTextEncode",
    roles: [{ at: "generation.prompt", input: "clip", label: "CLIP" }],
    inputs: { clip: ["clip", 0] } }]);
  assert.equal(boxes(node).length, 0);
});

test("an input holding something other than text gets no box", () => {
  const node = host();
  createPrompts([{ id: "latent", node: "EmptyLatent",
    roles: [{ at: "generation.prompt", input: "width", label: "Width" }],
    inputs: { width: 512 } }]);
  assert.equal(boxes(node).length, 0, "a number was offered a prompt box");
});

test("a slot with no roles puts nothing anywhere", () => {
  const node = host();
  createPrompts([{ id: "latent", node: "EmptyLatent", inputs: { width: 512 } }]);
  assert.equal(boxes(node).length, 0);
});

test("the pipeline changing underneath takes the boxes with it", () => {
  // The pipeline window can remove the slot a box belongs to. A box left behind
  // sends a value for a slot that is gone, and the server refuses the whole run
  // over a control the user has no way to find.
  const node = host();
  const p = createPrompts(SLOTS);
  boxes(node)[0].value = "a dog";
  fire(boxes(node)[0], "input");

  p.sync(SLOTS.filter((s) => s.id !== "positive"));
  assert.equal(boxes(node).length, 1);
  assert.deepEqual(p.overrides(), { negative: { text: "" } });
});

test("a value saved in the pipeline window is what the box then shows", () => {
  const node = host();
  const p = createPrompts(SLOTS);
  boxes(node)[0].value = "typed here";
  fire(boxes(node)[0], "input");

  p.sync([{ ...SLOTS[0], inputs: { clip: ["clip", 0], text: "saved there" } }]);
  assert.equal(boxes(node)[0].value, "saved there",
    "two windows held different text for one input");
  assert.deepEqual(p.overrides(), { positive: { text: "saved there" } });
});

test("rebuilding leaves nothing behind", () => {
  const node = host();
  const p = createPrompts(SLOTS);
  p.sync([]);
  assert.equal(node.children.length, 0, "a labelled empty space was left behind");
});
