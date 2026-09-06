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

// What the server says a node's inputs look like. The control a role gets comes
// from HERE, not from a guess about the value, which is what makes a combo a
// dropdown of the node's own choices.
const NODES = {
  CLIPTextEncode: { widgets: [{ name: "text", type: "STRING", multiline: true, default: "" }] },
  EmptyLatent: { widgets: [
    { name: "width", type: "INT", default: 512, min: 16, max: 4096, step: 8 },
    { name: "length", type: "INT", default: 1, min: 1, max: 512 },
  ] },
  Sampler: { widgets: [
    { name: "sampler_name", type: "COMBO", choices: ["euler", "dpmpp_2m"], default: "euler" },
  ] },
};
const describe = async (classes) =>
  Object.fromEntries(classes.map((c) => [c, NODES[c] || null]));

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

const make = (slots, opts = {}) => createPrompts(slots, { describe, ...opts });
const boxes = (node) => [...node.querySelectorAll("textarea")];

test("a slot that names a place gets a control there", async () => {
  const node = host();
  const p = await make(SLOTS);

  assert.equal(boxes(node).length, 2);
  assert.deepEqual([...node.querySelectorAll(".cx-label")].map((l) => l.textContent),
    ["Prompt", "Negative"]);
  assert.equal(boxes(node)[0].value, "a cat", "the control did not show what the slot holds");
  assert.deepEqual(p.fields.map((f) => f.slot), ["positive", "negative"]);
});

test("what is typed comes back addressed by slot and input", async () => {
  const node = host();
  const p = await make(SLOTS);

  // Committed, not per keystroke: the control is the node's own, and the node's
  // own control commits on blur. boot blurs before it saves for the same reason.
  boxes(node)[0].value = "a dog on a roof";
  fire(boxes(node)[0], "blur");

  assert.deepEqual(p.overrides(), {
    positive: { text: "a dog on a roof" },
    negative: { text: "" },
  });
});

test("the control is the node's, so a combo is its own list of choices", async () => {
  // The whole reason this asks the server what a node looks like: a dropdown of
  // the real samplers, and a number that obeys the real bounds, without this
  // file knowing that samplers exist.
  const node = host("generation.sampling");
  const p = await make([{ id: "sampler", node: "Sampler",
    roles: [{ at: "generation.sampling", input: "sampler_name", label: "Sampler" }],
    inputs: { sampler_name: "dpmpp_2m" } }]);

  const select = node.querySelector("select");
  assert.ok(select, "a combo did not get a dropdown");
  assert.deepEqual([...select.options].map((o) => o.value), ["euler", "dpmpp_2m"]);
  assert.equal(select.value, "dpmpp_2m", "the dropdown did not show the slot's value");

  select.value = "euler";
  fire(select, "change");
  assert.deepEqual(p.overrides(), { sampler: { sampler_name: "euler" } });
});

test("a numeric input gets a number control, not a prompt box", async () => {
  const node = host("project.video");
  const p = await make([{ id: "latent", node: "EmptyLatent",
    roles: [{ at: "project.video", input: "width", label: "Width" },
            { at: "project.video", input: "length", label: "Length" }],
    inputs: { width: 832, length: 97 } }]);

  assert.equal(boxes(node).length, 0);
  assert.equal(node.querySelectorAll('input[type="number"]').length, 2);
  assert.deepEqual(p.overrides(), { latent: { width: 832, length: 97 } });
});

test("a changed control says which input it was", async () => {
  // Two places take a value -- a scene's text and the project's settings -- and
  // "something changed, go and read it all" does not say which one.
  const node = host("project.video");
  const told = [];
  await make([{ id: "latent", node: "EmptyLatent",
    roles: [{ at: "project.video", input: "width", label: "Width" }],
    inputs: { width: 512 } }], { onChange: (field) => told.push(field) });

  const box = node.querySelector('input[type="number"]');
  box.value = "832";
  fire(box, "blur");

  assert.equal(told.length, 1);
  assert.equal(told[0].at, "project.video");
  assert.equal(told[0].input, "width");
  assert.equal(told[0].slot, "latent");
  assert.equal(told[0].value, 832);
});

test("setting a control's value is what the run then sends", async () => {
  // The box and the value behind it are one thing. Seeded from the project on
  // open and from the scene on selection, and either would be a lie if the
  // control moved and the value did not.
  const node = host("project.video");
  const p = await make([{ id: "latent", node: "EmptyLatent",
    roles: [{ at: "project.video", input: "width", label: "Width" },
            { at: "project.video", input: "length", label: "Length" }],
    inputs: { width: 512, length: 1 } }]);

  const found = p.controlsAt("project.video");
  assert.deepEqual(found.map((f) => f.input), ["width", "length"]);
  found[0].control.setValue(640);

  assert.equal(found[0].control.value, 640);
  assert.equal(node.querySelector('input[type="number"]').value, "640");
  assert.deepEqual(p.overrides(), { latent: { width: 640, length: 1 } });
  assert.deepEqual(p.controlsAt("nowhere.at.all"), []);
});

test("a role naming a place nobody offers is not shown", async () => {
  // The same rule a module's panel lives by: absent, not broken, and not a
  // warning either -- this shell may simply not have that region yet.
  const node = host("somewhere.else");
  const p = await make(SLOTS);
  assert.equal(boxes(node).length, 0);
  assert.deepEqual(p.overrides(), {});
});

test("an input fed by another node gets no control", async () => {
  // A value written over a link unwires a node, and the server refuses it. A
  // control offering to do that is one whose every use is an error.
  const node = host();
  await make([{ id: "positive", node: "CLIPTextEncode",
    roles: [{ at: "generation.prompt", input: "clip", label: "CLIP" }],
    inputs: { clip: ["clip", 0] } }]);
  assert.equal(node.children.length, 0);
});

test("an input the node does not describe gets no control", async () => {
  // A pipeline can name an input a node has not got -- after a node swap, or a
  // hand-edited pipeline. Nothing sane to draw means nothing drawn.
  const node = host();
  await make([{ id: "positive", node: "CLIPTextEncode",
    roles: [{ at: "generation.prompt", input: "gone", label: "Gone" }],
    inputs: {} }]);
  assert.equal(node.children.length, 0);
});

test("a slot with no roles puts nothing anywhere", async () => {
  const node = host();
  await make([{ id: "latent", node: "EmptyLatent", inputs: { width: 512 } }]);
  assert.equal(node.children.length, 0);
});

test("the pipeline changing underneath takes the controls with it", async () => {
  // The pipeline window can remove the slot a control belongs to. One left
  // behind sends a value for a slot that is gone, and the server refuses the
  // whole run over a control the user has no way to find.
  const node = host();
  const p = await make(SLOTS);
  boxes(node)[0].value = "a dog";
  fire(boxes(node)[0], "blur");

  await p.sync(SLOTS.filter((s) => s.id !== "positive"));
  assert.equal(boxes(node).length, 1);
  assert.deepEqual(p.overrides(), { negative: { text: "" } });
});

test("a value saved in the pipeline window is what the control then shows", async () => {
  const node = host();
  const p = await make(SLOTS);
  boxes(node)[0].value = "typed here";
  fire(boxes(node)[0], "blur");

  await p.sync([{ ...SLOTS[0], inputs: { clip: ["clip", 0], text: "saved there" } }]);
  assert.equal(boxes(node)[0].value, "saved there",
    "two windows held different text for one input");
  assert.deepEqual(p.overrides(), { positive: { text: "saved there" } });
});

test("rebuilding leaves nothing behind", async () => {
  const node = host();
  const p = await make(SLOTS);
  await p.sync([]);
  assert.equal(node.children.length, 0, "a labelled empty space was left behind");
});
