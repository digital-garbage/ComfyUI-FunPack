// The parts added to make v5 an app rather than a generate button: the action
// registry and the wheel that shows it, the inspector, and the way in.
//
// One file, because each is small and they are all the same shape: something
// announces, something else draws whatever announced.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom, fire } from "../../composer/tests/_dom.js";

let actions, createWheel, createInspector, openWizard;
test.before(async () => {
  setupDom();
  await import("../../composer/composer.js");
  actions = await import("../actions.js");
  ({ createWheel } = await import("../wheel.js"));
  ({ createInspector } = await import("../inspector.js"));
  ({ open: openWizard } = await import("../wizard.js"));
});
test.after(() => teardownDom());

// --- actions ----------------------------------------------------------------

test("an action is offered by whoever owns it, and taken away with it", () => {
  actions._reset();
  const ran = [];
  const drop = actions.offerAction({ id: "go", label: "Go", run: () => ran.push("go") });

  assert.deepEqual(actions.offered().map((a) => a.id), ["go"]);
  assert.equal(actions.run("go"), true);
  assert.deepEqual(ran, ["go"]);

  drop();
  assert.deepEqual(actions.offered(), []);
  // Not an error: whatever offered it may simply be gone.
  assert.equal(actions.run("go"), false);
});

test("one broken action does not take down the list that showed it", () => {
  actions._reset();
  actions.offerAction({ id: "bad", run: () => { throw new Error("nope"); } });
  actions.offerAction({ id: "good", run: () => {} });

  assert.equal(actions.run("bad"), false);
  assert.equal(actions.run("good"), true, "a neighbour's failure stopped this one");
});

test("offering the same id twice replaces it", () => {
  // Two entries that look identical and do different things is worse than one.
  actions._reset();
  const ran = [];
  actions.offerAction({ id: "go", label: "First", run: () => ran.push(1) });
  actions.offerAction({ id: "go", label: "Second", run: () => ran.push(2) });

  assert.equal(actions.offered().length, 1);
  assert.equal(actions.offered()[0].label, "Second");
  actions.run("go");
  assert.deepEqual(ran, [2]);
});

// --- the wheel --------------------------------------------------------------

const middle = (node, init = {}) => node.dispatchEvent(
  new window.MouseEvent("mousedown", { button: 1, bubbles: true, cancelable: true, ...init }));

test("the middle button opens the wheel on whatever is offered", () => {
  actions._reset();
  actions.offerAction({ id: "generate", label: "Generate", run: () => {} });
  actions.offerAction({ id: "cancel", label: "Cancel", run: () => {} });
  const wheel = createWheel();

  middle(document.body);
  assert.equal(wheel.isOpen, true);
  assert.match(document.body.textContent, /Generate/);
  wheel.close();
  assert.equal(wheel.isOpen, false);
  wheel.destroy();
});

test("a wheel of one is not a wheel", () => {
  // Every action comes from a part of the app that loaded, so "only one" is a
  // real state -- and there is nothing to aim between. The kit refuses it; no
  // wheel is a better answer than an error thrown under the pointer.
  actions._reset();
  const wheel = createWheel();
  middle(document.body);
  assert.equal(wheel.isOpen, false, "a wheel opened with nothing in it");

  actions.offerAction({ id: "a", run: () => {} });
  middle(document.body);
  assert.equal(wheel.isOpen, false, "a wheel opened with one item in it");
  wheel.destroy();
});

test("the middle button's own behaviour is prevented", () => {
  // Otherwise Chrome starts autoscroll under the wheel.
  actions._reset();
  actions.offerAction({ id: "a", run: () => {} });
  actions.offerAction({ id: "b", run: () => {} });
  const wheel = createWheel();
  const event = new window.MouseEvent("mousedown", { button: 1, bubbles: true, cancelable: true });
  document.body.dispatchEvent(event);
  assert.equal(event.defaultPrevented, true);
  wheel.destroy();
});

test("a wheel that is already open is not opened again", () => {
  actions._reset();
  actions.offerAction({ id: "a", run: () => {} });
  actions.offerAction({ id: "b", run: () => {} });
  const wheel = createWheel();
  middle(document.body);
  const first = document.querySelectorAll(".cx-wheel").length;
  middle(document.body);
  assert.equal(document.querySelectorAll(".cx-wheel").length, first);
  wheel.destroy();
});

// --- the inspector ----------------------------------------------------------

function fakeProject(over = {}) {
  const scene = { id: "aaaaaaaaaaaa", text: "", result: null, length: null, rating: null };
  const written = [];
  return {
    written,
    project: { id: "p", name: "Untitled" },
    scenes: [scene],
    selectedId: scene.id,
    selected: scene,
    video: { length: 97 },
    setScene: (id, key, value) => { written.push([id, key, value]); scene[key] = value; },
    rename: (name) => written.push(["project", "name", name]),
    ...over,
  };
}

test("the scene tab edits the scene, and says the crop is not what regenerates", () => {
  const project = fakeProject();
  const inspector = createInspector({ project });
  document.body.replaceChildren(inspector.node);

  assert.match(inspector.node.textContent, /Regenerating uses the project's length/);
  const length = inspector.node.querySelector('input[type="number"]');
  assert.equal(length.value, "97", "a scene with no crop of its own shows the project's length");

  length.value = "48";
  fire(length, "blur");
  assert.deepEqual(project.written.at(-1), [project.selected.id, "length", 48]);
});

test("a rating is kept with the scene", () => {
  const project = fakeProject();
  const inspector = createInspector({ project });
  document.body.replaceChildren(inspector.node);

  const good = [...inspector.node.querySelectorAll("button")].find((b) => b.textContent === "Good");
  fire(good, "click");
  assert.deepEqual(project.written.at(-1), [project.selected.id, "rating", "good"]);
});

test("the project tab renames the project", () => {
  const project = fakeProject();
  const renamed = [];
  const inspector = createInspector({ project, onRename: (n) => renamed.push(n) });
  document.body.replaceChildren(inspector.node);

  inspector.show("project");
  const name = inspector.node.querySelector('input[type="text"]');
  assert.equal(name.value, "Untitled");
  name.value = "Rooftops";
  fire(name, "blur");
  assert.deepEqual(renamed, ["Rooftops"]);
});

test("with no scene selected the inspector says so instead of drawing nothing", () => {
  const inspector = createInspector({ project: fakeProject({ selected: null, scenes: [] }) });
  document.body.replaceChildren(inspector.node);
  assert.match(inspector.node.textContent, /No scene/);
});

// --- the way in -------------------------------------------------------------

test("the wizard can be left, and leaving is an answer", () => {
  // v4's could not be dismissed: a page refresh was the only way out, which
  // makes it a trap rather than a wizard.
  const picked = [];
  const window_ = openWizard({ onPick: (id) => picked.push(id) });
  assert.ok(document.querySelector(".cx-modal"));

  window_.close("dismissed");
  assert.deepEqual(picked, [null], "dismissing said nothing at all");
  assert.equal(document.querySelector(".cx-modal"), null);
});

test("a choice is answered once, however it was closed", () => {
  const picked = [];
  const window_ = openWizard({ onPick: (id) => picked.push(id) });
  const card = [...document.querySelectorAll(".cx-card")]
    .find((c) => /A few scenes/.test(c.textContent));
  fire(card, "click");

  assert.deepEqual(picked, ["scenes"]);
  window_.close("again");
  assert.deepEqual(picked, ["scenes"], "closing after a choice answered a second time");
});
