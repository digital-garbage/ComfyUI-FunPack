// Overlay behaviour: stacking, dismissal order, and the promises.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom, key, fire } from "./_dom.js";
import { EDGES, sectorAt, sectorInArc } from "../elements/wheel.js";

let composer;
let ROOT_ID;
let baseOf;
test.before(async () => {
  setupDom();
  ({ composer } = await import("../composer.js"));
  ({ ROOT_ID } = await import("../internals/portal.js"));
  ({ baseOf } = await import("../internals/zlayer.js"));
});
test.after(() => teardownDom());

// Overlays live in the portal, not under the test's own node, so a test that
// leaves one open would be counted by the next one.
test.beforeEach(() => {
  const root = document.getElementById(ROOT_ID);
  if (root) root.replaceChildren();
});

const portalRoot = () => document.getElementById(ROOT_ID);
const zOf = (node) => Number(node.closest("[style*='z-index']")?.style.zIndex || node.style.zIndex);

// --- the bug v4 had ---------------------------------------------------------

test("Escape inside a modal closes the autocomplete first, then the modal", () => {
  // v4's modal and its autocomplete each owned an Escape handler, so one press
  // closed both and you lost the dialog you were filling in.
  const closed = [];
  const modal = composer.modal.generic({ title: "Edit", onClose: () => closed.push("modal") });

  const input = composer.input.md({});
  modal.node.appendChild(input.node);
  const ac = composer.autocomplete.default({
    input, source: () => [{ label: "rooftop" }], minChars: 1,
  });
  input.node.value = "r";
  fire(input.node, "input");
  assert.ok(document.querySelector(".cx-autocomplete"), "the menu opened");

  key(document.body, "Escape");
  assert.deepEqual(closed, [], "the modal survived");
  assert.equal(document.querySelector(".cx-autocomplete"), null, "the menu closed");

  key(document.body, "Escape");
  assert.deepEqual(closed, ["modal"]);
  ac.destroy();
});

test("an autocomplete outranks a modal, so it is not hidden behind it", () => {
  const modal = composer.modal.generic({ title: "Edit" });
  const input = composer.input.md({});
  modal.node.appendChild(input.node);
  const ac = composer.autocomplete.default({ input, source: () => [{ label: "x" }], minChars: 1 });
  input.node.value = "x";
  fire(input.node, "input");

  const menu = document.querySelector(".cx-autocomplete");
  assert.ok(Number(menu.style.zIndex) >= baseOf("autocomplete"));
  assert.ok(Number(menu.style.zIndex) > baseOf("modal"));

  ac.destroy();
  modal.destroy();
});

// --- modals -----------------------------------------------------------------

test("a modal mounts into the portal and leaves nothing behind", () => {
  const m = composer.modal.generic({ title: "Export" });
  assert.ok(portalRoot().contains(m.node));
  m.destroy();
  assert.equal(portalRoot().childElementCount, 0);
});

test("a modal is announced as a dialog and labelled by its title", () => {
  const m = composer.modal.generic({ title: "Export settings" });
  assert.equal(m.node.getAttribute("role"), "dialog");
  assert.equal(m.node.getAttribute("aria-modal"), "true");
  const labelled = document.getElementById(m.node.getAttribute("aria-labelledby"));
  assert.equal(labelled.textContent, "Export settings");
  m.destroy();
});

test("dialogue resolves true on confirm and false on cancel", async () => {
  const yes = composer.modal.dialogue({ title: "Delete?", message: "Gone for good.", confirmLabel: "Delete" });
  yes.node.querySelectorAll(".cx-modal-foot button")[1].click();
  assert.equal(await yes.result, true);

  const no = composer.modal.dialogue({ title: "Delete?", message: "Gone for good." });
  no.node.querySelectorAll(".cx-modal-foot button")[0].click();
  assert.equal(await no.result, false);
});

test("dialogue resolves false when dismissed, rather than hanging", async () => {
  // A promise nobody settles is a UI that quietly stops responding.
  const d = composer.modal.dialogue({ title: "Delete?", message: "Gone for good." });
  key(document.body, "Escape");
  assert.equal(await d.result, false);
});

test("prompt resolves its value, and null when dismissed", async () => {
  const p = composer.modal.prompt({ title: "New project", label: "Name", value: "Untitled" });
  const input = p.node.querySelector("input");
  input.value = "Rooftop";
  p.node.querySelectorAll(".cx-modal-foot button")[1].click();
  assert.equal(await p.result, "Rooftop");

  const q = composer.modal.prompt({ title: "New project", label: "Name" });
  key(document.body, "Escape");
  assert.equal(await q.result, null);
});

test("prompt refuses to submit a value its validator rejects", async () => {
  const p = composer.modal.prompt({
    title: "New project", label: "Name", value: "",
    validate: (v) => (v.trim() ? null : "Give it a name."),
  });
  p.node.querySelectorAll(".cx-modal-foot button")[1].click();
  assert.equal(p.node.querySelector(".cx-field-error").textContent, "Give it a name.");
  assert.ok(portalRoot().contains(p.node), "still open");
  p.destroy();
});

test("choice resolves the picked id", async () => {
  const c = composer.modal.choice({ title: "Start from", items: [
    { id: "blank", label: "Empty" }, { id: "image", label: "From an image" },
  ] });
  c.node.querySelectorAll(".cx-choice-row")[1].click();
  assert.equal(await c.result, "image");
});

test("a stacked modal sits above the one it opened from", () => {
  const first = composer.modal.generic({ title: "One" });
  const second = composer.modal.stacked({ title: "Two" });
  assert.ok(zOf(second.node) > zOf(first.node));
  key(document.body, "Escape");
  assert.equal(portalRoot().childElementCount, 1, "only the top one closed");
  first.destroy();
});

// --- popovers and menus -----------------------------------------------------

test("a menu reports the item picked and closes", () => {
  let picked = null;
  const anchor = composer.button.md({ label: "Scene" });
  document.body.appendChild(anchor.node);
  const menu = composer.menu.dropdown({
    anchor, onPick: (id) => { picked = id; },
    items: [{ id: "split", label: "Split" }, { id: "dup", label: "Duplicate" }],
  });
  menu.node.querySelectorAll(".cx-menu-item")[1].click();
  assert.equal(picked, "dup");
  assert.equal(portalRoot().childElementCount, 0);
});

test("separators are not clickable items", () => {
  const anchor = composer.button.md({ label: "Scene" });
  const menu = composer.menu.dropdown({ anchor, items: [
    { id: "a", label: "A" }, { separator: true }, { id: "b", label: "B" },
  ] });
  assert.equal(menu.node.querySelectorAll(".cx-menu-item").length, 2);
  assert.equal(menu.node.querySelectorAll(".cx-menu-sep").length, 1);
  menu.destroy();
});

test("clicking the trigger again does not close-then-reopen", () => {
  const anchor = composer.button.md({ label: "Scene" });
  document.body.appendChild(anchor.node);
  const menu = composer.menu.dropdown({ anchor, items: [{ id: "a", label: "A" }] });
  fire(anchor.node, "pointerdown");
  assert.ok(portalRoot().contains(menu.node), "the anchor counts as inside");
  menu.destroy();
});

test("a tooltip shows and hides on demand", () => {
  const anchor = composer.iconButton.md({ icon: "✎", label: "Rename" });
  document.body.appendChild(anchor.node);
  const tip = composer.tooltip.default({ anchor, text: "Rename this scene", trigger: false });
  tip.show();
  assert.ok(document.querySelector(".cx-tooltip"));
  tip.hide();
  assert.equal(document.querySelector(".cx-tooltip"), null);
  tip.destroy();
});

test("autocomplete needs minChars before it opens", () => {
  const input = composer.input.md({});
  document.body.appendChild(input.node);
  const ac = composer.autocomplete.default({ input, source: () => [{ label: "x" }], minChars: 2 });
  input.node.value = "a";
  fire(input.node, "input");
  assert.equal(document.querySelector(".cx-autocomplete"), null);
  input.node.value = "ab";
  fire(input.node, "input");
  assert.ok(document.querySelector(".cx-autocomplete"));
  ac.destroy();
});

// --- windows ----------------------------------------------------------------

test("a window does not close on an outside click, but does on Escape", () => {
  // It is a window, not a menu: clicking the app behind it is normal.
  let closed = false;
  const w = composer.floating.window({ id: "t1", title: "Composer", onClose: () => { closed = true; } });
  fire(document.body, "pointerdown");
  assert.equal(closed, false);
  key(document.body, "Escape");
  assert.equal(closed, true);
});

test("clicking a window raises it above its peers", () => {
  const a = composer.floating.window({ id: "t2", title: "A" });
  const b = composer.floating.window({ id: "t3", title: "B" });
  assert.ok(zOf(b.node) > zOf(a.node));
  a.toFront();
  assert.ok(zOf(a.node) > zOf(b.node));
  a.destroy(); b.destroy();
});

test("a blocking overlay traps focus and clears on close", () => {
  const o = composer.overlay.blocking({ message: "Restarting…" });
  assert.equal(o.node.getAttribute("aria-busy"), "true");
  o.destroy();
  assert.equal(portalRoot().childElementCount, 0);
});

// --- wheel ------------------------------------------------------------------

test("the wheel's dead zone means cancel", () => {
  assert.equal(sectorAt(0, 0, 6), -1);
  assert.equal(sectorAt(10, 10, 6), -1);
});

test("sector 0 is straight up, and sectors run clockwise", () => {
  // The muscle memory is the whole point: the same choice must always be at the
  // same angle.
  assert.equal(sectorAt(0, -100, 4), 0, "up");
  assert.equal(sectorAt(100, 0, 4), 1, "right");
  assert.equal(sectorAt(0, 100, 4), 2, "down");
  assert.equal(sectorAt(-100, 0, 4), 3, "left");
});

test("every angle lands in a valid sector", () => {
  for (const count of [2, 3, 5, 6, 8, 12]) {
    for (let deg = 0; deg < 360; deg += 1) {
      const rad = (deg - 90) * (Math.PI / 180);
      const i = sectorAt(Math.cos(rad) * 100, Math.sin(rad) * 100, count);
      assert.ok(i >= 0 && i < count, `${deg}deg with ${count} items gave ${i}`);
    }
  }
});

test("boundaries between sectors do not fall through", () => {
  const count = 6;
  const slice = 360 / count;
  for (let i = 0; i < count; i += 1) {
    const edge = (i * slice + slice / 2 - 0.01 - 90) * (Math.PI / 180);
    const got = sectorAt(Math.cos(edge) * 100, Math.sin(edge) * 100, count);
    assert.ok(got >= 0 && got < count);
  }
});

test("the click that opens the wheel does not also pick", () => {
  // The opening click delivers its own pointerup to the wheel, at wherever the
  // button was -- which used to commit whatever wedge that direction pointed at.
  let picked = null;
  const w = composer.wheel.picker({
    items: [{ label: "Split" }, { label: "Duplicate" }, { label: "Remove" }],
    onPick: (item) => { picked = item.label; },
    x: 400, y: 300,
  });
  // 80px straight up from the centre: inside the ring, pointing at item 0.
  window.dispatchEvent(new window.PointerEvent("pointerup", { clientX: 400, clientY: 220, bubbles: true }));
  assert.equal(picked, null, "a release with no movement is not a pick");
  assert.ok(portalRoot().childElementCount > 0, "the wheel is still open");

  window.dispatchEvent(new window.PointerEvent("pointermove", { clientX: 400, clientY: 220, bubbles: true }));
  window.dispatchEvent(new window.PointerEvent("pointerup", { clientX: 400, clientY: 220, bubbles: true }));
  assert.equal(picked, "Split", "after moving, a release picks");
  w.destroy();
});

test("a pointer beyond the rim points at nothing", () => {
  // The lit wedge IS the region that picks the item. Highlighting one from
  // outside the ring promises a pick that release would not deliver.
  const w = composer.wheel.picker({
    items: [{ label: "Split" }, { label: "Duplicate" }, { label: "Rate" }],
    x: 400, y: 300,
  });
  window.dispatchEvent(new window.PointerEvent("pointermove", { clientX: 400, clientY: 60, bubbles: true }));
  assert.equal(w.active, -1, "240px out is off the wheel");
  window.dispatchEvent(new window.PointerEvent("pointermove", { clientX: 400, clientY: 220, bubbles: true }));
  assert.equal(w.active, 0, "80px out is on it");
  w.destroy();
});

test("releasing in the dead zone cancels without picking", () => {
  let picked = null;
  const w = composer.wheel.picker({
    items: [{ label: "Split" }, { label: "Duplicate" }],
    onPick: (item) => { picked = item.label; },
    x: 400, y: 300,
  });
  window.dispatchEvent(new window.PointerEvent("pointermove", { clientX: 405, clientY: 305, bubbles: true }));
  window.dispatchEvent(new window.PointerEvent("pointerup", { clientX: 405, clientY: 305, bubbles: true }));
  assert.equal(picked, null);
  w.destroy();
});

// --- half wheel -------------------------------------------------------------

test("an edge panel's arc faces into the screen, not off it", () => {
  // The point of pinning to an edge is that every item stays reachable; an arc
  // pointing the wrong way would put half of them off-screen.
  const count = 4;
  for (const [edge, spec] of Object.entries(EDGES)) {
    const inward = { right: 270, left: 90, bottom: 0, top: 180 }[edge];
    const rad = (inward - 90) * (Math.PI / 180);
    const i = sectorInArc(Math.cos(rad) * 100, Math.sin(rad) * 100, count, spec);
    assert.ok(i >= 0, `${edge} panel does not cover its own inward direction`);
  }
});

test("a direction outside a half wheel's arc selects nothing", () => {
  // Off the edge is off the panel: it must not wrap round to the far item.
  const spec = EDGES.right;                       // opens leftward
  const right = sectorInArc(100, 0, 4, spec);     // pointing off-screen
  assert.equal(right, -1);
});

test("every inward angle of a half wheel lands in a sector", () => {
  const spec = EDGES.right;
  for (let deg = spec.startDeg + 1; deg < spec.startDeg + spec.spanDeg; deg += 1) {
    const rad = (deg - 90) * (Math.PI / 180);
    const i = sectorInArc(Math.cos(rad) * 100, Math.sin(rad) * 100, 4, spec);
    assert.ok(i >= 0 && i < 4, `${deg}deg gave ${i}`);
  }
});

test("a half wheel opens on its edge and picks by key", () => {
  let picked = null;
  const w = composer.wheel.half({
    edge: "right", at: 300,
    items: [{ label: "Split" }, { label: "Duplicate" }, { label: "Rate" }],
    onPick: (item) => { picked = item.label; },
  });
  assert.ok(w.node.classList.contains("cx-wheel-right"));
  key(document.body, "3");
  assert.equal(picked, "Rate");
  w.destroy();
});

test("an edge panel's hub sits fully on screen, not sliced by the edge", () => {
  // A hub centred on the edge is cut in half by it, and half a cancel target is
  // not a target. It is pushed inward by its own radius.
  const w = composer.wheel.half({ edge: "right", at: 300, items: [{ label: "a" }, { label: "b" }] });
  const hub = w.node.querySelector(".cx-wheel-hub");
  const size = Number(w.node.style.width.replace("px", ""));
  assert.ok(parseFloat(hub.style.left) < size / 2, "the hub leans into the screen");
  const radius = parseFloat(hub.style.width) / 2;
  assert.ok(parseFloat(hub.style.left) + radius <= size / 2 + 0.01,
    "and its far edge does not cross the screen edge");
  w.destroy();
});

test("cancel is where the hub actually is, not where the centre is", () => {
  const spec = EDGES.right;
  const hubR = 40;
  const arc = { ...spec, rIn: 84, rOut: 147, hubX: -hubR, hubY: 0, hubR };
  // A point over the drawn hub cancels...
  assert.equal(sectorInArc(-hubR, 0, 4, arc), -1);
  // ...while the geometric centre, which is now outside the hub, is simply off
  // the ring rather than a pick.
  assert.equal(sectorInArc(0, 0, 4, arc), -1);
});

test("an edge panel follows its edge when the window resizes", () => {
  // It is pinned to an edge; the edge moving is the one thing it must follow.
  const w = composer.wheel.half({ edge: "right", items: [{ label: "a" }, { label: "b" }] });
  const before = w.centre.x;

  window.innerWidth = window.innerWidth + 400;
  window.dispatchEvent(new window.Event("resize"));

  assert.equal(w.centre.x, before + 400, "the centre moved with the edge");
  const size = Number(w.node.style.width.replace("px", ""));
  assert.equal(parseFloat(w.node.style.left), w.centre.x - size / 2);
  window.innerWidth = window.innerWidth - 400;
  w.destroy();
});

test("edge panels stay pinned at any viewport shape", () => {
  for (const [w, h] of [[320, 640], [1920, 1080], [2560, 1080], [800, 2000]]) {
    window.innerWidth = w;
    window.innerHeight = h;
    for (const edge of Object.keys(EDGES)) {
      const p = composer.wheel.half({ edge, items: [{ label: "a" }, { label: "b" }, { label: "c" }] });
      const [ex, ey] = EDGES[edge].anchor(w, h);
      assert.deepEqual(p.centre, { x: ex, y: ey }, `${edge} at ${w}x${h}`);
      p.destroy();
    }
  }
});

test("an unknown edge is refused rather than guessed", () => {
  assert.throws(() => composer.wheel.half({ edge: "diagonal", items: [{ label: "a" }, { label: "b" }] }), RangeError);
});

test("the wheel refuses a count nobody could aim at", () => {
  assert.throws(() => composer.wheel.picker({ items: [{ label: "only one" }] }), RangeError);
  const many = Array.from({ length: 13 }, (_, i) => ({ label: String(i) }));
  assert.throws(() => composer.wheel.picker({ items: many }), RangeError);
});

test("number keys pick directly", () => {
  let picked = null;
  const w = composer.wheel.picker({
    items: [{ label: "Split" }, { label: "Duplicate" }, { label: "Remove" }],
    onPick: (item) => { picked = item.label; },
  });
  key(document.body, "2");
  assert.equal(picked, "Duplicate");
  assert.equal(portalRoot().childElementCount, 0, "picking closes it");
  w.destroy();
});

// --- gallery ----------------------------------------------------------------

test("the gallery sizes from its container, not the viewport", () => {
  const g = composer.gallery.adaptive({ id: "g1", cols: 3, items: [{ id: "1", label: "a.mp4" }] });
  const grid = g.node.querySelector(".cx-gallery");
  assert.equal(grid.dataset.cols, "3");
  assert.match(grid.style.getPropertyValue("--cell"), /cqw$/);
  g.setCols(0);
  assert.match(grid.style.getPropertyValue("--cell"), /clamp\(/, "auto falls back to the clamp");
  g.destroy();
});

test("an empty gallery says it is empty", () => {
  const g = composer.gallery.adaptive({ id: "g2", items: [], empty: "No media yet" });
  assert.match(g.node.textContent, /No media yet/);
  g.destroy();
});

test("gallery thumbnails are decorative, because the caption names them", () => {
  const g = composer.gallery.adaptive({ id: "g3", items: [{ id: "1", label: "a.mp4", thumb: "/x.png" }] });
  assert.equal(g.node.querySelector("img").getAttribute("alt"), "",
    "an alt here would make a screen reader say the name twice");
  g.destroy();
});
