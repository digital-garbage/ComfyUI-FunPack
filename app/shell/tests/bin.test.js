// The bin: what a session has produced, and what is on screen.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom, fire } from "../../composer/tests/_dom.js";

let createBin, keyOf, kindOf, recallView;
test.before(async () => {
  setupDom();
  await import("../../composer/composer.js");
  ({ createBin, keyOf, kindOf, recallView } = await import("../bin.js"));
});
test.after(() => teardownDom());

const file = (filename, extra = {}) => ({ filename, subfolder: "", type: "output", ...extra });

function bin(opts = {}) {
  const opened = [];
  const handle = createBin({ persist: false, onOpen: (item) => opened.push(item), ...opts });
  document.body.replaceChildren(handle.host.node);
  // Not a spread of the handle: `items`, `selected` and `view` are getters, and
  // spreading one reads it ONCE -- every assertion would then be about the bin
  // as it was before the test did anything to it.
  return {
    opened, control: handle.control, node: handle.host.node,
    absorb: (...a) => handle.absorb(...a),
    open: (...a) => handle.open(...a),
    setView: (...a) => handle.setView(...a),
    get items() { return handle.items; },
    get selected() { return handle.selected; },
    get view() { return handle.view; },
    cells: () => [...handle.host.node.querySelectorAll(".cx-cell, .cx-media-row")],
  };
}

test("a result arrives, is shown, and is the one selected", () => {
  const b = bin();
  b.absorb([file("a.png")]);

  assert.equal(b.items.length, 1);
  assert.equal(b.opened.length, 1);
  assert.equal(b.opened[0].url, "/view?filename=a.png&subfolder=&type=output");
  assert.equal(b.opened[0].kind, "image");
  assert.equal(b.selected, keyOf(file("a.png")));
  assert.equal(b.cells().length, 1);
  assert.equal(b.cells()[0].getAttribute("aria-selected"), "true");
});

test("the same file arriving twice is one entry", () => {
  // A reload adopts a finished run and replays its outputs, so the same three
  // fields arrive again. Keyed by arrival, the bin would fill with copies of
  // whatever was generated before the last reload.
  const b = bin();
  b.absorb([file("a.png")]);
  b.absorb([file("a.png")]);
  b.absorb([file("a.png"), file("b.png")]);

  assert.deepEqual(b.items.map((i) => i.label), ["b.png", "a.png"]);
});

test("a state carrying nothing new leaves the user's choice alone", () => {
  // absorb() runs on EVERY state change -- a progress message is a state
  // change. Redrawing unconditionally walked a click on an older result back to
  // the newest one within milliseconds.
  const b = bin();
  b.absorb([file("first.png"), file("second.png")]);
  assert.equal(b.items[0].label, "second.png");

  b.open(keyOf(file("first.png")));
  const chosen = b.selected;
  b.opened.length = 0;

  b.absorb([file("first.png"), file("second.png")]);   // the same run, again
  b.absorb([]);
  b.absorb();

  assert.equal(b.selected, chosen, "an unchanged state moved the selection");
  assert.deepEqual(b.opened, [], "an unchanged state re-showed a result");
});

test("a result from a NEW run takes the view back", () => {
  const b = bin();
  b.absorb([file("old.png")]);
  b.open(keyOf(file("old.png")));
  b.opened.length = 0;

  b.absorb([file("new.png")]);
  assert.equal(b.selected, keyOf(file("new.png")));
  assert.equal(b.opened.length, 1);
  assert.equal(b.opened[0].label, "new.png");
});

test("the newest of a batch is the one shown", () => {
  const b = bin();
  b.absorb([file("one.png"), file("two.png"), file("three.png")]);
  assert.equal(b.opened.length, 1, "every file in a batch was shown in turn");
  assert.equal(b.opened[0].label, "three.png");
  assert.deepEqual(b.items.map((i) => i.label), ["three.png", "two.png", "one.png"]);
});

test("a video is known as one, and never becomes a <video> in the bin", () => {
  // v4 put live <video> elements in the bin and Chrome's six-per-origin
  // connection pool wedged the whole API behind them: the app stopped answering
  // while the bin loaded.
  const b = bin();
  b.absorb([file("clip.mp4"), file("clip.webm"), file("still.png")]);

  assert.equal(kindOf(file("clip.mp4")), "video");
  assert.equal(b.items.find((i) => i.label === "clip.mp4").kind, "video");
  assert.equal(b.node.querySelector("video"), null);
  const sources = [...b.node.querySelectorAll("img")].map((i) => i.getAttribute("src"));
  assert.deepEqual(sources.filter((s) => /\.(mp4|webm)/.test(s)), [],
    "a video file was handed to an <img> as a thumbnail");
});

test("changing the view keeps the items and the selection", () => {
  const b = bin();
  b.absorb([file("a.png"), file("b.png")]);
  b.open(keyOf(file("a.png")));

  for (const view of ["list", "icons", "grid"]) {
    b.setView(view);
    assert.equal(b.view, view);
    assert.equal(b.items.length, 2, `${view} lost the items`);
    assert.equal(b.cells().length, 2, `${view} drew the wrong number of entries`);
    assert.equal(b.selected, keyOf(file("a.png")), `${view} lost the selection`);
    const on = b.cells().filter((c) => c.getAttribute("aria-selected") === "true");
    assert.equal(on.length, 1, `${view} did not draw the selection`);
  }
});

test("each view lets a result be picked", () => {
  for (const view of ["grid", "list", "icons"]) {
    const b = bin({ view });
    b.absorb([file("a.png"), file("b.png")]);
    b.opened.length = 0;
    fire(b.cells()[1], "click");
    assert.equal(b.opened.length, 1, `${view} does not activate`);
    assert.equal(b.opened[0].label, "a.png", `${view} activated the wrong entry`);
  }
});

test("a view name nobody offers is refused, not drawn", () => {
  const b = bin({ view: "mosaic" });
  assert.equal(b.view, "grid");
  assert.equal(b.setView("mosaic"), "grid");
  assert.equal(b.view, "grid");
});

test("the view is remembered, and a stored name nobody offers is not", () => {
  window.localStorage.removeItem("funpack.bin.view");
  assert.equal(recallView(), "grid");

  const b = createBin({ view: "list" });
  b.setView("icons");
  assert.equal(window.localStorage.getItem("funpack.bin.view"), "icons");
  assert.equal(recallView(), "icons");

  window.localStorage.setItem("funpack.bin.view", "mosaic");
  assert.equal(recallView(), "grid");
  window.localStorage.removeItem("funpack.bin.view");
});

test("the view control and the bin cannot disagree", () => {
  const b = bin({ view: "grid" });
  b.setView("list");
  assert.equal(b.control.value, "list");
});

test("an empty bin says so", () => {
  const b = bin();
  assert.match(b.node.textContent, /Results appear here/);
  assert.equal(b.cells().length, 0);
});
