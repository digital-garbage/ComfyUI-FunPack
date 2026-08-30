// The regions, and the one wire between two of them.
//
// build() is mostly composition, which tests badly and breaks rarely. The
// exception is where two regions are joined: the bin decides what the viewer
// shows, and nothing else in either file says so. Tested here rather than in
// bin.js, because a bin with a fake viewer proves the fake was wired, not the
// app.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom, fire } from "../../composer/tests/_dom.js";

let build, resetMounts;
test.before(async () => {
  setupDom();
  await import("../../composer/composer.js");
  ({ build } = await import("../layout.js"));
  ({ _reset: resetMounts } = await import("../mounts.js"));
});
test.after(() => teardownDom());

const file = (filename) => ({ filename, subfolder: "", type: "output" });

function page() {
  resetMounts();
  const root = document.createElement("div");
  document.body.replaceChildren(root);
  return build(root);
}

test("a result taken into the bin is what the viewer shows", () => {
  const p = page();
  assert.match(p.viewer.node.textContent, /result of the last run/);

  p.bin.absorb([file("a.png")]);
  const img = p.viewer.node.querySelector("img");
  assert.ok(img, "the viewer did not show the result");
  assert.equal(img.getAttribute("src"), "/view?filename=a.png&subfolder=&type=output");
});

test("a video result reaches the viewer as a video", () => {
  const p = page();
  p.bin.absorb([file("clip.mp4")]);
  assert.ok(p.viewer.node.querySelector("video"), "a video was handed to an <img>");
});

test("picking an older result in the bin changes what the viewer shows", () => {
  const p = page();
  p.bin.absorb([file("first.png")]);
  p.bin.absorb([file("second.png")]);
  assert.match(p.viewer.node.querySelector("img").getAttribute("src"), /second\.png/);

  const rows = [...p.assets.node.querySelectorAll(".cx-cell, .cx-media-row")];
  fire(rows[rows.length - 1], "click");            // the oldest is at the end
  assert.match(p.viewer.node.querySelector("img").getAttribute("src"), /first\.png/);
});

test("a control sits in the head of the zone it acts on", () => {
  // The arrangement that makes this an editor rather than a dashboard: the bin's
  // view control is in the Assets head, and Generate is in the head of the zone
  // the run fills -- not in one bar at the bottom acting on whatever is in front.
  const p = page();
  assert.ok(p.assets.node.contains(p.bin.control.node), "the view control left the Assets head");
  assert.ok(p.timeline.node.contains(p.transport.generate.node), "Generate is not in the Timeline head");
  assert.equal(p.assets.node.contains(p.transport.generate.node), false);
  assert.equal(p.page.footer, null, "there is still a bar across the bottom");
});

test("what a run is doing is said in the same head that starts it", () => {
  const p = page();
  const status = p.timeline.node.querySelector(".cx-panel-status");
  assert.ok(status, "the Timeline head says nothing about the run");
  assert.ok(status.contains(p.transport.statusText.node));
  assert.ok(status.contains(p.transport.progress.node));
});

test("a warning about the next run sits inside the zone that starts it", () => {
  // Under the button it is about, not in the head beside it: it is a sentence
  // rather than a chip, and a 34px band is not where a sentence goes.
  const p = page();
  assert.ok(p.timeline.body.contains(p.transport.warning.node));
});

test("the assets mount point still exists for a module", () => {
  const p = page();
  p.bin.absorb([file("a.png")]);
  assert.ok(p.assets.body.contains(p.bin.host.node),
    "the bin is not in the panel body a module also mounts into");
});

test("the prompt is written in a window, not in a zone", () => {
  // A permanent third of the centre column for the thing that is written in
  // bursts and read rarely, taken from the timeline, which is what the project
  // IS. It opens, is written in, and closes.
  const p = page();
  assert.equal(p.constructor.isOpen, false, "the Constructor is open before anyone asked");
  assert.equal(document.querySelector(".cx-modal"), null);

  p.constructor.open();
  assert.equal(p.constructor.isOpen, true);
  const modal = document.querySelector(".cx-modal");
  assert.ok(modal, "the Constructor did not open");
  assert.ok(modal.contains(p.constructor.host.node), "the prompt is not in the window");
  assert.match(modal.textContent, /Constructor/);

  p.constructor.close();
  assert.equal(document.querySelector(".cx-modal"), null);
});

test("what was typed survives the window being closed", () => {
  // The host outlives every opening. A window that rebuilt its contents each
  // time would throw away whatever was written and not saved -- which for a
  // prompt is the whole of it.
  const p = page();
  const box = document.createElement("textarea");
  p.constructor.host.node.appendChild(box);
  box.value = "a cat on a rooftop";

  p.constructor.open();
  p.constructor.close();
  p.constructor.open();

  const found = document.querySelector(".cx-modal textarea");
  assert.equal(found, box, "the window built itself a new prompt box");
  assert.equal(found.value, "a cat on a rooftop");
  p.constructor.close();
});

test("the timeline is where a run is started from", () => {
  const p = page();
  assert.match(p.timeline.node.textContent, /Nothing on the timeline/);
  assert.ok(p.timeline.node.querySelector(".cx-panel-actions"));
  assert.match(p.timeline.node.querySelector(".cx-panel-actions").textContent, /Constructor/);
});
