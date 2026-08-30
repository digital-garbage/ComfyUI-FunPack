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

test("the bin's view control sits in the Assets panel, not in the transport row", () => {
  // Every control that is about a run belongs beside Generate; this one is not
  // about a run, and putting it there is how that row stops being readable.
  const p = page();
  assert.ok(p.assets.node.contains(p.bin.control.node));
  assert.equal(p.transport.node.contains(p.bin.control.node), false);
});

test("the assets mount point still exists for a module", () => {
  const p = page();
  p.bin.absorb([file("a.png")]);
  assert.ok(p.assets.body.contains(p.bin.host.node),
    "the bin is not in the panel body a module also mounts into");
});
