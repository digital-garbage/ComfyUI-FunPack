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

test("what a run produced can be saved to the bin from wherever it is shown", () => {
  // Opened from the temp browser, or anywhere else that hands the viewer a
  // real file identity -- "keep this" should not mean a download and a
  // re-upload by hand.
  const p = page();
  p.viewer.setSource("/view?filename=stray.png&subfolder=&type=temp", "image",
    { filename: "stray.png", subfolder: "", type: "temp" });

  const saveToBin = p.preview.node.querySelector('button[aria-label="Save to bin"]');
  assert.ok(saveToBin, "no Save to bin action in the Preview head");
  saveToBin.click();

  assert.equal(p.bin.items.length, 1);
  assert.equal(p.bin.items[0].file.filename, "stray.png");
});

test("saving something already in the bin says so instead of duplicating it", () => {
  const p = page();
  p.bin.absorb([file("a.png")]);           // also becomes what the viewer shows

  const saveToBin = p.preview.node.querySelector('button[aria-label="Save to bin"]');
  saveToBin.click();

  assert.equal(p.bin.items.length, 1, "the same file was saved twice");
});

test("saving with nothing shown does not throw, and says why", () => {
  const p = page();
  const saveToBin = p.preview.node.querySelector('button[aria-label="Save to bin"]');
  assert.doesNotThrow(() => saveToBin.click());
  assert.equal(p.bin.items.length, 0);
});

test("saving a frame with no video playing says so rather than uploading nothing", async () => {
  // jsdom has no real video decoder, so a genuine capture can only be proven
  // in a real browser (see the Playwright spec) -- this is the guard that
  // keeps a silent no-op from reaching the upload at all.
  const p = page();
  const saveFrame = p.preview.node.querySelector('button[aria-label="Save this frame"]');
  assert.ok(saveFrame, "no Save frame action in the Preview head");

  let asked = false;
  globalThis.fetch = async () => { asked = true; return { ok: true, json: async () => ({}) }; };
  saveFrame.click();
  await new Promise((r) => setTimeout(r, 0));   // let the click's own async handler settle

  assert.equal(asked, false, "uploaded with no frame to upload");
  assert.equal(p.bin.items.length, 0);
});

test("a frame that dedupes to something already in the bin says so, not that it saved", async () => {
  // saveToBin already checked bin.absorb()'s own dedup result; saveFrame did
  // not, so a second capture that happened to upload to the same identity
  // (two rapid clicks, a server that overwrites rather than renames on
  // collision) claimed success while adding nothing.
  const p = page();
  p.viewer.captureFrame = async () => new Blob(["x"], { type: "image/png" });
  globalThis.fetch = async () => ({
    ok: true, json: async () => ({ name: "captured.png", subfolder: "", type: "input" }),
  });

  const saveFrame = p.preview.node.querySelector('button[aria-label="Save this frame"]');
  saveFrame.click();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(p.bin.items.length, 1);
  assert.match(p.preview.node.querySelector(".cx-panel-status").textContent, /Frame saved/);

  saveFrame.click();          // resolves to the exact same identity again
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(p.bin.items.length, 1, "the same capture was added twice");
  assert.match(p.preview.node.querySelector(".cx-panel-status").textContent, /Already in the bin/);
});

test("a frame upload that resolves after the user moved on does not yank the viewer back", async () => {
  // The upload is a real round trip -- by the time it lands the user may
  // already be looking at something else. A save must add to the bin without
  // moving what is on screen, whenever it finishes.
  const p = page();
  p.bin.absorb([file("first.png")]);   // the viewer shows this

  // jsdom cannot decode video; stand in for a real captured frame (the real
  // capture itself is proven in the Playwright spec, against a real video).
  p.viewer.captureFrame = async () => new Blob(["x"], { type: "image/png" });

  let resolveUpload;
  globalThis.fetch = () => new Promise((resolve) => { resolveUpload = resolve; });

  p.preview.node.querySelector('button[aria-label="Save this frame"]').click();
  // Let captureFrame()'s own await, then the call into fetch(), actually
  // happen -- the upload is now in flight, waiting on resolveUpload.
  await Promise.resolve(); await Promise.resolve();
  assert.equal(typeof resolveUpload, "function", "the upload was never started");

  // The user moves on before the upload resolves.
  p.bin.absorb([file("second.png")]);
  assert.match(p.viewer.node.querySelector("img").getAttribute("src"), /second\.png/);

  resolveUpload({ ok: true, json: async () => ({ name: "captured.png", subfolder: "", type: "input" }) });
  await Promise.resolve(); await Promise.resolve(); await Promise.resolve();

  assert.equal(p.bin.items.length, 3, "the captured frame was never saved");
  assert.match(p.viewer.node.querySelector("img").getAttribute("src"), /second\.png/,
    "a save that finished late pulled the viewer back to what it saved");
});

test("a result that failed to load is not something Save to bin can still save", () => {
  // The viewer's own error state says "nothing is showing here" -- `file`
  // disagreeing with that let a dead reference be saved as if it were real.
  const p = page();
  p.bin.absorb([file("gone.png")]);
  p.viewer.node.querySelector("img").dispatchEvent(new window.Event("error"));
  assert.match(p.viewer.node.textContent, /could not be loaded/);
  assert.equal(p.viewer.file, null, "the viewer still claims a file is showing after it failed to load");

  const saveToBin = p.preview.node.querySelector('button[aria-label="Save to bin"]');
  saveToBin.click();

  assert.equal(p.bin.items.length, 1, "the failed file was saved again, over the guard");
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
