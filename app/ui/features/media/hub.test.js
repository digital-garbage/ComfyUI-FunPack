import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let hub;
test.before(async () => { setupDom(); hub = (await import("./index.js")).default; });
test.after(() => teardownDom());

test("a click on a tile looks at it on the monitor; the context menu's resolution-source entry sets the scene's source, audio refused", async () => {
  const set = [], said = [];
  const project = { selected: { id: "s1" }, setScene: (...a) => set.push(a) };
  const api = { media: async () => ({ media: [{ id: "m1", name: "cat.png", kind: "image" }, { id: "m2", name: "a.wav", kind: "audio" }] }) };
  const host = document.createElement("div");
  const app = { project, api, mediaPeek: {}, say: (w) => said.push(w), on: () => () => {} };
  hub.setup({ host, app });
  await new Promise((r) => setTimeout(r, 10));
  const cells = [...host.querySelectorAll(".cx-cell")];
  assert.equal(cells.length, 2);
  const named = (n) => cells.find((x) => x.textContent.includes(n));
  named("cat.png").click();
  assert.deepEqual([app.mediaPeek.item.id, said.filter((w) => w === "media.peek")], ["m1", ["media.peek"]]);
  assert.deepEqual(set, []);                                // looking changes nothing
  named("a.wav").click();
  assert.equal(app.mediaPeek.item.id, "m2");
  named("cat.png").dispatchEvent(new window.MouseEvent("contextmenu", { bubbles: true, cancelable: true }));
  [...document.querySelectorAll(".cx-menu-item")].find((b) => b.textContent.includes("aspect ratio source")).click();
  assert.deepEqual(set, [["s1", "source_image", "m1"]]);
});


test("the selected scene's references are marked on their tiles, and the menu takes one back off", async () => {
  const set = [];
  const project = { selected: { id: "s1", references: ["m2", "m1"] }, setScene: (...a) => set.push(a) };
  const api = { media: async () => ({ media: [{ id: "m1", name: "cat.png", kind: "image" }, { id: "m2", name: "dog.png", kind: "image" }, { id: "m3", name: "owl.png", kind: "image" }] }) };
  const host = document.createElement("div");
  hub.setup({ host, app: { project, api, mediaPeek: {}, say: () => {}, on: () => () => {} } });
  await new Promise((r) => setTimeout(r, 10));
  const named = (n) => [...host.querySelectorAll(".cx-cell")].find((x) => x.textContent.includes(n));
  const flag = (n) => [...named(n).querySelectorAll(".cx-cell-flag")].map((x) => x.textContent).join();
  assert.deepEqual([flag("dog.png"), flag("cat.png"), flag("owl.png")], ["R1", "R2", ""]);
  const clear = [...host.querySelectorAll("button")].find((b) => b.textContent.startsWith("Clear references"));
  assert.ok(clear && !clear.hidden && /\(2\)/.test(clear.textContent), "one click takes them all off");
  document.querySelectorAll(".cx-popover").forEach((n) => n.remove());
  named("dog.png").dispatchEvent(new window.MouseEvent("contextmenu", { bubbles: true, cancelable: true }));
  const stop = [...document.querySelectorAll(".cx-menu-item")].find((b) => b.textContent.includes("Stop using as a reference"));
  assert.ok(stop && !stop.disabled, "not greyed: it can be taken off");
  stop.click();
  clear.click();
  assert.deepEqual(set, [["s1", "references", ["m1"]], ["s1", "references", []]]);
});

test("a changed pin repaints the slot numbers on the tiles even when the list of references is the same", async () => {
  const fixed = { m1: 2 };
  const project = { selected: { id: "s1", references: ["m1", "m2"] }, project: { fixed_references: fixed }, setScene: () => {} };
  const api = { media: async () => ({ media: [{ id: "m1", name: "cat.png", kind: "image" }, { id: "m2", name: "dog.png", kind: "image" }] }) };
  const host = document.createElement("div");
  let fire;
  hub.setup({ host, app: { project, api, mediaPeek: {}, say: () => {}, on: (cb) => { fire = cb; return () => {}; } } });
  await new Promise((r) => setTimeout(r, 10));
  const flag = (n) => [...[...host.querySelectorAll(".cx-cell")].find((x) => x.textContent.includes(n)).querySelectorAll(".cx-cell-flag")].map((x) => x.textContent).join();
  assert.deepEqual([flag("cat.png"), flag("dog.png")], ["📌R2", "R1"], "pinned at 2 (marked with a pin), the other takes the free number");
  delete fixed.m1;
  fire("project");
  assert.deepEqual([flag("cat.png"), flag("dog.png")], ["R1", "R2"], "unpinned: back to list order");
});

test("a reference's tile menu pins it at the slot number it holds, for every scene, and takes the pin off again", async () => {
  const doc = { fixed_references: {} };
  const project = { selected: { id: "s1", references: ["m1", "m2"] }, project: doc, setScene: () => {}, edit: (fn) => fn(doc) };
  const api = { media: async () => ({ media: [{ id: "m1", name: "cat.png", kind: "image" }, { id: "m2", name: "dog.png", kind: "image" }, { id: "m3", name: "owl.png", kind: "image" }] }) };
  const host = document.createElement("div");
  hub.setup({ host, app: { project, api, mediaPeek: {}, say: () => {}, on: () => () => {} } });
  await new Promise((r) => setTimeout(r, 10));
  const named = (n) => [...host.querySelectorAll(".cx-cell")].find((x) => x.textContent.includes(n));
  const menuFor = (n) => {
    document.querySelectorAll(".cx-popover").forEach((x) => x.remove());
    named(n).dispatchEvent(new window.MouseEvent("contextmenu", { bubbles: true, cancelable: true }));
    return [...document.querySelectorAll(".cx-menu-item")].map((b) => b.textContent);
  };
  assert.ok(menuFor("dog.png").includes("Pin reference to slot R2"));
  assert.ok(!menuFor("owl.png").some((t) => t.includes("slot R")), "a picture that is no reference has nothing to pin");
  menuFor("dog.png");
  [...document.querySelectorAll(".cx-menu-item")].find((b) => b.textContent === "Pin reference to slot R2").click();
  assert.deepEqual(doc.fixed_references, { m2: 2 });
  assert.ok(menuFor("dog.png").includes("Unpin reference from slot R2"));
  [...document.querySelectorAll(".cx-menu-item")].find((b) => b.textContent === "Unpin reference from slot R2").click();
  assert.deepEqual(doc.fixed_references, {});
});

test("toggling a reference leaves the other tiles untouched, so the bin does not jump or reload its thumbnails", async () => {
  const scene = { id: "s1", references: [] };
  const project = { selected: scene, project: { fixed_references: {} }, setScene: () => {} };
  const api = { media: async () => ({ media: [{ id: "m1", name: "cat.png", kind: "image" }, { id: "m2", name: "dog.png", kind: "image" }, { id: "m3", name: "owl.png", kind: "image" }] }) };
  const host = document.createElement("div");
  let fire;
  hub.setup({ host, app: { project, api, mediaPeek: {}, say: () => {}, on: (cb) => { fire = cb; return () => {}; } } });
  await new Promise((r) => setTimeout(r, 10));
  const tiles = () => [...host.querySelectorAll(".cx-cell")];
  const before = tiles();
  scene.references = ["m2"];
  fire("change");
  const after = tiles();
  assert.equal(after.length, 3);
  assert.equal(after[0], before[0], "cat.png: same tile");
  assert.equal(after[2], before[2], "owl.png: same tile");
  assert.notEqual(after[1], before[1], "dog.png is the one that changed: it gained its R1 mark");
  assert.equal(after[1].querySelector(".cx-cell-flag").textContent, "R1");
});

test("redrawing the bin keeps keyboard focus on the tile that has it", async () => {
  const project = { selected: { id: "s1", references: [] }, project: {}, setScene: () => {} };
  const api = { media: async () => ({ media: [{ id: "m1", name: "cat.png", kind: "image" }, { id: "m2", name: "dog.png", kind: "image" }] }) };
  const host = document.createElement("div");
  document.body.append(host);
  let fire;
  hub.setup({ host, app: { project, api, mediaPeek: {}, say: () => {}, on: (cb) => { fire = cb; return () => {}; } } });
  await new Promise((r) => setTimeout(r, 10));
  const tile = [...host.querySelectorAll(".cx-cell")].find((x) => x.textContent.includes("dog.png"));
  tile.focus(); const was = document.activeElement === tile;
  project.selected.references = ["m1"];
  fire("project");
  const now = [...host.querySelectorAll(".cx-cell")].find((x) => x.textContent.includes("dog.png")); assert.ok(was, "focusable"); assert.equal(now, tile, "same tile node"); assert.equal(document.activeElement, tile, "the focused tile survives the redraw");
  host.remove();
});

test("pinning the first reference repaints its tile even though no number moves; toggling the focused tile keeps its focus", async () => {
  const fixed = {};
  const project = { selected: { id: "s1", references: ["m1", "m2"] }, project: { fixed_references: fixed }, setScene: () => {} };
  const api = { media: async () => ({ media: [{ id: "m1", name: "cat.png", kind: "image" }, { id: "m2", name: "dog.png", kind: "image" }] }) };
  const host = document.createElement("div");
  document.body.append(host);
  let fire;
  hub.setup({ host, app: { project, api, mediaPeek: {}, say: () => {}, on: (cb) => { fire = cb; return () => {}; } } });
  await new Promise((r) => setTimeout(r, 10));
  const cell = (n) => [...host.querySelectorAll(".cx-cell")].find((x) => x.textContent.includes(n));
  fixed.m1 = 1;
  fire("project");
  assert.match(cell("cat.png").textContent, /📌R1/, "pinned: the pin shows");
  cell("dog.png").focus();
  project.selected.references = ["m1"];
  fire("project");
  assert.equal(document.activeElement, cell("dog.png"), "the focused tile, rebuilt, keeps the focus");
  host.remove();
});
