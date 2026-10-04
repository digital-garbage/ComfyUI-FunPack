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
  [...document.querySelectorAll(".cx-menu-item")].find((b) => b.textContent.includes("resolution source")).click();
  assert.deepEqual(set, [["s1", "source_image", "m1"]]);
});

