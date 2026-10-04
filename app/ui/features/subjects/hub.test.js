import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { buildInputs } from "../generate/inputs.js";

let hub, bin;
test.before(async () => { setupDom(); hub = (await import("./index.js")).default; bin = await import("../../shell/bin.js"); });
test.after(() => teardownDom());

test("the picked references' subject lines come first, in the order they were picked, and leave with the reference", async () => {
  bin.learn([{ id: "a", name: "A", subject: "<Subject 1> is the woman in <Picture 1>." }, { id: "b", name: "B" }, { id: "c", name: "C", subject: "<Subject 2> is the robot." }]);
  const app = { api: { setMediaSubject: async () => ({}) }, mediaMenu: [], promptPrefix: [], say() {} };
  const off = hub.setup({ host: document.createElement("div"), app });
  const [prefix] = app.promptPrefix;
  assert.deepEqual(prefix({ references: ["c", "b", "a"] }), ["<Subject 2> is the robot.", "<Subject 1> is the woman in <Picture 1>."]);
  assert.deepEqual(prefix({ references: ["b"] }), []);
  assert.deepEqual(prefix({ references: [] }), []);
  assert.equal(app.mediaMenu[0]({ id: "a" }, { kind: "image" })[0].label, "Edit subject text…");
  assert.equal(app.mediaMenu[0]({ id: "b" }, { kind: "image" })[0].label, "Subject text…");
  assert.deepEqual(app.mediaMenu[0]({ id: "x" }, { kind: "audio" }), []);
  off();
  assert.equal(app.promptPrefix.length + app.mediaMenu.length, 0);
});

test("the lines go in front of the anchor of the text that is sent to be expanded", async () => {
  let sent;
  const slots = [{ id: "p", roles: [{ at: "generation.prompt", input: "text" }] }];
  await buildInputs({ project: { anchor: "cinematic", video: {} }, scene: { text: "she walks" }, slots, frames: 9,
    prefix: ["<Subject 1> is the woman."], expand: async (b) => { sent = b; return { text: "x" }; } });
  assert.equal(sent.anchor, "<Subject 1> is the woman. cinematic");
  await buildInputs({ project: { anchor: "", video: {} }, scene: { text: "t" }, slots, frames: 9, prefix: [], expand: async (b) => { sent = b; return { text: "x" }; } });
  assert.equal(sent.anchor, "");
});
