import test from "node:test";
import assert from "node:assert/strict";
import hub from "./index.js";

const media = { filename: "a.mp4", subfolder: "", type: "output" };
const slots = [{ id: "7", roles: [{ at: "assets.source_image", input: "media_id" }] }];
const project = (extra = {}) => ({ id: "pj", num_frames_per_scene: 97, frame_rate: 25, scenes: [{ id: "a" }, { id: "b", source: { type: "carry" } }], scene_renders: { a: { media, inSec: 0 } }, ...extra });
const make = (values, lastFrame) => { const app = { api: { lastFrame }, pipeline: { currentValues: () => ({ continuity: values }) }, inputHooks: [], mediaMenu: [], project: { project: null, edit: () => true } }; hub.setup({ app }); return app.inputHooks[0]; };

test("a carry shot gets the previous clip's last frame as its start picture", async () => {
  const calls = [];
  const hook = make({ carry: true, dark_guard: true }, async (pid, body) => { calls.push([pid, body.scene_id, body.dur]); return { media_id: "m9", dark: false }; });
  const p = project();
  assert.deepEqual((await hook({ project: p, scene: p.scenes[1], slots })).inputs, { 7: { media_id: "m9" } });
  assert.deepEqual([calls[0][0], calls[0][1], Math.round(calls[0][2] * 25)], ["pj", "a", 97]);
});

test("nothing happens when it is off, the shot has its own picture, the source is another kind, or the pipeline has no start picture", async () => {
  const hook = make({ carry: false }, async () => assert.fail("must not ask"));
  const p = project();
  assert.deepEqual(await hook({ project: p, scene: p.scenes[1], slots }), { inputs: {}, notes: [] });
  const on = make({}, async () => assert.fail("must not ask"));
  p.scenes[1].source_image = "mine";
  assert.deepEqual(await on({ project: p, scene: p.scenes[1], slots }), { inputs: {}, notes: [] });
  p.scenes[1].source_image = "";
  p.scenes[1].source = { type: "image" };
  assert.deepEqual(await on({ project: p, scene: p.scenes[1], slots }), { inputs: {}, notes: [] });
  p.scenes[1].source = { type: "carry" };
  assert.deepEqual(await on({ project: p, scene: p.scenes[1], slots: [] }), { inputs: {}, notes: [] });
});

test("a dark last frame, a failed extraction and a missing predecessor each say why the shot starts without a picture", async () => {
  const p = project();
  const dark = make({ dark_guard: true }, async () => ({ media_id: "m", dark: true }));
  assert.match((await dark({ project: p, scene: p.scenes[1], slots })).notes[0], /fade to black/);
  const allowed = make({ dark_guard: false }, async () => ({ media_id: "m", dark: true }));
  assert.equal((await allowed({ project: p, scene: p.scenes[1], slots })).inputs[7].media_id, "m");
  const broken = make({}, async () => { throw new Error("no ffmpeg"); });
  assert.match((await broken({ project: p, scene: p.scenes[1], slots })).notes[0], /no ffmpeg/);
  assert.match((await broken({ project: p, scene: p.scenes[0], slots })).notes[0], /first clip/);
});

test("an identity picture rides along as the last reference of every scene, and says so when the pipeline has no room for it", async () => {
  const refSlots = [...slots, { id: "9", roles: [{ at: "assets.reference_1", input: "media_id" }] }];
  const hook = make({ carry: false }, async () => assert.fail("must not ask"));
  const p = project({ continuity_settings: { identity_pin_ref: "face" } });
  assert.deepEqual((await hook({ project: p, scene: p.scenes[1], slots: refSlots })).inputs, { 9: { media_id: "face" } });
  assert.match((await hook({ project: p, scene: p.scenes[1], slots })).notes[0], /no free reference input/);
  p.scenes[1].references = ["face"];
  assert.deepEqual((await hook({ project: p, scene: p.scenes[1], slots: refSlots })).inputs, {});       // already one of the scene's own
});
