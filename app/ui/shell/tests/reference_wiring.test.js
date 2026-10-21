import test from "node:test";
import assert from "node:assert/strict";

import { referenceNumbers, routeFor, sceneReferences, wireReferences } from "../reference_wiring.js";

function refSlot(n) {
  return { id: `ref_source_${n}`, roles: [{ at: `assets.reference_${n}`,
    input: "media_id", wireTo: { slot: "r2v", input: `ref_images.ref_image_${n - 1}` } }] };
}

test("each picked reference is wired to its slot and to r2v", () => {
  const slots = [refSlot(1), refSlot(2)];
  const { overrides, unwired } = wireReferences(["aaa111111111", "bbb222222222"], slots);
  assert.equal(unwired, 0);
  assert.equal(overrides.ref_source_1.media_id, "aaa111111111");
  assert.deepEqual(overrides.r2v["ref_images.ref_image_0"], ["ref_source_1", 0]);
  assert.deepEqual(overrides.r2v["ref_images.ref_image_1"], ["ref_source_2", 0]);
});

test("a reference beyond the pipeline's slots is counted, not silently dropped", () => {
  // This is the bug an adversarial review caught: a scene can carry more
  // references than the CURRENT pipeline offers slots for (e.g. added under
  // a bigger preset, then the preset was swapped for a smaller one). The old
  // code in boot.js just returned from the forEach iteration and moved on --
  // no override, no count, no way for the caller to tell the user anything
  // was left out.
  const slots = [refSlot(1)]; // only one slot offered
  const { overrides, unwired } = wireReferences(
    ["aaa111111111", "bbb222222222", "ccc333333333"], slots);
  assert.equal(unwired, 2);
  assert.equal(overrides.ref_source_1.media_id, "aaa111111111");
  assert.equal(Object.keys(overrides).length, 2); // ref_source_1 + r2v, nothing else
});

test("no references and no slots wires nothing and reports no shortfall", () => {
  const { overrides, unwired } = wireReferences([], []);
  assert.deepEqual(overrides, {});
  assert.equal(unwired, 0);
});

test("slots not yet loaded (null, boot.js's own initial state) is treated as no slots, not a crash", () => {
  const { overrides, unwired } = wireReferences(["aaa111111111"], null);
  assert.deepEqual(overrides, {});
  assert.equal(unwired, 1);
});

test("a slot whose roles claim two reference numbers does not let the second silently overwrite the first", () => {
  // Not producible by the shipped H3 preset (one role per slot), but nothing
  // stops a future preset's roles array from doing this -- the function's own
  // contract ("no reference silently dropped") has to hold for the input
  // shape, not just the one preset that happens to exist today.
  const combo = { id: "combo_slot", roles: [
    { at: "assets.reference_1", wireTo: { slot: "r2v", input: "ref_images.ref_image_0" } },
    { at: "assets.reference_2", wireTo: { slot: "r2v", input: "ref_images.ref_image_1" } },
  ] };
  const { overrides, unwired } = wireReferences(["AAA", "BBB"], [combo]);
  assert.equal(overrides.combo_slot.media_id, "AAA");
  assert.equal(unwired, 1);
});

test("two different slots wired to the same downstream input do not let the second overwrite the first", () => {
  // Same failure as the combo-slot case above, reached a different way: two
  // separate reference slots whose `wireTo` both point at one destination
  // (e.g. a copy-pasted preset). The destination can hold one value.
  const slots = [
    { id: "ref_source_1", roles: [{ at: "assets.reference_1",
      wireTo: { slot: "r2v", input: "ref_images.ref_image_0" } }] },
    { id: "ref_source_2", roles: [{ at: "assets.reference_2",
      wireTo: { slot: "r2v", input: "ref_images.ref_image_0" } }] },
  ];
  const { overrides, unwired } = wireReferences(["AAA", "BBB"], slots);
  assert.deepEqual(overrides.r2v["ref_images.ref_image_0"], ["ref_source_1", 0]);
  assert.equal(overrides.ref_source_1.media_id, "AAA");
  assert.equal(overrides.ref_source_2, undefined);
  assert.equal(unwired, 1);
});

test("a slot the caller already assigned (e.g. the source-image slot) is not silently overwritten by a reference", () => {
  // boot.js sets the source-image slot's media_id BEFORE calling
  // wireReferences() -- if a preset ever reused that same slot id for a
  // reference role too, the reference would clobber the source image with
  // no toast, since wireReferences() has no visibility into that earlier
  // write on its own. The caller passes it in as reserved instead.
  const slots = [refSlot(1)];
  const { overrides, unwired } = wireReferences(
    ["AAA"], slots, ["ref_source_1"]);
  assert.equal(overrides.ref_source_1, undefined);
  assert.equal(unwired, 1);
});

test("with no pins the references take numbers 1, 2, 3 in the order they were added", () => {
  assert.deepEqual(referenceNumbers(["a", "b", "c"], {}), [1, 2, 3]);
  assert.deepEqual(referenceNumbers(["a", "b"], undefined), [1, 2]);
});

test("switching off R1 leaves a pinned R2 as R2, not R1", () => {
  // The artstyle case: R1 is off (removed from the list), R2 is pinned to slot 2.
  assert.deepEqual(referenceNumbers(["b"], { b: 2 }), [2]);
  // Pinned references keep their numbers; the unpinned one takes the lowest free number.
  assert.deepEqual(referenceNumbers(["a", "b"], { b: 2 }), [1, 2]);
  assert.deepEqual(referenceNumbers(["a", "b", "c"], { c: 1 }), [2, 3, 1]);
});

test("a pin that is taken, out of range or not a whole number is ignored", () => {
  assert.deepEqual(referenceNumbers(["a", "b"], { a: 1, b: 1 }), [1, 2]);
  assert.deepEqual(referenceNumbers(["a", "b"], { a: 0, b: 17 }), [1, 2]);
  assert.deepEqual(referenceNumbers(["a", "b"], { a: 1.5, b: "2" }), [1, 2]);
});

test("pinned references are wired to their own slot number, not their list position", () => {
  const slot = (n) => ({ id: `ref_source_${n}`, roles: [{ at: `assets.reference_${n}`,
    input: "media_id", wireTo: { slot: "r2v", input: `ref_images.ref_image_${n - 1}` } }] });
  const slots = [slot(1), slot(2)];
  const { overrides, unwired } = wireReferences(["bbb222222222"], slots, [], { bbb222222222: 2 });
  assert.equal(unwired, 0);
  assert.equal(overrides.ref_source_2.media_id, "bbb222222222");
  assert.equal(overrides.ref_source_1, undefined);
  assert.deepEqual(overrides.r2v["ref_images.ref_image_1"], ["ref_source_2", 0]);
});

test("a fixed reference rides in every scene at its own number, ahead of the scene's own, and wins a clash", () => {
  const plan = sceneReferences({ references: ["x", "fx"] }, { fx: 2 });
  assert.deepEqual(plan.ids, ["fx", "x"]);
  assert.deepEqual(referenceNumbers(plan.ids, plan.pins), [2, 1]);
  assert.deepEqual(sceneReferences({ references: ["a"] }, undefined), { ids: ["a"], pins: {} });
});

// R2V as core describes it: Autogrow groups spelled out, types per input; the loader's outputs by type.
const R2V = { inputs: { prompt: "STRING", "ref_images.ref_image_0": "IMAGE", "ref_images.ref_image_1": "IMAGE",
  "ref_videos.ref_video_0": "IMAGE", "ref_videos.ref_video_1": "IMAGE", "ref_videos.ref_video_10": "IMAGE",
  "ref_video_audios.ref_video_audio_0": "AUDIO", "ref_video_audios.ref_video_audio_1": "AUDIO", "ref_audios.ref_audio_0": "AUDIO", "ref_audios.ref_audio_1": "AUDIO" } };
const LOADER = { outputs: ["IMAGE", "IMAGE", "AUDIO"] };

test("routeFor reads a video's and a sound's inputs off the node, not off the preset", () => {
  assert.deepEqual(routeFor("video", R2V, LOADER, "r2v"),
    { slot: "r2v", names: ["ref_videos.ref_video_0", "ref_videos.ref_video_1", "ref_videos.ref_video_10"], output: 1 });
  assert.deepEqual(routeFor("audio", R2V, LOADER, "r2v"),
    { slot: "r2v", names: ["ref_audios.ref_audio_0", "ref_audios.ref_audio_1"], output: 2 });
  assert.equal(routeFor("video", { inputs: { a: "IMAGE" } }, LOADER, "r2v"), null);
  assert.equal(routeFor("video", R2V, { outputs: ["IMAGE"] }, "r2v"), null);
});

test("a preset with no routing data still sends a video and a sound to R2V's own inputs, numbered per kind", () => {
  const slot = (n) => ({ id: `ref_source_${n}`, node: "Loader", roles: [{ at: `assets.reference_${n}`, input: "media_id",
    wireTo: { slot: "r2v", input: `ref_images.ref_image_${n - 1}` } }] });
  const kinds = { pic: "image", vid: "video", vid2: "video", snd: "audio" };
  const routeOf = (k, s, role) => routeFor(k, R2V, LOADER, role.wireTo.slot);
  const { overrides, unwired } = wireReferences(["vid", "pic", "snd", "vid2"], [slot(1), slot(2), slot(3), slot(4)], [], {}, (id) => kinds[id], routeOf);
  assert.equal(unwired, 0);
  assert.deepEqual(overrides.r2v["ref_videos.ref_video_0"], ["ref_source_1", 1]);
  assert.deepEqual(overrides.r2v["ref_images.ref_image_0"], ["ref_source_2", 0]);      // pictures are numbered among pictures
  assert.deepEqual(overrides.r2v["ref_audios.ref_audio_0"], ["ref_source_3", 2]);
  assert.deepEqual(overrides.r2v["ref_videos.ref_video_1"], ["ref_source_4", 1]);
  assert.equal(overrides.ref_source_1.full, true);
  assert.equal(overrides.ref_source_2.full, undefined);
});

test("a pipeline with no place for a video says so, and so does one that has run out of them", () => {
  const slots = [refSlot(1), refSlot(2)];
  let r = wireReferences(["vid"], slots, [], {}, () => "video", () => null);
  assert.equal(r.unwired, 1);
  assert.match(r.left[0].why, /no input for a video/);
  r = wireReferences(["a", "b"], slots, [], {}, () => "video", () => ({ slot: "r2v", names: ["only_video_0"], output: 1 }));
  assert.equal(r.unwired, 1);
  assert.match(r.left[0].why, /takes only 1 video/);
});

test("a video with a soundtrack also feeds the audio input paired with its number; a silent one does not", () => {
  const slot = (n) => ({ id: `ref_source_${n}`, node: "Loader", roles: [{ at: `assets.reference_${n}`, input: "media_id", wireTo: { slot: "r2v", input: `ref_images.ref_image_${n - 1}` } }] });
  const kinds = { loud: "video", quiet: "video" };
  const routeOf = (k, s, role) => routeFor(k, R2V, LOADER, role.wireTo.slot);
  const { overrides } = wireReferences(["quiet", "loud"], [slot(1), slot(2)], [], {}, (id) => kinds[id], routeOf, (id) => id === "loud");
  assert.deepEqual(overrides.r2v["ref_videos.ref_video_0"], ["ref_source_1", 1]);
  assert.deepEqual(overrides.r2v["ref_videos.ref_video_1"], ["ref_source_2", 1]);
  assert.equal(overrides.r2v["ref_video_audios.ref_video_audio_0"], undefined);
  assert.deepEqual(overrides.r2v["ref_video_audios.ref_video_audio_1"], ["ref_source_2", 2]);
});

test("fifteen references of mixed kinds all find an input: 9 pictures, 3 videos with sound, 3 sounds", () => {
  const many = { inputs: { ...Object.fromEntries([...Array(9).keys()].map((i) => [`ref_images.ref_image_${i}`, "IMAGE"])),
    ...Object.fromEntries([...Array(3).keys()].flatMap((i) => [[`ref_videos.ref_video_${i}`, "IMAGE"], [`ref_video_audios.ref_video_audio_${i}`, "AUDIO"], [`ref_audios.ref_audio_${i}`, "AUDIO"]])) } };
  const ids = [...Array(9).keys()].map((i) => `p${i}`).concat(["v0", "v1", "v2", "a0", "a1", "a2"]);
  const kind = (id) => ({ p: "image", v: "video", a: "audio" })[id[0]];
  const slots = ids.map((_, i) => ({ id: `s${i + 1}`, node: "Loader", roles: [{ at: `assets.reference_${i + 1}`, input: "media_id", wireTo: { slot: "r2v", input: `ref_images.ref_image_${i}` } }] }));
  const { unwired, overrides } = wireReferences(ids, slots, [], {}, kind, (k, s, role) => routeFor(k, many, LOADER, role.wireTo.slot), (id) => id[0] === "v");
  assert.equal(unwired, 0);
  assert.deepEqual(overrides.r2v["ref_images.ref_image_8"], ["s9", 0]);
  assert.deepEqual(overrides.r2v["ref_video_audios.ref_video_audio_2"], ["s12", 2]);
  assert.deepEqual(overrides.r2v["ref_audios.ref_audio_2"], ["s15", 2]);
});
