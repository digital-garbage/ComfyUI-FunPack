import test from "node:test";
import assert from "node:assert/strict";
import { applyStory, clash, joinStory, match, storyRoots } from "./story.js";

const proj = () => ({ scenes: [{ id: "a", text: "one" }, { id: "b", text: "two" }, { id: "v", text: "", source: { type: "video", media_ref: "m" } }, { id: "c", text: "three" }], scene_renders: {}, scene_ghosts: [] });
const texts = (p) => p.scenes.map((s) => `${s.id}:${s.text}`);

test("the story is the generated roots' texts with the marker between them; video clips and left-out scenes are not in it", () => {
  const p = proj(); p.scenes[3].excluded = true;
  assert.deepEqual(storyRoots(p).map((s) => s.id), ["a", "b"]);
  assert.equal(joinStory(p, "qcut"), "one\nqcut\ntwo");
});

test("match: exact text pins a scene through a reorder; the rest take the leftovers in order", () => {
  assert.deepEqual(match(["three", "one", "new"], ["one", "two", "three"]), [2, 0, 1]);
  assert.deepEqual(match(["x", "two"], ["one", "two"]), [0, 1]);
  assert.deepEqual(match(["a", "b", "c"], ["a"]), [0, -1, -1]);
});

test("editing a scene's words keeps the scene (its id, so its render and anchor)", () => {
  const p = proj();
  assert.equal(applyStory(p, ["one!", "two", "three"]), true);
  assert.deepEqual(texts(p), ["a:one!", "b:two", "v:", "c:three"]);
});

test("a dropped part removes its scene (a rendered one leaves a ghost); the video clip stays", () => {
  const p = proj(); p.scene_renders.b = { media: { filename: "b.mp4" } };
  assert.equal(applyStory(p, ["one", "three"]), true);
  assert.deepEqual(texts(p), ["a:one", "v:", "c:three"]);
  assert.deepEqual(p.scene_ghosts.map((g) => g.id), ["b"]);
});

test("a part nothing owns becomes a new scene at the end of the story", () => {
  const p = proj();
  assert.equal(applyStory(p, ["one", "two", "three", "four"]), true);
  assert.deepEqual(p.scenes.map((s) => s.text), ["one", "two", "", "three", "four"]);
});

test("a part with new words takes over the leftover scene in place, so its render and anchor are kept", () => {
  const p = proj();
  applyStory(p, ["one", "extra", "three"]);
  assert.deepEqual(texts(p), ["a:one", "b:extra", "v:", "c:three"]);
});

test("a reordered story reorders the plan; an unchanged story changes nothing", () => {
  const p = proj();
  assert.equal(applyStory(p, ["three", "one", "two"]), true);
  assert.deepEqual(p.scenes.map((s) => s.id), ["c", "a", "v", "b"]);       // the video clip keeps its slot
  const q = proj();
  assert.equal(applyStory(q, ["one", "two", "three"]), false);
});

test("an empty split changes nothing", () => assert.equal(applyStory(proj(), []), false));

test("an emptied box does not delete the scenes", () => {
  const p = proj(), before = JSON.stringify(p);
  assert.equal(applyStory(p, [""]), false);
  assert.equal(JSON.stringify(p), before);
});

test("editing one scene leaves the plan order alone, even with a montage or a cut unit interleaved", () => {
  const p = { scenes: [{ id: "A", text: "alpha" }, { id: "B", text: "beta" }, { id: "a1", gen_unit_id: "A", cut_offset_frames: 40, text: "" }, { id: "b1", gen_unit_id: "B", cut_offset_frames: 40, text: "" }, { id: "a2", gen_unit_id: "A", cut_offset_frames: 0, text: "alpha" }], scene_renders: {}, scene_ghosts: [] };
  applyStory(p, ["alpha!", "beta"]);
  assert.deepEqual(p.scenes.map((s) => s.id), ["A", "B", "a1", "b1", "a2"]);
  assert.equal(p.scenes.find((s) => s.id === "a2").text, "alpha!", "the first piece shares the prompt");
});

test("dropping a cut scene's paragraph removes the scene with its cuts instead of promoting a cut", () => {
  const p = { scenes: [{ id: "A", text: "alpha" }, { id: "A2", gen_unit_id: "A", cut_offset_frames: 40, text: "" }, { id: "B", text: "beta" }], scene_renders: {}, scene_ghosts: [] };
  assert.equal(applyStory(p, ["beta"]), true);
  assert.deepEqual(p.scenes.map((s) => s.id), ["B"]);
});

test("clash finds a scene whose own text contains a cut word", () => {
  const p = proj(); p.scenes[1].text = "a quick qcut to b";
  assert.equal(clash(p, ["qcut"]).id, "b");
  assert.equal(clash(p, ["other"]), null);
});
