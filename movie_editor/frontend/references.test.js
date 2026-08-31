// Reference numbering (references.js).
// Run: node --test movie_editor/frontend/references.test.js
//
// The rule these guard: the R number shown must be the one a reference slot resolves
// against. builder._reference_by_slot picks the nth mark OF THAT KIND, so a global count
// makes "Reference video 1" point at an item the UI labels R2 — and leaves no way to have
// an R1 video and an R1 image at once.
const test = require("node:test");
const assert = require("node:assert");

const R = require("./references.js");

const BIN = [
  { id: "i1", kind: "image", name: "cat.png" },
  { id: "v1", kind: "video", name: "clip.mp4" },
  { id: "i2", kind: "image", name: "dog.png" },
  { id: "a1", kind: "audio", name: "score.wav" },
];

test("an image and a video marked together are both R1", () => {
  const marks = ["i1", "v1"];
  assert.equal(R.referenceNumber(marks, BIN, "i1"), 1);
  assert.equal(R.referenceNumber(marks, BIN, "v1"), 1);
});

test("marking order still separates two references of the same kind", () => {
  const marks = ["i1", "v1", "i2"];
  assert.equal(R.referenceNumber(marks, BIN, "i1"), 1);
  assert.equal(R.referenceNumber(marks, BIN, "i2"), 2);
  assert.equal(R.referenceNumber(marks, BIN, "v1"), 1);
});

test("marking another kind never renumbers the images", () => {
  const before = R.referenceNumber(["i1", "i2"], BIN, "i2");
  const after = R.referenceNumber(["i1", "a1", "i2"], BIN, "i2");
  assert.equal(before, 2);
  assert.equal(after, 2);
});

test("the number matches the slot that resolves it", () => {
  // builder._reference_by_slot(kind, n) takes the marks of one kind in mark order and
  // returns the nth. Same list, same answer, or the badge is pointing at the wrong file.
  const marks = ["i1", "v1", "i2"];
  const videos = R.referencesOfKind(marks, BIN, "video");
  assert.equal(videos[R.referenceNumber(marks, BIN, "v1") - 1].id, "v1");
  const images = R.referencesOfKind(marks, BIN, "image");
  assert.equal(images[R.referenceNumber(marks, BIN, "i2") - 1].id, "i2");
});

test("unmarked media has no number", () => {
  assert.equal(R.referenceNumber(["i1"], BIN, "v1"), 0);
  assert.equal(R.referenceNumber([], BIN, "i1"), 0);
});

test("a mark left behind by a deleted file is not a reference", () => {
  // The id stays in project.references until something prunes it; it must not occupy a
  // number, or every later reference of that kind counts one too high.
  const marks = ["gone", "i1"];
  assert.equal(R.referenceNumber(marks, BIN, "gone"), 0);
  assert.equal(R.referenceNumber(marks, BIN, "i1"), 1);
  assert.equal(R.referenceCountOfKind(marks, BIN, "image"), 1);
});

test("counts are per kind", () => {
  const marks = ["i1", "v1", "i2"];
  assert.equal(R.referenceCountOfKind(marks, BIN, "image"), 2);
  assert.equal(R.referenceCountOfKind(marks, BIN, "video"), 1);
  assert.equal(R.referenceCountOfKind(marks, BIN, "audio"), 0);
});

test("media with no kind counts as an image, matching the resolver's default", () => {
  const bin = [{ id: "x", name: "mystery" }, { id: "i1", kind: "image", name: "cat.png" }];
  assert.equal(R.referenceNumber(["x", "i1"], bin, "i1"), 2);
});
