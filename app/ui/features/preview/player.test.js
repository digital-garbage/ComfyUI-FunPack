import test from "node:test";
import assert from "node:assert/strict";
import { timecode, sourceOf } from "./index.js";

test("timecode is hours:minutes:seconds:frames at the project rate", () => {
  assert.equal(timecode(0, 25), "00:00:00:00");
  assert.equal(timecode(3725.5, 25), "01:02:05:12");
});

test("a clip's picture: a render from where the clip starts in it, a bin clip by its ref, nothing when not made yet", () => {
  const open = { scene_renders: { a: { media: { filename: "a.mp4", subfolder: "", type: "output" }, inSec: 1 } } };
  assert.deepEqual(sourceOf({ id: "a", source_in: 0.5 }, open), { url: "/view?filename=a.mp4&subfolder=&type=output", from: 1.5 });
  assert.deepEqual(sourceOf({ id: "v", source: { type: "video", media_ref: "m 1" }, source_in: 2 }, open), { url: "/funpack/api/media/m%201/file", from: 2 });
  assert.equal(sourceOf({ id: "b" }, open), null);
  assert.equal(sourceOf({ id: "v", source: { type: "video" } }, open), null);
});
