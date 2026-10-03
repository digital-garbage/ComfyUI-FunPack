import test from "node:test";
import assert from "node:assert/strict";
import { timecode, sourceOf } from "./index.js";

test("timecode is hours:minutes:seconds:frames at the project rate", () => {
  assert.equal(timecode(0, 25), "00:00:00:00");
  assert.equal(timecode(3725.5, 25), "01:02:05:12");
});

test("a clip's picture: a generated clip comes from the server's trimmed segment, a bin clip by its ref, nothing when not made yet", () => {
  const open = { id: "p 1", scene_renders: { a: { media: { filename: "a.mp4", subfolder: "", type: "output" }, inSec: 1 } } };
  const got = sourceOf({ id: "a", source_in: 0.5, effects: { reverse: true } }, open, 3);
  assert.equal(got.from, 0);
  const u = new URL(got.url, "http://x");
  assert.equal(u.pathname, "/funpack/api/m/render/projects/p%201/preview-segment/a");
  assert.deepEqual(Object.fromEntries(u.searchParams), { filename: "a.mp4", subfolder: "", type: "output", render_in: "1", src_in: "0.5", dur: "3", rev: "1" });
  assert.deepEqual(sourceOf({ id: "v", source: { type: "video", media_ref: "m 1" }, source_in: 2 }, open, 3), { url: "/funpack/api/media/m%201/file", from: 2 });
  assert.equal(sourceOf({ id: "b" }, open, 3), null);
  assert.equal(sourceOf({ id: "v", source: { type: "video" } }, open, 3), null);
});
