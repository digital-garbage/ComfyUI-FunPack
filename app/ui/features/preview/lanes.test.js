import test from "node:test";
import assert from "node:assert/strict";
import { laneSource } from "./lanes.js";

test("a file lane plays from its in-point; a separated lane from its pinned bin file or ComfyUI render; none without a source", () => {
  assert.deepEqual(laneSource({ media_ref: "m", source_in_sec: 1, source_dur: 2 }), { url: "/funpack/api/media/m/file", from: 1, dur: 2 });
  assert.deepEqual(laneSource({ kind: "separated", pinned_bin_ref: "v", pinned_in_sec: 0.5, pinned_dur: 3 }), { url: "/funpack/api/media/v/file", from: 0.5, dur: 3 });
  assert.match(laneSource({ kind: "separated", pinned_media: { filename: "a b.mp4", type: "output" }, pinned_dur: 1 }).url, /^\/view\?filename=a%20b\.mp4&subfolder=&type=output$/);
  assert.equal(laneSource({ kind: "separated" }), null);
  assert.equal(laneSource({}), null);
});
