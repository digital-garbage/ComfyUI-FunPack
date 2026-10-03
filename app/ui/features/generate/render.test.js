import test from "node:test";
import assert from "node:assert/strict";
import { renderFor } from "./index.js";
import { split } from "../../shell/edits.js";

test("both halves of a split unit get the one clip, the second from its cut", () => {
  const p = { num_frames_per_scene: 97, frame_rate: 24, scenes: [{ id: "a", text: "cat" }] };
  split(p, "a");
  const [a, b] = p.scenes;
  const media = { filename: "x.mp4" };
  assert.equal(renderFor(a, p, media, 4).inSec, 0);
  assert.ok(renderFor(b, p, media, 4).inSec > 0);
  assert.equal(b.gen_unit_id, a.gen_unit_id);
});
