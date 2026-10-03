import test from "node:test";
import assert from "node:assert/strict";
import { videoLane } from "./lanes.js";

test("scenes, a ghost and a pause become clips laid end to end", () => {
  const p = { num_frames_per_scene: 97, frame_rate: 25, scene_ghosts: [{ id: "g", afterSceneId: "a", text: "gone", durationSec: 2 }],
    scenes: [{ id: "a", text: "one\ntwo", gap_after_sec: 1 }, { id: "b", text: "" }] };
  const lane = videoLane(p, ["b"], "b");
  assert.deepEqual(lane.clips.map((c) => c.id), ["a", "ghost:g", "gap:a", "b"]);
  assert.equal(lane.clips[0].title, "one");
  assert.equal(lane.clips[3].selected, true);
  assert.equal(lane.clips[3].start, lane.clips[2].start + 1);
});
