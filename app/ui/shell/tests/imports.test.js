import test from "node:test";
import assert from "node:assert/strict";
import { addVideoClip, addAudioTrack } from "../imports.js";

test("a bin video becomes a clip as long as the file; one of unknown length follows the project until it is probed", () => {
  const p = { scenes: [], frame_rate: 25 };
  const known = addVideoClip(p, { id: "m1", kind: "video" }, 4);
  assert.deepEqual([known.source.media_ref, known.frames_mode, known.frames], ["m1", "timeline", 97]);
  const unknown = addVideoClip(p, { id: "m2", kind: "video" }, 0);
  assert.deepEqual([unknown.frames_mode, unknown.source_dur], ["project", null]);
  assert.equal(addVideoClip(p, { id: "i", kind: "image" }), false);
  assert.equal(p.scenes.length, 2);
});

test("a bin sound becomes a lane from the playhead; only audio is accepted", () => {
  const p = {};
  const t = addAudioTrack(p, { id: "a", kind: "audio", name: "tone" }, -3, 2);
  assert.deepEqual([t.start_sec, t.source_dur, t.label, p.audio_tracks.length], [0, 2, "tone", 1]);
  assert.equal(addAudioTrack(p, { id: "v", kind: "video" }, 0), false);
});
