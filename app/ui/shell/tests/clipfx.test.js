import test from "node:test";
import assert from "node:assert/strict";
import { applyEffect, applyTransition, tags } from "../clipfx.js";

test("switches toggle, values set, reset clears; unknown ids change nothing", () => {
  const sc = {};
  applyEffect(sc, "flip_h"); applyEffect(sc, "fill_frame"); applyEffect(sc, "blur", 0.3); applyEffect(sc, "crop", 50);
  assert.deepEqual(tags(sc), ["⇄ flip", "fill", "crop 40%", "blur 0.30"]);
  applyEffect(sc, "flip_h"); applyEffect(sc, "fill_frame");
  assert.deepEqual(tags(sc), ["crop 40%", "blur 0.30"]);
  assert.equal(applyEffect(sc, "nope"), false);
  applyEffect(sc, "reset");
  assert.deepEqual(sc.effects, {});
});

test("zoom keeps the ramp settings it already has; a transition names the seam and its frames", () => {
  const sc = { effects: { zoom_ratio: 0.3 } };
  applyEffect(sc, "zoom_in");
  assert.deepEqual([sc.effects.zoom, sc.effects.zoom_ratio, sc.effects.zoom_frames], ["in", 0.3, 25]);
  applyTransition(sc, { id: "x", type: "wipeleft", param: { default: 8 } });
  assert.deepEqual([sc.video_transition, sc.transition_frames, tags(sc).pop()], ["wipeleft", 8, "→ wipeleft"]);
});
