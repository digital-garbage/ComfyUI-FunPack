import test from "node:test";
import assert from "node:assert/strict";
import { fxStyle, zoomScale } from "./fx.js";

test("flips, fill, crop and blur become CSS; nothing set means nothing applied", () => {
  assert.deepEqual(fxStyle({}), { transform: "", filter: "", objectFit: "contain", opacity: "" });
  const s = fxStyle({ flip_h: true, fit: "fill", crop_inset: 0.1, blur: 0.5 });
  assert.deepEqual([s.transform, s.objectFit, s.filter], ["scaleX(-1) scale(1.2500)", "cover", "blur(4.0px)"]);
});

test("zoom in ramps from 1 to 1+ratio over its frames and holds; zoom out is the reverse; fades follow the clip's edges", () => {
  const fx = { zoom: "in", zoom_ratio: 0.2, zoom_frames: 10, zoom_start_frame: 5 };
  assert.equal(zoomScale(fx, 0, 4, 25), 1);
  assert.equal(zoomScale(fx, 1, 4, 25), 1.2);
  assert.ok(Math.abs(zoomScale(fx, 0.3, 4, 25) - 1.04) < 1e-9);
  assert.equal(zoomScale({ ...fx, zoom: "out" }, 0, 4, 25), 1.2);
  assert.equal(fxStyle({ fade_in: 1 }, 0.5, 4).opacity, "0.500");
  assert.equal(fxStyle({ fade_out: 1 }, 3.5, 4).opacity, "0.500");
});
