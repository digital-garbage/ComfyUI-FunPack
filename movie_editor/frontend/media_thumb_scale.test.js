// _thumbScale: the pure downscale-to-fit math behind media bin thumbnails (mediabrowser.js).
// Extracted rather than imported since mediabrowser.js is a plain IIFE with no exports (it
// binds to the live DOM/Store/API globals) -- see block_sweep_config.test.js for the same
// pattern against store.js.
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");

const src = fs.readFileSync(__dirname + "/mediabrowser.js", "utf8");
const grab = (name) => src.match(new RegExp(`function ${name}[\\s\\S]*?\\n  \\}\\n`))[0];
const { _thumbScale } = new Function(grab("_thumbScale") + "; return { _thumbScale };")();

test("already-small media is left at native size, not upscaled", () => {
  assert.deepStrictEqual(_thumbScale(64, 32, 320), { w: 64, h: 32 });
  assert.deepStrictEqual(_thumbScale(320, 320, 320), { w: 320, h: 320 });
});

test("oversized media is scaled down so the long edge hits the cap", () => {
  assert.deepStrictEqual(_thumbScale(3840, 2160, 320), { w: 320, h: 180 });
});

test("a portrait image scales by its long edge (height), not width", () => {
  assert.deepStrictEqual(_thumbScale(1080, 1920, 320), { w: 180, h: 320 });
});

test("degenerate zero/negative dimensions are floored to 1, never zero or NaN", () => {
  const { w, h } = _thumbScale(0, 0, 320);
  assert.ok(w >= 1 && h >= 1 && Number.isFinite(w) && Number.isFinite(h));
});
