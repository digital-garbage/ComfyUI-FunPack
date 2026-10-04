import test from "node:test";
import assert from "node:assert/strict";
import { swap, upscale } from "./job.js";

const m = (f, sub = "") => ({ filename: f, subfolder: sub, type: "output" });

test("swap replaces the file under every render and ghost that plays it, and nothing else", () => {
  const p = { scene_renders: { a: { media: m("x.mp4"), inSec: 1 }, b: { media: m("x.mp4") }, c: { media: m("y.mp4") } }, scene_ghosts: [{ id: "g", media: m("x.mp4") }] };
  assert.equal(swap(p, m("x.mp4"), m("up.mp4", "funpack_upscaled")), 3);
  assert.equal(p.scene_renders.a.media.filename, "up.mp4"); assert.equal(p.scene_renders.a.inSec, 1);
  assert.equal(p.scene_renders.c.media.filename, "y.mp4");
});

test("upscale waits while queued, then returns the video; an error is thrown in words", async () => {
  const seq = [null, null, { videos: [m("up.mp4", "funpack_upscaled")] }];
  const api = { queueUpscale: async () => "id", upscaleResult: async () => seq.shift() };
  assert.equal((await upscale(api, m("x.mp4"), "mdl", async () => {})).filename, "up.mp4");
  const bad = { queueUpscale: async () => "id", upscaleResult: async () => ({ error: "out of memory" }) };
  await assert.rejects(upscale(bad, m("x.mp4"), "mdl", async () => {}), /out of memory/);
});

test("a job that vanishes is reported after a few looks, a dropped reply is retried longer", async () => {
  const gone = { queueUpscale: async () => "id", upscaleResult: async () => ({ gone: true }) };
  await assert.rejects(upscale(gone, m("x.mp4"), "mdl", async () => {}), /left ComfyUI's queue/);
  let n = 0;
  const flaky = { queueUpscale: async () => "id", upscaleResult: async () => (++n < 5 ? { retry: true } : { videos: [m("u.mp4")] }) };
  assert.equal((await upscale(flaky, m("x.mp4"), "mdl", async () => {})).filename, "u.mp4");
});

test("a take that plays the old file gets the upscaled one too, so stepping back does not undo it", async () => {
  const { swap } = await import("./job.js");
  const old = { filename: "a.mp4", subfolder: "", type: "output" }, up = { filename: "funpack_upscaled_a.mp4", subfolder: "", type: "output" };
  const p = { scene_renders: { s: { media: old } }, scene_variants: { s: [{ media: old, promptId: "p" }, { media: { filename: "other.mp4" }, promptId: "q" }] } };
  swap(p, old, up);
  assert.equal(p.scene_variants.s[0].media.filename, "funpack_upscaled_a.mp4");
  assert.equal(p.scene_variants.s[1].media.filename, "other.mp4");
});
