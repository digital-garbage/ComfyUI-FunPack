import test from "node:test";
import assert from "node:assert/strict";
import { api } from "../api.js";

test("each file is uploaded on its own, and a refused one is named without sinking the rest", async () => {
  const sent = [];
  globalThis.fetch = async (_url, opts) => {
    const name = opts.body.get("file").name;
    sent.push(name);
    return name === "big.mov"
      ? { ok: false, status: 413, json: async () => { throw new Error("not json"); } }
      : { ok: true, status: 200, json: async () => ({ media: [{ id: name }], problems: [] }) };
  };
  const files = ["a.png", "big.mov", "b.png"].map((n) => new File(["x"], n));
  const out = await api.uploadMedia(files);
  assert.deepEqual(sent, ["a.png", "big.mov", "b.png"]);
  assert.deepEqual(out.media.map((m) => m.id), ["a.png", "b.png"]);
  assert.match(out.problems[0], /big\.mov: HTTP 413/);
});
