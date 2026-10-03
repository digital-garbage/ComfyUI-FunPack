import test from "node:test";
import assert from "node:assert/strict";
import { RENDER, runJob } from "./job.js";

test("a job is started and polled under the render module's own routes", async () => {
  const seen = [];
  globalThis.fetch = async (url, init) => {
    seen.push(`${(init && init.method) || "GET"} ${url}`);
    const body = init && init.method === "POST" ? { job_id: "j1" } : { state: "done", media: { filename: "x.mp4" } };
    return { ok: true, status: 200, json: async () => body };
  };
  const real = globalThis.setTimeout;
  globalThis.setTimeout = (fn) => real(fn, 0);
  try {
    const state = await runJob(`${RENDER}/projects/p1/export-clips`, { clips: [] });
    assert.equal(state.state, "done");
    assert.deepEqual(seen, ["POST /funpack/api/m/render/projects/p1/export-clips", "GET /funpack/api/m/render/projects/p1/export-clips/j1"]);
  } finally { globalThis.setTimeout = real; }
});
