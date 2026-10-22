import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../composer/tests/_dom.js";

const tick = (ms = 30) => new Promise((r) => setTimeout(r, ms));

test("Restart ComfyUI says it is restarting once, not twice", async () => {
  setupDom();
  const { createMaintenance } = await import("./maintenance.js");
  // The server answers that it is restarting, then stays down.
  const api = { git: async () => ({ ok: true, restarting: true }), health: async () => { throw new Error("down"); } };
  const m = createMaintenance({ api, flush: async () => {} });
  m.restart();
  await tick();
  [...document.querySelectorAll("button")].find((b) => /Go ahead/.test(b.textContent)).click();
  await tick(100);
  const said = (document.body.textContent.match(/Restarting ComfyUI…/g) || []).length;
  teardownDom();
  assert.equal(said, 1, "the working card repeats the restarting line");
});
test.after(() => setTimeout(() => process.exit(0), 10));
