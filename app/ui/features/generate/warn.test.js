import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let warn;
test.before(async () => { setupDom(); warn = (await import("./warn.js")).default; });
test.after(() => teardownDom());

const chip = (host) => [...host.querySelectorAll("button")].find((b) => /No Taste key/.test(b.textContent));

test("the No Taste key chip shows only while a learning feature is on and the key is empty", () => {
  const taste = { id: "taste" };
  let values = { taste: { key: "" } }, learning = true, listener;
  const pipeline = { slots: () => [], allOff: () => false, modulesById: () => ({ taste }), activeModules: () => [taste],
    useful: (m) => m === taste && learning, currentValues: () => values, subscribe: (fn) => { listener = fn; return () => {}; } };
  const host = document.createElement("div");
  warn.setup({ host, app: { pipeline } });
  assert.equal(chip(host).hidden, false, "learning on, no key: said");
  values = { taste: { key: "  portraits " } }; listener();
  assert.equal(chip(host).hidden, true, "a key is set");
  values = { taste: { key: "" } }; learning = false; listener();
  assert.equal(chip(host).hidden, true, "nothing learns, so a missing key changes nothing");
});
