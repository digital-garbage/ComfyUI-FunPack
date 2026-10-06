import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let hub;
test.before(async () => {
  setupDom();
  Object.assign(globalThis, { HTMLTextAreaElement: window.HTMLTextAreaElement, requestAnimationFrame: window.requestAnimationFrame.bind(window), cancelAnimationFrame: window.cancelAnimationFrame.bind(window) });
  hub = (await import("./index.js")).default;
});
test.after(() => teardownDom());

test("the 💡 stays beside its prompt box when the box's window is moved", async () => {
  const app = { project: { pref: () => true }, api: {}, selection: {}, editorSettings: [] };
  const off = hub.setup({ host: document.createElement("div"), app });
  const box = document.createElement("textarea");
  box.setAttribute("aria-label", "Story");
  document.body.append(box);
  let left = 100;
  box.getBoundingClientRect = () => ({ left, right: left + 300, top: 200, bottom: 300, width: 300, height: 100 });
  box.focus();
  await new Promise((r) => setTimeout(r, 50));
  const bulb = [...document.body.querySelectorAll("button")].find((b) => b.textContent.includes("💡"));
  assert.equal(bulb.style.left, "360px");
  left = 500;                                      // the Composer window dragged: no page event says so
  await new Promise((r) => setTimeout(r, 80));
  assert.equal(bulb.style.left, "760px");
  off();
});
