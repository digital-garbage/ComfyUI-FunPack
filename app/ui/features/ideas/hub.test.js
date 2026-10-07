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

test("the 💡 sits inside its prompt box's bottom-right corner, and follows the box when its window is moved", async () => {
  const app = { project: { pref: () => true }, api: {}, selection: {}, editorSettings: [] };
  const off = hub.setup({ host: document.createElement("div"), app });
  try {
    const box = document.createElement("textarea");
    box.setAttribute("aria-label", "Story");
    document.body.append(box);
    let left = 100;
    box.getBoundingClientRect = () => ({ left, right: left + 300, top: 200, bottom: 300, width: 300, height: 100 });
    box.focus();
    await new Promise((r) => setTimeout(r, 50));
    const bulb = [...document.body.querySelectorAll("button")].find((b) => b.textContent.includes("💡"));
    bulb.getBoundingClientRect = () => ({ width: 24, height: 20 });
    await new Promise((r) => setTimeout(r, 50));
    assert.deepEqual([bulb.style.left, bulb.style.top], ["360px", "276px"], "inside the box, left of its resize grip, above its bottom edge");
    left = 500;                                      // the Composer window dragged: no page event says so
    await new Promise((r) => setTimeout(r, 80));
    assert.equal(bulb.style.left, "760px");
  } finally { off(); }
});
