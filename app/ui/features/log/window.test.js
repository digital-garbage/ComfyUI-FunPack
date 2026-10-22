import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let mount;
test.before(async () => {
  setupDom();
  ({ mount } = await import("./window.js"));
});
test.after(() => teardownDom());

const dom = { selectingIn: () => false };
const later = () => new Promise((r) => setTimeout(r, 20));

function serve(payload) {
  globalThis.fetch = async () => ({ json: async () => payload });
}

test("only the lines that need a look are coloured; a healthy log is all plain", async () => {
  serve({
    lines: ["got prompt", "[Log] Warning: X | slow", "ValueError: nope", "Prompt executed"],
    tones: ["plain", "warn", "error", "plain"], path: "p",
  });
  const view = mount({ dom, poll: 1e9 });
  await later();
  const spans = [...view.node.querySelectorAll(".cx-code-line")];
  assert.deepEqual(spans.map((s) => s.className.replace("cx-code-line", "").trim()),
    ["", "cx-code-warn", "cx-code-error", ""]);
  view.destroy();

  serve({ lines: ["got prompt", "done"], tones: ["plain", "plain"], path: "p" });
  const calm = mount({ dom, poll: 1e9 });
  await later();
  assert.equal(calm.node.querySelectorAll(".cx-code-warn, .cx-code-error").length, 0);
  calm.destroy();
});

test("an unknown tone paints nothing, and log text stays text", async () => {
  serve({ lines: ["<b>hi</b>"], tones: ["loud"], path: "p" });
  const view = mount({ dom, poll: 1e9 });
  await later();
  assert.equal(view.node.querySelector(".cx-code-line").className, "cx-code-line");
  assert.equal(view.node.querySelector("b"), null);
  view.destroy();
});
