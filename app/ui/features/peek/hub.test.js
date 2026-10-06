import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let peek;
test.before(async () => { setupDom(); globalThis.localStorage = window.localStorage; peek = (await import("./index.js")).default; });
test.after(() => { delete globalThis.localStorage; teardownDom(); });

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const point = (node) => node.dispatchEvent(new window.Event("pointerover", { bubbles: true }));

test("auto-hide: the title bar (Generate, Rate) never opens the timeline; resting on the strip does", async () => {
  localStorage.setItem("funpack_timeline_peek", "1");
  document.body.innerHTML = `<div class="fp-main"><div class="fp-row"></div><section><header class="cx-panel-head"><button id="gen">Generate</button><span id="status"></span></header><div class="cx-panel-body"><div id="lane"></div></div></section></div>`;
  const root = document.documentElement, isOpen = () => root.classList.contains("fp-peek-open");
  const off = peek.setup({ host: document.getElementById("status") });
  const strip = document.querySelector(".fp-peek-strip");
  assert.ok(strip && strip.nextElementSibling.classList.contains("cx-panel-body"), "the strip sits between title bar and timeline");

  point(document.getElementById("gen")); await wait(300);
  assert.equal(isOpen(), false, "reaching for Generate keeps it shut");

  point(strip); point(document.querySelector(".fp-row")); await wait(300);
  assert.equal(isOpen(), false, "passing over the strip does not flash it open");

  point(strip); await wait(300);
  assert.equal(isOpen(), true, "resting on the strip opens it");
  point(document.getElementById("lane")); await wait(10);
  assert.equal(isOpen(), true, "the timeline itself keeps it open");
  point(document.getElementById("gen"));
  assert.equal(isOpen(), false, "back on the title bar, it shuts");
  off();
});
