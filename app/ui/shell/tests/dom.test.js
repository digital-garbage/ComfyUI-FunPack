import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let createDom;
test.before(async () => { setupDom(); ({ createDom } = await import("../dom.js")); });
test.after(() => teardownDom());

test("typingIn is true only for a text field inside the node that has the caret", () => {
  document.body.innerHTML = `<div id="a"><input id="i"><button id="b"></button></div><textarea id="t"></textarea>`;
  const dom = createDom(), a = document.getElementById("a");
  assert.equal(dom.typingIn(a), false);
  document.getElementById("i").focus();
  assert.equal(dom.typingIn(a), true);
  assert.equal(dom.typingIn(a, "TEXTAREA"), false, "narrowed to one tag");
  document.getElementById("t").focus();
  assert.equal(dom.typingIn(a), false, "a field outside is not inside");
  assert.equal(dom.focusOn(document.getElementById("t")), true);
  document.getElementById("b").focus();
  assert.equal(dom.holdsFocus(a), true);
  dom.blur();
  assert.equal(dom.holdsFocus(a), false);
});

test("listeners come back with their own off switch", () => {
  const dom = createDom(); let n = 0;
  const off = dom.onPage("pointerdown", () => { n += 1; });
  document.body.dispatchEvent(new window.Event("pointerdown", { bubbles: true }));
  off();
  document.body.dispatchEvent(new window.Event("pointerdown", { bubbles: true }));
  assert.equal(n, 1);
});

test("a dialog on screen is reported, and a landmark is found by name", () => {
  document.body.innerHTML = `<header class="cx-frame-head"></header><main class="cx-frame-main"><section class="cx-zone"></section><section class="cx-zone"></section><button> ▶ Generate </button></main>`;
  const dom = createDom();
  assert.equal(dom.dialogOpen(), false);
  document.body.insertAdjacentHTML("beforeend", `<div role="dialog"></div>`);
  assert.equal(dom.dialogOpen(), true);
  assert.equal(dom.landmark("menubar").className, "cx-frame-head");
  assert.equal(dom.landmark("zone:1"), document.querySelectorAll("section.cx-zone")[1]);
  assert.equal(dom.landmark("button:▶ Generate").tagName, "BUTTON");
  assert.equal(dom.landmark("zone:9"), null);
  assert.equal(dom.landmark("nonsense"), null);
});

test("a download is a click on a link with the file's name", () => {
  const dom = createDom(), seen = [];
  const click = window.HTMLAnchorElement.prototype.click;
  window.HTMLAnchorElement.prototype.click = function () { seen.push([this.getAttribute("href"), this.download]); };
  try { dom.download("/funpack/api/x", "x.json"); } finally { window.HTMLAnchorElement.prototype.click = click; }
  assert.deepEqual(seen, [["/funpack/api/x", "x.json"]]);
});

test("a file chooser that is cancelled gives null, and one that is answered gives the file", async () => {
  const dom = createDom();
  const click = window.HTMLInputElement.prototype.click;
  let input;
  window.HTMLInputElement.prototype.click = function () { input = this; };
  try {
    const cancelled = dom.pickFile(".json"); input.oncancel();
    assert.equal(await cancelled, null);
    assert.equal(input.accept, ".json");
    const picked = dom.pickFile(".json");
    Object.defineProperty(input, "files", { value: [{ name: "a.json" }] });
    input.onchange();
    assert.deepEqual(await picked, { name: "a.json" });
  } finally { window.HTMLInputElement.prototype.click = click; }
});

test("selectingIn is true only while text inside the node is selected", () => {
  document.body.innerHTML = `<pre id="p">some log text</pre><p id="q">elsewhere</p>`;
  const dom = createDom(), p = document.getElementById("p");
  assert.equal(dom.selectingIn(p), false);
  const range = document.createRange(); range.selectNodeContents(p.firstChild);
  const sel = document.getSelection(); sel.removeAllRanges(); sel.addRange(range);
  assert.equal(dom.selectingIn(p), true);
  assert.equal(dom.selectingIn(document.getElementById("q")), false);
});
