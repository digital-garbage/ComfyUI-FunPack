// Shortcut autocomplete on a prompt textarea, over a mocked shortcuts.js.
//
// The matching algorithm itself (word-run under the caret, longest trigger
// first) is exercised through real typing/caret positions here rather than
// as isolated unit tests -- attach() is the only export, by design (see its
// own file), so the behaviour worth pinning is what a user actually sees.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let attach;
const fetched = [];
test.before(async () => {
  setupDom();
  await import("../../composer/composer.js");
  // shortcuts.js's fetchAll() hits fetch() itself; mocked once for the whole
  // file since every test wants the same small library.
  globalThis.fetch = async () => {
    fetched.push(1);
    return {
      ok: true,
      json: async () => ({ shortcuts: [
        { name: "Fox", triggers: ["golden hour", "golden"], replacements: ["warm light"], enabled: true, category: "", sub_category: "" },
        { name: "Off", triggers: ["off"], replacements: ["x"], enabled: false, category: "", sub_category: "" },
      ] }),
    };
  };
  ({ attach } = await import("../autocomplete.js"));
});
test.after(() => teardownDom());

function field(value = "") {
  // A fresh document each time: a popover from the PREVIOUS test's field is
  // portal-mounted onto document.body too, and without this a later test's
  // querySelector(".cx-autocomplete") can find that leftover menu instead of
  // (or as well as) its own, since nothing here ever closes one.
  document.body.replaceChildren();
  const ta = document.createElement("textarea");
  ta.value = value;
  document.body.appendChild(ta);
  return ta;
}

const typeAt = (ta, value, caret = value.length) => {
  ta.value = value;
  ta.selectionStart = ta.selectionEnd = caret;
  ta.dispatchEvent(new window.Event("input", { bubbles: true }));
};

const wait = () => new Promise((r) => setTimeout(r, 0));

test("attach is a no-op the second time on the same field", () => {
  const ta = field();
  attach(ta);
  attach(ta);
  assert.equal(ta._acAttached, true);
});

test("typing a matching trigger's tail shows it as a suggestion", async () => {
  const ta = field();
  attach(ta);
  await wait();               // let the warm-up fetch land
  typeAt(ta, "a gold");
  await wait();
  const menu = document.querySelector(".cx-autocomplete");
  assert.ok(menu, "no suggestion menu appeared");
  assert.match(menu.textContent, /golden/);
});

test("the longer trigger is offered ahead of the shorter one it contains", async () => {
  const ta = field();
  attach(ta);
  await wait();
  typeAt(ta, "golden hou");
  await wait();
  const labels = [...document.querySelectorAll(".cx-autocomplete .cx-menu-label")].map((n) => n.textContent);
  assert.equal(labels[0], "golden hour");
});

test("a disabled shortcut's trigger is never suggested", async () => {
  const ta = field();
  attach(ta);
  await wait();
  typeAt(ta, "of");
  await wait();
  assert.equal(document.querySelector(".cx-autocomplete"), null);
});

test("accepting a suggestion replaces only the matched word run", async () => {
  const ta = field();
  attach(ta);
  await wait();
  typeAt(ta, "a gold");
  await wait();
  // "gold" ties "golden" and "golden hour" on rank/position, and the shorter
  // trigger sorts first on that tie (matchTriggers' own final ordering) --
  // so the top suggestion is "golden", not the longer one.
  const item = document.querySelector(".cx-autocomplete .cx-menu-item");
  assert.equal(item.querySelector(".cx-menu-label").textContent, "golden");
  item.click();
  await wait();
  assert.equal(ta.value, "a golden ");
});

test("accepting before an existing delimiter does not add a second space", async () => {
  const ta = field();
  attach(ta);
  await wait();
  ta.value = "a gold, cool";
  ta.selectionStart = ta.selectionEnd = "a gold".length;
  ta.dispatchEvent(new window.Event("input", { bubbles: true }));
  await wait();
  const item = document.querySelector(".cx-autocomplete .cx-menu-item");
  item.click();
  await wait();
  assert.equal(ta.value, "a golden, cool");
});

test("a query shorter than two characters suggests nothing", async () => {
  const ta = field();
  attach(ta);
  await wait();
  typeAt(ta, "a g");
  await wait();
  assert.equal(document.querySelector(".cx-autocomplete"), null);
});
