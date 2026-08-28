// Rule 4: module-supplied content is DATA, never markup.
//
// This test iterates THE REGISTRY, not a list, so an element added tomorrow is
// covered the moment it calls define(). That is the same inversion the kit uses
// everywhere else -- announcement, not enumeration -- applied to its own tests.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom } from "./_dom.js";
import { DEMOS } from "../../catalogue/demos.js";
import { ROOT_ID } from "../internals/portal.js";

const PAYLOAD = '<img src=x onerror="boom()">';

let entries;
let composer;
test.before(async () => {
  setupDom();
  ({ composer, entries } = await import("../composer.js"));
});

// A container's demo is a function, because it has to build the handles it
// composes. Poisoning then applies to that container's OWN content props -- the
// elements inside it were attacked when their own demos ran.
const resolve = (demo) => (typeof demo === "function" ? demo(composer) : demo);
test.after(() => teardownDom());

// Poison CONTENT only. A prop like `tone` or `side` is kit vocabulary, not
// something a module writes prose into -- and those already refuse an unknown
// value loudly, which is a different guarantee tested elsewhere. Whitelisting
// content keeps this test honest as the kit grows: a new content prop has to be
// named here, and until it is, `assertPoisoned` fails rather than passing
// vacuously.
const CONTENT_KEYS = new Set([
  "text", "label", "hint", "title", "placeholder", "unit", "message",
  "subtitle", "addLabel", "confirmLabel", "cancelLabel", "name", "caption",
]);

const isHandle = (v) => v && typeof v === "object" && v.node && v.node.nodeType === 1;

function poison(value, key = null, counter = { n: 0 }) {
  // A handle is a built element, not props. Copying it entry by entry would
  // strip its getters and hand the container something that only looks like one.
  if (isHandle(value)) return value;
  if (Array.isArray(value)) return value.map((v) => poison(v, key, counter));
  if (value && typeof value === "object") {
    const out = {};
    for (const [k, v] of Object.entries(value)) out[k] = poison(v, k, counter);
    return out;
  }
  if (typeof value === "string" && CONTENT_KEYS.has(key)) {
    counter.n += 1;
    return PAYLOAD;
  }
  return value;
}

test("every registered element has a demo", () => {
  const missing = [...entries()]
    .map((e) => `${e.group}.${e.variant}`)
    .filter((name) => !(name in DEMOS));
  assert.deepEqual(missing, [],
    "add these to catalogue/demos.js -- an element with no demo is neither shown nor attacked");
});

test("no element renders module content as markup", () => {
  for (const { group, variant, factory } of entries()) {
    const name = `${group}.${variant}`;
    const demo = DEMOS[name];
    if (!demo) continue;

    const counter = { n: 0 };
    const resolved = resolve(demo);
    const props = poison(resolved, null, counter);

    if (counter.n === 0) {
      // A pure container takes handles and no content of its own -- there is
      // nothing here to attack, and its children were attacked under their own
      // demos. Anything else with no content prop is a demo that forgot one, or
      // a prop name CONTENT_KEYS has not learned yet.
      const composes = Object.values(resolved).some(
        (v) => isHandle(v) || (Array.isArray(v) && v.some(isHandle)));
      assert.ok(composes,
        `${name}'s demo has no content prop this test recognises and composes nothing -- ` +
        "either the demo is missing content, or its prop name belongs in CONTENT_KEYS");
      continue;
    }

    const handle = factory(props);
    // An overlay puts its content in the portal, not under the handle's node --
    // and a tooltip only builds its content when shown. Search both.
    if (handle.show) handle.show();
    const portalRoot = document.getElementById(ROOT_ID);
    const roots = [handle.node, portalRoot].filter(Boolean);
    const node = handle.node;

    for (const root of roots) {
      assert.equal(root.querySelector("img"), null, `${name} rendered the payload as an element`);
      assert.equal(root.querySelector("script"), null, `${name} rendered a script`);
    }
    // Some elements carry content in an attribute instead of text -- an
    // icon-only button's label is its accessible name. Either is fine; the DOM
    // escapes attribute values. What must never happen is it becoming markup.
    const everywhere = roots.flatMap((root) => [root, ...root.querySelectorAll("*")]);
    const inAttributes = everywhere.some((child) =>
      child.getAttributeNames().some((attr) => child.getAttribute(attr).includes(PAYLOAD)));
    const inText = roots.some((root) => root.textContent.includes(PAYLOAD));
    assert.ok(inText || inAttributes,
      `${name} dropped the payload entirely -- it renders neither as text nor as an attribute`);

    // The kit may write inline styles -- a slider's fill percentage, a
    // popover's position -- that is rule 1 permitting the KIT to style. What is
    // forbidden is CONTENT reaching a style, or any inline handler at all.
    for (const child of everywhere) {
      const style = child.getAttribute("style");
      assert.ok(!style || !style.includes(PAYLOAD), `${name} put content into an inline style`);
      for (const attr of child.getAttributeNames()) {
        assert.ok(!attr.startsWith("on"), `${name} set an inline handler (${attr})`);
      }
    }

    handle.destroy();
  }
});

test("every element returns the same shape of handle", () => {
  // The panel renderer treats them alike; an element that returns a bare node
  // would break it only at the point some module happens to use that element.
  for (const { group, variant, factory } of entries()) {
    const name = `${group}.${variant}`;
    const demo = DEMOS[name];
    if (!demo) continue;
    const handle = factory(resolve(demo));
    assert.ok(handle && handle.node, `${name} returned no node`);
    assert.equal(typeof handle.destroy, "function", `${name} has no destroy()`);
    assert.equal(handle.node.nodeType, 1, `${name} node is not an element`);
    handle.destroy();
  }
});

test("destroy() leaves nothing behind", () => {
  // Overlays own their placement -- appending one here would drag it out of the
  // portal and then destroy() would have nothing to remove. For those the test
  // is that the portal comes back empty; for everything else, that the node
  // detaches from wherever the caller put it.
  for (const { group, variant, factory } of entries()) {
    const name = `${group}.${variant}`;
    const demo = DEMOS[name];
    if (!demo) continue;

    const handle = factory(resolve(demo));
    if (handle.isOverlay) {
      handle.destroy();
      const portalRoot = document.getElementById(ROOT_ID);
      assert.equal(portalRoot ? portalRoot.childElementCount : 0, 0,
        `${name} left something in the overlay root`);
    } else {
      document.body.appendChild(handle.node);
      handle.destroy();
      assert.equal(handle.node.isConnected, false, `${name} stayed in the document`);
    }
  }
});
