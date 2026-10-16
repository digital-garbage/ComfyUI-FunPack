// A DOM for the internals that genuinely need one.
//
// jsdom is honest about structure and events and dishonest about layout -- it
// has no real box model, so anything that depends on measured geometry belongs
// in the browser (see the catalogue and the Playwright pass), not here.

import { JSDOM } from "jsdom";

const GLOBALS = [
  "window", "document", "Node", "HTMLElement", "Event", "KeyboardEvent",
  "CustomEvent", "getComputedStyle", "DocumentFragment", "Text",
];

export function setupDom(html = "<!doctype html><html><body></body></html>") {
  // A url, because without one jsdom gives the document an OPAQUE ORIGIN and
  // `window.localStorage` throws a SecurityError on the property access itself
  // -- not on get or set. Anything that remembers a preference then silently
  // took its fallback path in every test, so "it is remembered" was untestable
  // here and looked like it worked.
  const dom = new JSDOM(html, { pretendToBeVisual: true, url: "http://localhost/funpack/" });
  for (const name of GLOBALS) globalThis[name] = dom.window[name];
  globalThis.window = dom.window;
  globalThis.document = dom.window.document;
  return dom;
}

export function teardownDom() {
  for (const name of GLOBALS) delete globalThis[name];
}

/** Dispatch an event as it would arrive from a real interaction. */
export const fire = (node, type, init = {}) =>
  node.dispatchEvent(new globalThis.window.Event(type, { bubbles: true, cancelable: true, ...init }));

export const key = (node, k) =>
  node.dispatchEvent(new globalThis.window.KeyboardEvent("keydown", {
    key: k, bubbles: true, cancelable: true,
  }));
