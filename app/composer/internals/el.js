// The only document.createElement in the app.
//
// Everything the kit renders comes through here, which is what makes the
// no-styling rule checkable: a module has no route to a raw element, and the
// scanner only has one construction site to watch.

import { asText } from "./text.js";

const UNSAFE_ATTRS = new Set(["style", "class", "srcdoc", "onload", "onerror"]);

/**
 * el("div", { cls, text, attrs, on, children })
 *
 * `cls` takes kit class names only -- callers inside the kit. `text` goes
 * through asText. `attrs` refuses style/class and any inline handler, so no
 * amount of prop-forwarding can turn data into presentation or script.
 */
export function el(tag, opts = {}) {
  const node = document.createElement(tag);
  const { cls, text, attrs, on, children } = opts;

  if (cls) node.className = Array.isArray(cls) ? cls.filter(Boolean).join(" ") : cls;
  if (text !== undefined) node.appendChild(asText(text));

  if (attrs) {
    for (const [name, value] of Object.entries(attrs)) {
      if (value == null || value === false) continue;
      const lower = name.toLowerCase();
      if (UNSAFE_ATTRS.has(lower) || lower.startsWith("on")) {
        throw new TypeError(`Refusing to set "${name}": presentation and handlers are the kit's, not a caller's.`);
      }
      node.setAttribute(name, value === true ? "" : String(value));
    }
  }

  if (on) for (const [event, handler] of Object.entries(on)) node.addEventListener(event, handler);
  if (children) for (const child of [].concat(children)) if (child) node.appendChild(child);

  return node;
}

const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * An SVG element. Separate because SVG needs createElementNS -- createElement
 * would produce an HTMLUnknownElement that renders as nothing at all.
 * Attributes only: SVG carries no module content, so there is no text path here.
 */
export function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    node.setAttribute(name, String(value));
  }
  return node;
}

/** A detached fragment. Panels build into one so a half-built tree never lands. */
export function frag() {
  return document.createDocumentFragment();
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}
