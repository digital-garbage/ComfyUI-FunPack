// Type. Nothing here carries a margin: spacing belongs to the container, so a
// heading can be dropped anywhere without bringing its own gap.

import { define } from "../internals/register.js";
import { el } from "../internals/el.js";
import { setText } from "../internals/text.js";
import { uid } from "../internals/ids.js";

const handle = (node) => ({
  node,
  setText: (value) => setText(node, value),
  destroy: () => node.remove(),
});

const TAGS = { display: "h1", xl: "h1", lg: "h2", md: "h3", sm: "h4" };

for (const size of Object.keys(TAGS)) {
  define("header", size, ({ text, id } = {}) =>
    handle(el(TAGS[size], { cls: `cx-h cx-h-${size}`, text, attrs: { id } })));
}

for (const size of ["lg", "md", "sm", "xs"]) {
  define("text", size, ({ text } = {}) =>
    handle(el("p", { cls: `cx-t cx-t-${size}`, text })));
}

// One hint, one look. v4 had ten class names for this exact thing.
define("hint", "default", ({ text, id } = {}) =>
  handle(el("p", { cls: "cx-hint", text, attrs: { id } })));

define("label", "field", ({ text, for: htmlFor } = {}) =>
  handle(el("label", { cls: "cx-label", text, attrs: { for: htmlFor } })));

define("label", "section", ({ text } = {}) =>
  handle(el("div", { cls: "cx-eyebrow", text })));

/** Who this is. One mark and one word, in the display face, at the far left of
 *  the menu bar -- the only place in the app that names the app. */
define("brand", "default", ({ name = "FunPack" } = {}) =>
  ({ node: el("span", { cls: "cx-brand", children: [
      el("span", { cls: "cx-brand-mark", attrs: { "aria-hidden": "true" } }),
      el("span", { cls: "cx-brand-name", text: name }),
    ] }),
     destroy() { this.node.remove(); } }));

define("code", "inline", ({ text } = {}) =>
  handle(el("code", { cls: "cx-code", text })));

define("code", "block", ({ text, label } = {}) => {
  const id = uid("code");
  const pre = el("pre", { cls: "cx-code-block", text, attrs: { id, "aria-label": label } });
  return handle(pre);
});
