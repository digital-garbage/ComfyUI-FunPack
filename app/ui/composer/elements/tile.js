// Cards and tiles: a rounded group of fields, a pressable tile, and the grid that lays tiles out.
import { define, lookup } from "../internals/register.js";
import { el } from "../internals/el.js";
import { setText } from "../internals/text.js";

/** Related fields on one rounded card. Takes handles, like a stack. */
define("card", "default", ({ children = [], gap = "md" } = {}) => {
  const stack = lookup("region", "stack")({ children, gap });
  stack.node.classList.add("cx-fieldcard");
  return stack;
});

/** Tiles in rows. `pair` is two columns; `cards` fills the width with columns at least 190 wide. */
for (const kind of ["pair", "cards"]) {
  define("grid", kind, ({ children = [] } = {}) => {
    const stack = lookup("region", "stack")({ children, gap: "none" });
    stack.node.classList.add("cx-grid", `cx-grid-${kind}`);
    return stack;
  });
}

const TONES = { ready: "Ready", needs: "Needs attention", skipped: "Skipped: not available" };

function tile(kind, { title, sub, onClick, state, label } = {}) {
  if (state !== undefined && !TONES[state]) throw new RangeError(`Unknown tile state "${state}". Known: ${Object.keys(TONES).join(", ")}.`);
  const head = el("span", { cls: "cx-tile-title", text: title });
  const hint = el("span", { cls: "cx-tile-sub", text: sub });
  const dot = state ? el("span", { cls: ["cx-tile-state", `cx-tile-${state}`], attrs: { title: TONES[state] } }) : null;
  const node = el("button", { cls: ["cx-tile", `cx-tile-${kind}`, "cx-focusable"], attrs: { type: "button", "aria-label": label },
    children: [dot, head, sub != null && sub !== "" ? hint : null].filter(Boolean) });
  if (onClick) node.addEventListener("click", onClick);
  return { node, setTitle: (v) => setText(head, v), setSub: (v) => setText(hint, v), destroy: () => node.remove() };
}

/** A whole-tile button: a title over one line of hint. */
define("tile", "action", (props = {}) => tile("action", props));
/** A node of the pipeline: title, one line of what is set, and a dot that says whether it is ready. */
define("tile", "node", (props = {}) => tile("node", { state: "ready", ...props }));
/** The empty slot at the end of a group: "+ Add". */
define("tile", "add", (props = {}) => tile("add", props));
