// Buttons.
//
// Every one is focusable with a visible ring -- v4 styled a focus ring for
// inputs and gave buttons none, so keyboard users lost their place the moment
// they tabbed onto one.

import { define } from "../internals/register.js";
import { el } from "../internals/el.js";
import { setText } from "../internals/text.js";
import { roving } from "../internals/focus.js";

const TONES = new Set(["neutral", "primary", "danger", "ghost"]);

function base(size, { label, onClick, tone = "neutral", icon, disabled = false, busy = false, type = "button", title } = {}) {
  if (!TONES.has(tone)) throw new RangeError(`Unknown button tone "${tone}". Known: ${[...TONES].join(", ")}.`);

  const node = el("button", {
    cls: ["cx-btn", `cx-btn-${size}`, `cx-btn-${tone}`, "cx-focusable"],
    attrs: { type, disabled: disabled || busy, title, "aria-busy": busy || undefined },
  });

  const spinner = el("span", { cls: "cx-btn-spin", attrs: { "aria-hidden": "true" } });
  const glyph = icon ? el("span", { cls: "cx-btn-icon", text: icon, attrs: { "aria-hidden": "true" } }) : null;
  const text = el("span", { cls: "cx-btn-label", text: label });

  if (glyph) node.appendChild(glyph);
  node.appendChild(text);
  if (busy) node.appendChild(spinner);

  if (onClick) node.addEventListener("click", onClick);

  return {
    node,
    setLabel: (value) => setText(text, value),
    setBusy(value) {
      node.toggleAttribute("disabled", Boolean(value) || disabled);
      if (value) { node.setAttribute("aria-busy", "true"); node.appendChild(spinner); }
      else { node.removeAttribute("aria-busy"); spinner.remove(); }
    },
    setDisabled(value) { node.toggleAttribute("disabled", Boolean(value)); },
    destroy: () => node.remove(),
  };
}

for (const size of ["xl", "lg", "md", "sm"]) define("button", size, (props) => base(size, props));

// Icon-only controls take `label` as their accessible name, not as decoration.
// Making it required is the cheapest way to stop a toolbar of unlabelled glyphs.
for (const size of ["md", "sm", "micro"]) {
  define("iconButton", size, ({ icon, label, onClick, tone = "ghost", disabled, pressed } = {}) => {
    if (!label) throw new TypeError("iconButton needs a label: it is the accessible name and the tooltip.");
    const node = el("button", {
      cls: ["cx-icon-btn", `cx-icon-btn-${size}`, `cx-btn-${tone}`, "cx-focusable"],
      attrs: {
        type: "button", disabled, title: label, "aria-label": label,
        "aria-pressed": pressed === undefined ? undefined : String(Boolean(pressed)),
      },
      children: el("span", { cls: "cx-btn-icon", text: icon, attrs: { "aria-hidden": "true" } }),
    });
    if (onClick) node.addEventListener("click", onClick);
    return {
      node,
      setPressed(value) { node.setAttribute("aria-pressed", String(Boolean(value))); },
      destroy: () => node.remove(),
    };
  });
}

/** A row of buttons where one (or several) stay active: B / I, align, flips. */
define("buttonGroup", "md", ({ items = [], value, onChange, multi = false, label } = {}) => {
  const selected = new Set(multi ? [].concat(value || []) : value == null ? [] : [value]);
  const node = el("div", {
    cls: "cx-btn-group",
    attrs: { role: multi ? "group" : "radiogroup", "aria-label": label },
  });

  const buttons = items.map((item) => {
    const b = el("button", {
      cls: ["cx-btn", "cx-btn-md", "cx-btn-neutral", "cx-focusable"],
      text: item.label,
      attrs: {
        type: "button", title: item.title,
        role: multi ? undefined : "radio",
        "aria-checked": multi ? undefined : String(selected.has(item.value)),
        "aria-pressed": multi ? String(selected.has(item.value)) : undefined,
      },
    });
    b.classList.toggle("cx-on", selected.has(item.value));
    b.addEventListener("click", () => {
      if (multi) selected.has(item.value) ? selected.delete(item.value) : selected.add(item.value);
      else { selected.clear(); selected.add(item.value); }
      sync();
      if (onChange) onChange(multi ? [...selected] : item.value);
    });
    node.appendChild(b);
    return { item, b };
  });

  function sync() {
    for (const { item, b } of buttons) {
      const on = selected.has(item.value);
      b.classList.toggle("cx-on", on);
      if (multi) b.setAttribute("aria-pressed", String(on));
      else b.setAttribute("aria-checked", String(on));
    }
  }

  // One tab stop for the group; arrows move inside it. Otherwise Tab walks
  // every option, which is tedious in a row of eight.
  const stopRoving = roving(node);

  return {
    node,
    get value() { return multi ? [...selected] : [...selected][0]; },
    setValue(next) {
      selected.clear();
      for (const v of multi ? [].concat(next || []) : [next]) if (v != null) selected.add(v);
      sync();
    },
    destroy() { stopRoving(); node.remove(); },
  };
});
