// Choosing: on/off, one-of, several-of.
//
// The checkbox is drawn rather than native. v4 used the OS control with only
// accent-color, which meant its size, focus ring and disabled look were the
// browser's opinion, not the kit's -- and it needed `width: auto` undone in four
// places because a global `input { width: 100% }` had stretched it.

import { define } from "../internals/register.js";
import { el } from "../internals/el.js";
import { uid } from "../internals/ids.js";
import { roving } from "../internals/focus.js";

// `ariaLabel` is passed only by the bare checkbox, which has no visible text of
// its own. In a checkbox ROW the adjacent text is the label (via `for`), and
// adding aria-label there would override what the user can actually see.
function checkboxNode({ checked = false, indeterminate = false, disabled, id, ariaLabel, describedBy }) {
  const input = el("input", {
    cls: "cx-check-input",
    attrs: { type: "checkbox", id, disabled, checked: checked || undefined,
             "aria-describedby": describedBy, "aria-label": ariaLabel },
  });
  input.checked = Boolean(checked);
  input.indeterminate = Boolean(indeterminate);
  const box = el("span", { cls: "cx-check-box", attrs: { "aria-hidden": "true" } });
  return { input, box };
}

define("checkbox", "default", ({ checked, indeterminate, disabled, onChange, label } = {}) => {
  if (!label) throw new TypeError("checkbox needs a label: with no visible text it is its only accessible name. Use checkboxRow for a labelled row.");
  const id = uid("check");
  const { input, box } = checkboxNode({ checked, indeterminate, disabled, id, ariaLabel: label });
  if (onChange) input.addEventListener("change", () => onChange(input.checked));
  const node = el("span", { cls: ["cx-check", "cx-focusable"], children: [input, box] });
  return {
    node,
    get value() { return input.checked; },
    setValue(v) { input.checked = Boolean(v); input.indeterminate = false; },
    destroy: () => node.remove(),
  };
});

define("checkboxRow", "default", ({ label, hint, checked, disabled, onChange } = {}) => {
  const id = uid("check");
  const hintId = hint ? uid("hint") : undefined;
  const { input, box } = checkboxNode({ checked, disabled, id, describedBy: hintId });
  if (onChange) input.addEventListener("change", () => onChange(input.checked));

  const text = el("span", { cls: "cx-check-text", children: [
    el("span", { cls: "cx-check-label", text: label }),
    hint ? el("span", { cls: "cx-hint", text: hint, attrs: { id: hintId } }) : null,
  ].filter(Boolean) });

  // The input is a SIBLING of its label, not a child of it.
  //
  // Both forms are valid, and browsers guard a nested control against being
  // activated twice by its own label. Keeping them separate means never relying
  // on that guard -- the CSS reaches the box through the label instead of past
  // it, which costs one selector and removes a whole class of subtlety.
  const node = el("div", { cls: "cx-check-row", children: [
    input,
    el("label", { cls: "cx-check-face", attrs: { for: id }, children: [box, text] }),
  ] });
  return {
    node,
    get value() { return input.checked; },
    setValue(v) { input.checked = Boolean(v); },
    destroy: () => node.remove(),
  };
});

define("checklist", "default", ({ items = [], values = [], onChange, label } = {}) => {
  const chosen = new Set(values);
  const node = el("div", { cls: "cx-checklist", attrs: { role: "group", "aria-label": label } });
  for (const item of items) {
    const id = uid("check");
    const { input, box } = checkboxNode({ checked: chosen.has(item.value), id, disabled: item.disabled });
    input.addEventListener("change", () => {
      input.checked ? chosen.add(item.value) : chosen.delete(item.value);
      if (onChange) onChange([...chosen]);
    });
    node.append(el("div", { cls: "cx-check-row", children: [
      input,
      el("label", { cls: "cx-check-face", attrs: { for: id }, children: [
        box, el("span", { cls: "cx-check-label", text: item.label }),
      ] }),
    ] }));
  }
  return {
    node,
    get value() { return [...chosen]; },
    setValue(next) {
      chosen.clear();
      for (const v of next || []) chosen.add(v);
      node.querySelectorAll(".cx-check-input").forEach((input, i) => {
        input.checked = chosen.has(items[i].value);
      });
    },
    destroy: () => node.remove(),
  };
});

define("radioGroup", "default", ({ options = [], value, onChange, orientation = "vertical", label } = {}) => {
  const name = uid("radio");
  let current = value;
  const node = el("div", {
    cls: ["cx-radio-group", `cx-radio-${orientation}`],
    attrs: { role: "radiogroup", "aria-label": label },
  });

  const inputs = options.map((option) => {
    const id = uid("radio-opt");
    const input = el("input", { cls: "cx-radio-input",
      attrs: { type: "radio", name, id, disabled: option.disabled } });
    input.checked = option.value === current;
    input.addEventListener("change", () => {
      if (!input.checked) return;
      current = option.value;
      if (onChange) onChange(current);
    });
    node.append(el("div", { cls: "cx-radio", children: [
      input,
      el("label", { cls: "cx-check-face", attrs: { for: id }, children: [
        el("span", { cls: "cx-radio-dot", attrs: { "aria-hidden": "true" } }),
        el("span", { cls: "cx-check-text", children: [
          el("span", { cls: "cx-check-label", text: option.label }),
          option.hint ? el("span", { cls: "cx-hint", text: option.hint }) : null,
        ].filter(Boolean) }),
      ] }),
    ] }));
    return { option, input };
  });

  return {
    node,
    get value() { return current; },
    setValue(v) { current = v; for (const { option, input } of inputs) input.checked = option.value === v; },
    destroy: () => node.remove(),
  };
});

for (const size of ["md", "sm"]) {
  define("select", size, ({ options = [], value, onChange, disabled, label, id } = {}) => {
    const node = el("select", {
      cls: ["cx-select", `cx-select-${size}`, "cx-focusable"],
      attrs: { disabled, "aria-label": label, id: id || uid("select") },
    });
    for (const option of options) {
      const o = el("option", { text: option.label, attrs: { value: option.value, disabled: option.disabled } });
      node.append(o);
    }
    if (value !== undefined) node.value = value;
    if (onChange) node.addEventListener("change", () => onChange(node.value));
    return {
      node,
      get value() { return node.value; },
      setValue(v) { node.value = v; },
      destroy: () => node.remove(),
    };
  });
}

for (const size of ["md", "sm"]) {
  define("segmented", size, ({ options = [], value, onChange, label } = {}) => {
    let current = value ?? options[0]?.value;
    const node = el("div", {
      cls: ["cx-segmented", `cx-segmented-${size}`],
      attrs: { role: "radiogroup", "aria-label": label },
    });

    const buttons = options.map((option) => {
      const b = el("button", {
        cls: ["cx-segment", "cx-focusable"],
        text: option.label,
        attrs: { type: "button", role: "radio", "aria-checked": String(option.value === current) },
      });
      b.classList.toggle("cx-on", option.value === current);
      b.addEventListener("click", () => { current = option.value; sync(); if (onChange) onChange(current); });
      node.append(b);
      return { option, b };
    });

    function sync() {
      for (const { option, b } of buttons) {
        const on = option.value === current;
        b.classList.toggle("cx-on", on);
        b.setAttribute("aria-checked", String(on));
      }
    }

    const stopRoving = roving(node);
    return {
      node,
      get value() { return current; },
      setValue(v) { current = v; sync(); },
      destroy() { stopRoving(); node.remove(); },
    };
  });
}

define("toggle", "default", ({ label, hint, checked, disabled, onChange } = {}) => {
  const id = uid("toggle");
  const input = el("input", { cls: "cx-toggle-input",
    attrs: { type: "checkbox", id, disabled, role: "switch" } });
  input.checked = Boolean(checked);
  if (onChange) input.addEventListener("change", () => onChange(input.checked));

  const node = el("div", { cls: "cx-toggle-row", children: [
    input,
    el("label", { cls: "cx-check-face cx-toggle-face", attrs: { for: id }, children: [
      el("span", { cls: "cx-check-text", children: [
        el("span", { cls: "cx-check-label", text: label }),
        hint ? el("span", { cls: "cx-hint", text: hint }) : null,
      ].filter(Boolean) }),
      el("span", { cls: "cx-toggle-track", attrs: { "aria-hidden": "true" },
        children: el("span", { cls: "cx-toggle-knob" }) }),
    ] }),
  ] });

  return {
    node,
    get value() { return input.checked; },
    setValue(v) { input.checked = Boolean(v); },
    destroy: () => node.remove(),
  };
});

// A colour, picked or typed.
//
// Two controls over one value because neither is enough alone: the swatch is how
// you choose one you have not named, the field is how you paste the one you were
// given. They stay in step, and the field only commits when what it holds is a
// real colour -- a half-typed "#ff8" must not repaint the app mid-keystroke.
define("color", "swatch", ({ label, value = "#000000", onChange, disabled = false } = {}) => {
  const full = (v) => (v.length === 4
    ? "#" + v.slice(1).split("").map((c) => c + c).join("")
    : v.slice(0, 7));

  const swatch = el("input", { cls: ["cx-swatch", "cx-focusable"],
    attrs: { type: "color", value: full(value), disabled: disabled || null,
             "aria-label": label ? `${label} colour` : "colour" } });

  const field = el("input", { cls: ["cx-input", "cx-input-md", "cx-swatch-hex", "cx-focusable"],
    attrs: { type: "text", value, spellcheck: "false", disabled: disabled || null,
             "aria-label": label ? `${label} hex value` : "hex value" } });

  let current = value;

  const valid = (v) => /^#([0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{8})$/i.test(v);

  function commit(next, from) {
    if (!valid(next) || next === current) return;
    current = next;
    if (from !== "swatch") swatch.value = full(next);
    if (from !== "field") field.value = next;
    if (onChange) onChange(next);
  }

  swatch.addEventListener("input", () => commit(swatch.value, "swatch"));
  field.addEventListener("change", () => {
    if (valid(field.value)) commit(field.value, "field");
    else field.value = current;          // reject rather than half-apply
  });

  const node = el("div", { cls: "cx-swatch-row", children: [swatch, field] });
  return {
    node,
    get value() { return current; },
    setValue: (next) => commit(next, null),
    destroy: () => node.remove(),
  };
});
