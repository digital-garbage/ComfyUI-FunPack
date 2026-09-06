// Text entry.
//
// onInput fires on every keystroke; onCommit fires on blur and Enter. Panels
// almost always want onCommit -- a settings value that updates per keystroke
// writes nine intermediate values while you type "0.65".

import { define } from "../internals/register.js";
import { el } from "../internals/el.js";
import { uid } from "../internals/ids.js";

function wire(node, { onInput, onCommit }) {
  if (onInput) node.addEventListener("input", () => onInput(node.value));
  if (onCommit) {
    let last = node.value;
    const commit = () => { if (node.value !== last) { last = node.value; onCommit(node.value); } };
    node.addEventListener("blur", commit);
    // And `change`, which is what autofill, a password manager and a script
    // setting the value all fire without ever taking focus. commit() only acts
    // on a real difference, so the pair costs nothing when both arrive.
    node.addEventListener("change", commit);
    // Enter commits a one-line field. In a TEXTAREA it is a newline, and
    // stealing it there means a multi-line value cannot be typed at all -- the
    // prompt included, which is the longest multi-line thing in the app.
    if (node.tagName !== "TEXTAREA") {
      node.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); commit(); } });
    }
  }
}

function textLike(tag, cls, { value = "", placeholder, onInput, onCommit, disabled, id, type, label } = {}) {
  const node = el(tag, {
    cls: [cls, "cx-focusable"],
    attrs: { placeholder, disabled, id: id || uid("input"), type, "aria-label": label },
  });
  node.value = value;
  wire(node, { onInput, onCommit });
  return {
    node,
    get value() { return node.value; },
    setValue(v) { node.value = v ?? ""; },
    focus: () => node.focus(),
    destroy: () => node.remove(),
  };
}

for (const size of ["md", "sm"]) {
  define("input", size, (props = {}) => textLike("input", `cx-input cx-input-${size}`, { type: "text", ...props }));
}

define("search", "md", (props = {}) => {
  const handle = textLike("input", "cx-input cx-input-md cx-search", { type: "search", placeholder: "Search", ...props });
  return handle;
});

define("number", "md", ({ value = 0, min, max, step = 1, precision, unit, onChange, disabled, id, label } = {}) => {
  const input = el("input", {
    cls: ["cx-input", "cx-input-md", "cx-number", "cx-focusable"],
    attrs: { type: "number", min, max, step, disabled, id: id || uid("num"), "aria-label": label },
  });
  input.value = String(value);

  let committed = Number(value);

  const clamp = (n) => {
    if (min !== undefined && n < min) return min;
    if (max !== undefined && n > max) return max;
    return precision === undefined ? n : Number(n.toFixed(precision));
  };

  // Clamping on commit rather than on input: clamping mid-typing makes "-" or
  // a leading "0." impossible to type.
  //
  // An emptied or unparseable field reverts to the last committed value rather
  // than being read as a number. A browser normalises invalid text in a number
  // input to "", and Number("") is 0 -- which is finite, so it sails past any
  // isFinite guard and lands on min, silently discarding what the user had.
  const commit = () => {
    const raw = input.value.trim();
    const parsed = raw === "" ? NaN : Number(raw);
    const next = Number.isFinite(parsed) ? clamp(parsed) : committed;
    committed = next;
    input.value = String(next);
    if (onChange) onChange(next);
  };
  input.addEventListener("blur", commit);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); commit(); } });

  const node = unit
    ? el("span", { cls: "cx-number-wrap", children: [input, el("span", { cls: "cx-unit", text: unit })] })
    : input;

  return {
    node,
    get value() { return committed; },
    setValue(v) {
      const parsed = Number(v);
      committed = Number.isFinite(parsed) ? clamp(parsed) : committed;
      input.value = String(committed);
    },
    destroy: () => node.remove(),
  };
});

define("stepper", "md", ({ value = 0, min, max, step = 1, onChange, label } = {}) => {
  let current = value;
  const out = el("input", { cls: ["cx-input", "cx-input-md", "cx-number", "cx-focusable"],
    attrs: { type: "number", min, max, step, "aria-label": label } });
  out.value = String(current);

  const set = (next) => {
    if (min !== undefined) next = Math.max(min, next);
    if (max !== undefined) next = Math.min(max, next);
    current = Number(next.toFixed(6));       // 0.1 + 0.2 must not become 0.30000000000000004
    out.value = String(current);
    if (onChange) onChange(current);
  };

  const bump = (dir) => el("button", {
    cls: ["cx-step-btn", "cx-focusable"],
    text: dir > 0 ? "+" : "−",
    attrs: { type: "button", "aria-label": `${dir > 0 ? "Increase" : "Decrease"}${label ? ` ${label}` : ""}` },
    on: { click: () => set(current + dir * step) },
  });

  out.addEventListener("change", () => set(Number(out.value)));
  const node = el("span", { cls: "cx-stepper", children: [bump(-1), out, bump(1)] });

  return { node, get value() { return current; }, setValue: set, destroy: () => node.remove() };
});

define("textarea", "md", ({ value = "", rows = 4, autoGrow = false, placeholder, onInput, onCommit, label } = {}) => {
  const node = el("textarea", {
    cls: ["cx-textarea", "cx-focusable", autoGrow ? "cx-autogrow" : null],
    attrs: { rows, placeholder, "aria-label": label },
  });
  node.value = value;

  const grow = () => { node.style.height = "auto"; node.style.height = `${node.scrollHeight}px`; };
  if (autoGrow) { node.addEventListener("input", grow); queueMicrotask(grow); }
  wire(node, { onInput, onCommit });

  return {
    node,
    get value() { return node.value; },
    setValue(v) { node.value = v ?? ""; if (autoGrow) grow(); },
    destroy: () => node.remove(),
  };
});

/**
 * A search box over a list. v4 had seven of these, each with its own filter.
 * Items are DATA -- {id, label, hint, icon} -- never a render callback, so a
 * module cannot smuggle markup in through a "just render this" escape hatch.
 */
define("filterList", "md", ({ items = [], value, onChange, placeholder = "Search", empty = "Nothing matches" } = {}) => {
  let selected = value;
  const search = el("input", { cls: ["cx-input", "cx-input-sm", "cx-search", "cx-focusable"],
    attrs: { type: "search", placeholder } });
  const list = el("div", { cls: "cx-filter-list", attrs: { role: "listbox" } });

  function draw() {
    const q = search.value.trim().toLowerCase();
    const shown = items.filter((i) =>
      !q || `${i.label} ${i.hint || ""}`.toLowerCase().includes(q));
    list.replaceChildren();
    if (!shown.length) { list.append(el("div", { cls: "cx-filter-empty", text: empty })); return; }
    for (const item of shown) {
      const row = el("button", {
        cls: ["cx-filter-row", "cx-focusable", item.id === selected ? "cx-on" : null],
        attrs: { type: "button", role: "option", "aria-selected": String(item.id === selected) },
        children: [
          item.icon ? el("span", { cls: "cx-filter-icon", text: item.icon }) : null,
          el("span", { cls: "cx-filter-label", text: item.label }),
          item.hint ? el("span", { cls: "cx-filter-hint", text: item.hint }) : null,
        ].filter(Boolean),
      });
      row.addEventListener("click", () => { selected = item.id; draw(); if (onChange) onChange(item.id); });
      list.append(row);
    }
  }

  search.addEventListener("input", draw);
  draw();

  const node = el("div", { cls: "cx-filter", children: [search, list] });
  return {
    node,
    get value() { return selected; },
    setValue(v) { selected = v; draw(); },
    destroy: () => node.remove(),
  };
});
