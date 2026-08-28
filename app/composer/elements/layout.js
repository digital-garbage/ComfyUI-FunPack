// Containers.
//
// These take other elements' HANDLES, never raw nodes: a handle is something the
// kit made, and insisting on one is what stops "just let me pass a div" becoming
// the hole through which modules build their own markup.

import { define } from "../internals/register.js";
import { el } from "../internals/el.js";
import { uid } from "../internals/ids.js";
import { drag } from "../internals/drag.js";
import { roving } from "../internals/focus.js";

function nodeOf(control, where) {
  if (!control) return null;
  if (control.node && control.node.nodeType === 1) return control.node;
  throw new TypeError(
    `${where} takes a Composer element handle, not a ${control.nodeType === 1 ? "raw DOM node" : typeof control}. ` +
    "Build the control with composer.<group>.<variant>() and pass what it returns."
  );
}

/** Label above a control, with an optional hint and error beneath. */
define("field", "default", ({ label, hint, control, error } = {}) => {
  const id = uid("field");
  const hintId = hint ? uid("hint") : null;
  const inner = nodeOf(control, "field.default");
  if (inner && !inner.id) inner.id = id;
  if (inner && hintId) inner.setAttribute("aria-describedby", hintId);

  const node = el("div", { cls: "cx-field", children: [
    label ? el("label", { cls: "cx-label", text: label, attrs: { for: inner ? inner.id : undefined } }) : null,
    inner,
    hint ? el("p", { cls: "cx-hint", text: hint, attrs: { id: hintId } }) : null,
    error ? el("p", { cls: "cx-field-error", text: error, attrs: { role: "alert" } }) : null,
  ].filter(Boolean) });

  return { node, control, destroy: () => node.remove() };
});

define("field", "row", ({ fields = [] } = {}) =>
  ({ node: el("div", { cls: "cx-field-row", children: fields.map((f) => nodeOf(f, "field.row")) }),
     destroy() { this.node.remove(); } }));

/** Title and hint on the left, control on the right. The settings-window row. */
define("settingsRow", "default", ({ label, hint, control, danger } = {}) => {
  const inner = nodeOf(control, "settingsRow.default");
  const hintId = hint ? uid("hint") : null;
  if (inner && hintId) inner.setAttribute("aria-describedby", hintId);

  const node = el("div", { cls: ["cx-settings-row", danger ? "cx-danger" : null], children: [
    el("div", { cls: "cx-settings-text", children: [
      el("div", { cls: "cx-settings-label", text: label }),
      hint ? el("p", { cls: "cx-hint", text: hint, attrs: { id: hintId } }) : null,
    ].filter(Boolean) }),
    inner ? el("div", { cls: "cx-settings-control", children: inner }) : null,
  ].filter(Boolean) });

  return { node, destroy: () => node.remove() };
});

define("group", "default", ({ label, rows = [], hint } = {}) => {
  const node = el("section", { cls: "cx-group", children: [
    label ? el("div", { cls: "cx-eyebrow", text: label }) : null,
    hint ? el("p", { cls: "cx-hint", text: hint }) : null,
    el("div", { cls: "cx-group-rows", children: rows.map((r) => nodeOf(r, "group.default")) }),
  ].filter(Boolean) });
  return { node, destroy: () => node.remove() };
});

define("panel", "default", ({ title, actions = [], body } = {}) => {
  const head = (title || actions.length)
    ? el("header", { cls: "cx-panel-head", children: [
        title ? el("h2", { cls: "cx-panel-title", text: title }) : null,
        actions.length ? el("div", { cls: "cx-panel-actions", children: actions.map((a) => nodeOf(a, "panel.default")) }) : null,
      ].filter(Boolean) })
    : null;
  const content = el("div", { cls: "cx-panel-body", children: body ? nodeOf(body, "panel.default") : null });
  const node = el("section", { cls: "cx-panel", children: [head, content].filter(Boolean) });
  return { node, body: content, destroy: () => node.remove() };
});

define("toolbar", "default", ({ items = [], label } = {}) =>
  ({ node: el("div", { cls: "cx-toolbar", attrs: { role: "toolbar", "aria-label": label },
      children: items.map((i) => nodeOf(i, "toolbar.default")) }),
     destroy() { this.node.remove(); } }));

define("actionBar", "sticky", ({ actions = [], note } = {}) =>
  ({ node: el("div", { cls: "cx-action-bar", children: [
      note ? el("p", { cls: "cx-hint", text: note }) : null,
      el("div", { cls: "cx-action-bar-buttons", children: actions.map((a) => nodeOf(a, "actionBar.sticky")) }),
    ].filter(Boolean) }),
     destroy() { this.node.remove(); } }));

/** A reorderable list of rows: LoRA stacks, text encoders, anything repeatable. */
define("list", "rows", ({ items = [], reorder = false, onRemove, onAdd, onReorder, addLabel = "Add", empty = "Nothing here yet" } = {}) => {
  const body = el("div", { cls: "cx-list" });

  function draw() {
    body.replaceChildren();
    if (!items.length) {
      body.append(el("p", { cls: "cx-list-empty", text: empty }));
    }
    items.forEach((item, index) => {
      const controls = [];
      if (reorder) {
        // "Move up/down", not bare arrows: next to a numeric control, an
        // unlabelled ▲▼ pair reads as increment/decrement.
        controls.push(el("button", { cls: ["cx-icon-btn", "cx-icon-btn-sm", "cx-list-move", "cx-focusable"], text: "▲",
          attrs: { type: "button", title: `Move ${item.label} up`,
                   "aria-label": `Move ${item.label} up`, disabled: index === 0 },
          on: { click: () => move(index, index - 1) } }));
        controls.push(el("button", { cls: ["cx-icon-btn", "cx-icon-btn-sm", "cx-list-move", "cx-focusable"], text: "▼",
          attrs: { type: "button", title: `Move ${item.label} down`,
                   "aria-label": `Move ${item.label} down`, disabled: index === items.length - 1 },
          on: { click: () => move(index, index + 1) } }));
      }
      if (onRemove) {
        controls.push(el("button", { cls: ["cx-icon-btn", "cx-icon-btn-sm", "cx-focusable"], text: "✕",
          attrs: { type: "button", "aria-label": `Remove ${item.label}` },
          on: { click: () => onRemove(item, index) } }));
      }
      // A row's control is a real element -- a LoRA's weight is edited here,
      // not implied by a number printed beside the reorder arrows, which reads
      // as a stepper for that number and is not one.
      const control = item.control ? nodeOf(item.control, "list.rows item.control") : null;

      body.append(el("div", { cls: "cx-list-row", children: [
        el("span", { cls: "cx-list-label", text: item.label }),
        item.hint ? el("span", { cls: "cx-list-hint", text: item.hint }) : null,
        control ? el("div", { cls: "cx-list-control", children: control }) : null,
        controls.length ? el("div", { cls: "cx-list-actions", children: controls }) : null,
      ].filter(Boolean) }));
    });
  }

  function move(from, to) {
    if (to < 0 || to >= items.length) return;
    const [moved] = items.splice(from, 1);
    items.splice(to, 0, moved);
    draw();
    if (onReorder) onReorder([...items]);
  }

  draw();

  const foot = onAdd
    ? el("button", { cls: ["cx-btn", "cx-btn-sm", "cx-btn-ghost", "cx-focusable"], text: `＋ ${addLabel}`,
        attrs: { type: "button" }, on: { click: onAdd } })
    : null;

  const node = el("div", { cls: "cx-list-wrap", children: [body, foot].filter(Boolean) });
  return { node, redraw: draw, destroy: () => node.remove() };
});

for (const kind of ["underline", "dock"]) {
  define("tabs", kind, ({ tabs = [], value, onChange, label } = {}) => {
    let current = value ?? tabs[0]?.value;
    const node = el("div", { cls: ["cx-tabs", `cx-tabs-${kind}`], attrs: { role: "tablist", "aria-label": label } });

    const buttons = tabs.map((tab) => {
      const b = el("button", { cls: ["cx-tab", "cx-focusable"], text: tab.label,
        attrs: { type: "button", role: "tab", "aria-selected": String(tab.value === current) } });
      b.classList.toggle("cx-on", tab.value === current);
      b.addEventListener("click", () => { current = tab.value; sync(); if (onChange) onChange(current); });
      node.append(b);
      return { tab, b };
    });

    function sync() {
      for (const { tab, b } of buttons) {
        const on = tab.value === current;
        b.classList.toggle("cx-on", on);
        b.setAttribute("aria-selected", String(on));
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

for (const axis of ["h", "v"]) {
  define("splitPane", axis, ({ panes = [], size = 50, min = 10, onResize, label } = {}) => {
    const horizontal = axis === "h";
    const node = el("div", { cls: ["cx-split", `cx-split-${axis}`] });
    const first = el("div", { cls: "cx-split-pane", children: nodeOf(panes[0], "splitPane") });
    const second = el("div", { cls: "cx-split-pane", children: nodeOf(panes[1], "splitPane") });
    const bar = el("div", { cls: "cx-split-bar",
      attrs: { role: "separator", tabindex: "0", "aria-label": label || "Resize",
               "aria-orientation": horizontal ? "vertical" : "horizontal",
               "aria-valuenow": String(size), "aria-valuemin": String(min), "aria-valuemax": String(100 - min) } });

    let current = size;
    const apply = (value) => {
      current = Math.min(100 - min, Math.max(min, value));
      first.style.flexBasis = `${current}%`;
      bar.setAttribute("aria-valuenow", String(Math.round(current)));
      if (onResize) onResize(current);
    };
    apply(size);

    let start = 0;
    const dispose = drag(bar, {
      onStart: () => { start = current; },
      onMove: ({ dx, dy }) => {
        const box = node.getBoundingClientRect();
        const span = horizontal ? box.width : box.height;
        if (!span) return;
        apply(start + ((horizontal ? dx : dy) / span) * 100);
      },
    });

    // A splitter that only responds to a mouse is unusable without one.
    bar.addEventListener("keydown", (e) => {
      const step = e.shiftKey ? 10 : 2;
      if (e.key === (horizontal ? "ArrowLeft" : "ArrowUp")) { e.preventDefault(); apply(current - step); }
      if (e.key === (horizontal ? "ArrowRight" : "ArrowDown")) { e.preventDefault(); apply(current + step); }
    });

    node.append(first, bar, second);
    return {
      node,
      get value() { return current; },
      setValue: apply,
      destroy() { dispose(); node.remove(); },
    };
  });
}

define("collapsible", "default", ({ label, open = false, body, hint } = {}) => {
  const node = el("details", { cls: "cx-collapsible", attrs: { open: open || undefined } });
  const summary = el("summary", { cls: ["cx-collapsible-head", "cx-focusable"], children: [
    el("span", { cls: "cx-collapsible-label", text: label }),
    hint ? el("span", { cls: "cx-hint", text: hint }) : null,
  ].filter(Boolean) });
  node.append(summary, el("div", { cls: "cx-collapsible-body", children: body ? nodeOf(body, "collapsible.default") : null }));
  return {
    node,
    get value() { return node.open; },
    setValue(v) { node.open = Boolean(v); },
    destroy: () => node.remove(),
  };
});

define("sidebar", "rail", ({ items = [], value, onChange, label } = {}) => {
  let current = value ?? items[0]?.value;
  const node = el("nav", { cls: "cx-rail", attrs: { "aria-label": label } });
  const buttons = items.map((item) => {
    const b = el("button", { cls: ["cx-rail-item", "cx-focusable"], attrs: { type: "button" }, children: [
      el("span", { cls: "cx-rail-icon", text: item.icon, attrs: { "aria-hidden": "true" } }),
      el("span", { cls: "cx-rail-label", text: item.label }),
    ] });
    b.classList.toggle("cx-on", item.value === current);
    b.addEventListener("click", () => {
      current = item.value;
      for (const { item: i, b: other } of buttons) other.classList.toggle("cx-on", i.value === current);
      if (onChange) onChange(current);
    });
    node.append(b);
    return { item, b };
  });
  return { node, get value() { return current; }, destroy: () => node.remove() };
});
