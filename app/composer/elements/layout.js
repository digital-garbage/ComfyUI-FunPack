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

/**
 * A titled box. Two of them, and the difference is only where they are used.
 *
 * `default` is a card: rounded, bordered, with a shadow, for a panel that
 * floats on something -- inside a window, on a page.
 *
 * `zone` is a REGION OF THE APP: square, edge to edge, divided from its
 * neighbours by the same hairline everything else uses. The app is one surface
 * cut into areas, not a tray of cards with the desk showing between them --
 * which is what an editing suite looks like, and what v4 was.
 */
const panel = (variant) => ({ title, actions = [], body, flush = false } = {}) => {
  const head = (title || actions.length)
    ? el("header", { cls: "cx-panel-head", children: [
        title ? el("h2", { cls: "cx-panel-title", text: title }) : null,
        actions.length ? el("div", { cls: "cx-panel-actions", children: actions.map((a) => nodeOf(a, `panel.${variant}`)) }) : null,
      ].filter(Boolean) })
    : null;
  const content = el("div", { cls: "cx-panel-body", children: body ? nodeOf(body, `panel.${variant}`) : null });
  // `flush` is for a body that IS the content -- a picture, a video, a canvas.
  // Padding around one of those is a frame nobody asked for, and the thing
  // being looked at is the reason the zone exists.
  const node = el("section", { cls: ["cx-panel", variant === "zone" ? "cx-zone" : null,
                                     flush ? "cx-panel-flush" : null],
    children: [head, content].filter(Boolean) });
  return { node, body: content, destroy: () => node.remove() };
};

define("panel", "default", panel("default"));
define("panel", "zone", panel("zone"));

// A region whose contents are replaced.
//
// Every part of the app that redraws needs one host it can empty and refill.
// Without it the shell reaches for the document to make a div of its own, and a
// div the shell made is a div the shell has to lay out -- which is the one thing
// that is supposed to live only here.
define("region", "stack", ({ children = [], gap = "md", label, fill = false } = {}) => {
  const node = el("div", { cls: ["cx-stack", `cx-stack-${gap}`, fill ? "cx-stack-fill" : null],
    attrs: { "aria-label": label } });
  const set = (next = []) => {
    // Handles, not nodes, like everywhere else -- so a caller cannot slip a
    // document node in through the one element that takes children late.
    node.replaceChildren(...next.filter(Boolean).map((c) => nodeOf(c, "region.stack")));
  };
  set(children);
  return { node, set, destroy: () => node.remove() };
});

define("toolbar", "default", ({ items = [], label } = {}) =>
  ({ node: el("div", { cls: "cx-toolbar", attrs: { role: "toolbar", "aria-label": label },
      children: items.map((i) => nodeOf(i, "toolbar.default")) }),
     destroy() { this.node.remove(); } }));

/**
 * The row a thing is started from: what is being SAID at the start of it, what
 * is PRESSED at the end.
 *
 * `lead` exists because everything used to go in `actions`, which is
 * right-aligned as a group -- so a status line sat against the button with the
 * whole width of the bar empty to its left, and read as part of the button.
 */
define("actionBar", "sticky", ({ lead = [], actions = [], note } = {}) =>
  ({ node: el("div", { cls: "cx-action-bar", children: [
      el("div", { cls: "cx-action-bar-lead", children: [
        note ? el("p", { cls: "cx-hint", text: note }) : null,
        ...lead.map((a) => nodeOf(a, "actionBar.sticky lead")),
      ].filter(Boolean) }),
      el("div", { cls: "cx-action-bar-buttons", children: actions.map((a) => nodeOf(a, "actionBar.sticky")) }),
    ] }),
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

// The page: one region that fills whatever is left, and a bar that does not.
//
// Height has to be established SOMEWHERE, and the honest place is one element
// rather than a percentage on each pane hoping its parent is definite. Without
// it a vertical split resolves its basis against content, and the app renders
// as a short strip at the top of an empty window -- which is exactly what the
// first build of this frame did.
define("frame", "app", ({ header, main, footer } = {}) => {
  const head = header
    ? el("div", { cls: "cx-frame-head", children: nodeOf(header, "frame.app") })
    : null;
  const body = el("div", { cls: "cx-frame-main",
    children: main ? nodeOf(main, "frame.app") : null });
  const foot = footer
    ? el("div", { cls: "cx-frame-foot", children: nodeOf(footer, "frame.app") })
    : null;
  const node = el("div", { cls: "cx-frame", children: [head, body, foot].filter(Boolean) });
  return { node, header: head, main: body, footer: foot, destroy: () => node.remove() };
});

// The app's own shape: a centre that shrinks, with panels docked either side.
//
// The toggles live in RAILS, outside the panels they open. v4 put the control
// inside the region it controlled and then hid the region -- a collapsed column
// was marked [hidden], which is display:none, and no transform slides a
// display:none element into view, so the button did nothing at all and said
// nothing. A control that can be hidden by the thing it operates is not a
// control, so here the rail is a sibling of the panel and never moves.
define("workspace", "docked", ({
  id = "workspace", centre, left, right,
  leftLabel = "Assets", rightLabel = "Properties",
  leftOpen = true, rightOpen = true, onToggle,
} = {}) => {
  const node = el("div", { cls: "cx-workspace" });

  // window.localStorage, not the bare global: they are the same object in a
  // browser, and only the explicit one exists under jsdom -- so persistence is
  // testable where the tests are cheap instead of only in the browser pass.
  const remember = (side, open) => {
    try { window.localStorage.setItem(`cx.ws.${id}.${side}`, open ? "1" : "0"); } catch { /* private mode */ }
  };
  const recall = (side, fallback) => {
    try {
      const v = window.localStorage.getItem(`cx.ws.${id}.${side}`);
      return v === null ? fallback : v === "1";
    } catch { return fallback; }
  };

  const state = { left: recall("left", leftOpen), right: recall("right", rightOpen) };
  // What the user last chose, kept apart from what is currently on screen.
  //
  // `state` is mutated by the one-at-a-time rule below, so it stops being a
  // record of anyone's preference the moment a window is narrow. Snapshotting
  // it there recorded a panel the RULE had just closed as a panel the user
  // wanted closed -- so a page loaded at a narrow width came back from being
  // widened with the right panel shut, and nothing the user did would explain
  // why.
  const wanted = { ...state };
  const panels = {};
  const toggles = {};

  // Below this the panels overlay the centre instead of docking beside it, so
  // only one may be open and neither starts open. The CSS decides how a panel
  // is drawn; this decides what is open, and the two have to agree on the
  // width, which is why it is written once here and matched there.
  const NARROW = "(max-width: 760px)";
  const narrow = typeof window.matchMedia === "function" ? window.matchMedia(NARROW) : null;
  const isNarrow = () => Boolean(narrow && narrow.matches);
  let wide = null;                    // what was open before the window shrank

  const side = (which, label, content) => {
    const panelId = uid(`ws-${which}`);
    const panel = el("div", { cls: ["cx-workspace-side", `cx-workspace-${which}`],
      attrs: { id: panelId, "aria-label": label },
      children: content ? nodeOf(content, "workspace.docked") : null });

    const button = el("button", { cls: ["cx-icon-btn", "cx-icon-btn-sm", "cx-focusable"],
      text: which === "left" ? "▎" : "▐",
      attrs: { type: "button", "aria-controls": panelId, "aria-label": label,
               "aria-expanded": String(state[which]) },
      on: { click: () => set(which, !state[which]) } });

    const rail = el("div", { cls: ["cx-workspace-rail", `cx-workspace-rail-${which}`],
      children: button });

    panels[which] = panel;
    toggles[which] = button;
    return { panel, rail };
  };

  function set(which, open, { remember: keep = true } = {}) {
    state[which] = Boolean(open);
    // One at a time while overlaid: two panels over a narrow centre is the same
    // problem the docking rule was avoiding, with an extra step.
    if (state[which] && isNarrow()) {
      const other = which === "left" ? "right" : "left";
      if (state[other]) set(other, false, { remember: false });
    }
    panels[which].classList.toggle("cx-collapsed", !state[which]);
    // Not [hidden] and not display:none: a panel with no width is out of the
    // way and still animatable, and its toggle is in the rail either way.
    panels[which].setAttribute("aria-hidden", String(!state[which]));
    toggles[which].setAttribute("aria-expanded", String(state[which]));
    // A window being small is not the user changing their mind, so an automatic
    // close does not overwrite what they last chose -- in storage or here.
    if (keep) {
      wanted[which] = state[which];
      remember(which, state[which]);
    }
    if (onToggle) onToggle(which, state[which]);
  }

  function fit() {
    if (isNarrow()) {
      if (wide === null) wide = { ...wanted };
      set("left", false, { remember: false });
      set("right", false, { remember: false });
    } else if (wide) {
      const was = wide;
      wide = null;
      set("left", was.left, { remember: false });
      set("right", was.right, { remember: false });
    }
  }

  const main = el("div", { cls: "cx-workspace-main",
    children: centre ? nodeOf(centre, "workspace.docked") : null });

  const l = side("left", leftLabel, left);
  const r = side("right", rightLabel, right);
  node.append(l.rail, l.panel, main, r.panel, r.rail);
  set("left", state.left, { remember: false });
  set("right", state.right, { remember: false });
  fit();
  if (narrow && narrow.addEventListener) narrow.addEventListener("change", fit);

  return {
    node,
    // Hosts, so the shell mounts INTO the regions rather than rebuilding them.
    get left() { return panels.left; },
    get right() { return panels.right; },
    get centre() { return main; },
    isOpen: (which) => Boolean(state[which]),
    open: (which) => set(which, true),
    close: (which) => set(which, false),
    toggle: (which) => set(which, !state[which]),
    narrow: isNarrow,
    destroy() {
      if (narrow && narrow.removeEventListener) narrow.removeEventListener("change", fit);
      node.remove();
    },
  };
});

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
