// Anchored overlays: popovers, menus, tooltips, autocomplete.
//
// v4 had seven copies of measure-flip-clamp-and-dismiss, one per feature. This
// is the one, and everything anchored is built on it.

import { define } from "../internals/register.js";
import { el } from "../internals/el.js";
import { mount, unmount } from "../internals/portal.js";
import { claim } from "../internals/zlayer.js";
import { push } from "../internals/dismiss.js";
import { anchorTo } from "../internals/anchor.js";

function nodeOf(control, where) {
  if (!control) return null;
  if (control.node && control.node.nodeType === 1) return control.node;
  throw new TypeError(`${where} takes a Composer element handle, not a ${typeof control}.`);
}

function floatOn(rung, cls, { anchor, side = "bottom", align = "start", gap = 6, onClose, closeOnOutside = true, content, role, label }) {
  const anchorNode = anchor && anchor.node ? anchor.node : anchor;
  const layer = claim(rung);
  const node = el("div", { cls, attrs: { role, "aria-label": label } });
  node.style.zIndex = String(layer.z);
  if (content) node.append(content);

  // Mounted before measuring: a hidden element measures 0x0 and lands in the
  // corner, which is the classic "my menu appears top-left" bug.
  mount(node);
  if (anchorNode) anchorTo(node, anchorNode, { side, align, gap });

  const reposition = () => { if (anchorNode) anchorTo(node, anchorNode, { side, align, gap }); };
  window.addEventListener("resize", reposition);
  // Capture, because the scroller is usually an ancestor panel, not the window.
  window.addEventListener("scroll", reposition, true);

  const dismissal = push({
    nodes: [node, anchorNode].filter(Boolean),
    closeOnOutside,
    onDismiss: (reason) => handle.close(reason),
  });

  let closed = false;
  const handle = {
    node,
    isOverlay: true,
    reposition,
    close(reason = "manual") {
      if (closed) return;
      closed = true;
      dismissal.release();
      window.removeEventListener("resize", reposition);
      window.removeEventListener("scroll", reposition, true);
      layer.release();
      unmount(node);
      if (onClose) onClose(reason);
    },
    destroy() { handle.close("destroy"); },
  };
  return handle;
}

define("popover", "anchored", ({ body, ...rest } = {}) =>
  floatOn("popover", "cx-popover", { ...rest, content: nodeOf(body, "popover.anchored"), role: "dialog" }));

function menuList(items, onPick, close) {
  const list = el("div", { cls: "cx-menu", attrs: { role: "menu" } });
  for (const item of items) {
    if (item.separator) { list.append(el("div", { cls: "cx-menu-sep", attrs: { role: "separator" } })); continue; }
    list.append(el("button", {
      cls: ["cx-menu-item", "cx-focusable", item.danger ? "cx-danger" : null],
      attrs: { type: "button", role: "menuitem", disabled: item.disabled },
      children: [
        item.icon ? el("span", { cls: "cx-menu-icon", text: item.icon, attrs: { "aria-hidden": "true" } }) : null,
        el("span", { cls: "cx-menu-label", text: item.label }),
        item.hint ? el("span", { cls: "cx-menu-hint", text: item.hint }) : null,
      ].filter(Boolean),
      on: { click: () => { if (onPick) onPick(item.id); close(); } },
    }));
  }
  return list;
}

define("menu", "dropdown", ({ items = [], onPick, ...rest } = {}) => {
  const handle = floatOn("popover", "cx-popover cx-popover-menu", {
    ...rest, role: "menu",
    content: null,
  });
  handle.node.append(menuList(items, onPick, () => handle.close("picked")));
  handle.reposition();
  return handle;
});

/** Right-click menu: anchored to a point rather than an element. */
define("menu", "context", ({ x = 0, y = 0, items = [], onPick, onClose } = {}) => {
  const point = { getBoundingClientRect: () => ({ x, y, width: 0, height: 0, top: y, left: x, right: x, bottom: y }) };
  const handle = floatOn("popover", "cx-popover cx-popover-menu", {
    anchor: point, onClose, role: "menu",
  });
  handle.node.append(menuList(items, onPick, () => handle.close("picked")));
  handle.reposition();
  return handle;
});

/**
 * A real tooltip. v4 used the native title attribute everywhere, which cannot be
 * styled, cannot be triggered by keyboard focus, and waits about a second.
 */
define("tooltip", "default", ({ anchor, text, side = "top", trigger = true } = {}) => {
  const anchorNode = anchor && anchor.node ? anchor.node : anchor;
  let live = null;

  const show = () => {
    if (live) return;
    live = floatOn("popover", "cx-tooltip", {
      anchor: anchorNode, side, align: "center", gap: 6, closeOnOutside: false,
      content: el("span", { text }), role: "tooltip",
    });
  };
  const hide = () => { if (live) { live.close("hide"); live = null; } };

  if (trigger && anchorNode) {
    anchorNode.addEventListener("pointerenter", show);
    anchorNode.addEventListener("pointerleave", hide);
    anchorNode.addEventListener("focus", show);
    anchorNode.addEventListener("blur", hide);
  }

  return {
    node: anchorNode || el("span"),
    isOverlay: true,
    show, hide,
    destroy() {
      hide();
      if (trigger && anchorNode) {
        anchorNode.removeEventListener("pointerenter", show);
        anchorNode.removeEventListener("pointerleave", hide);
        anchorNode.removeEventListener("focus", show);
        anchorNode.removeEventListener("blur", hide);
      }
    },
  };
});

/**
 * Completion under a field. This is the one overlay that must outrank modals --
 * it opens from a field that is usually inside one.
 */
define("autocomplete", "default", ({ input, source, onPick, minChars = 1 } = {}) => {
  const field = input && input.node ? input.node : input;
  let live = null;
  let index = -1;
  let current = [];

  const close = () => { if (live) { live.close("hide"); live = null; } index = -1; };

  const draw = (items) => {
    current = items;
    if (!items.length) { close(); return; }
    if (!live) {
      live = floatOn("autocomplete", "cx-popover cx-autocomplete", {
        anchor: field, side: "bottom", align: "start", gap: 2,
        role: "listbox", onClose: () => { live = null; },
      });
    }
    live.node.replaceChildren();
    items.forEach((item, i) => {
      live.node.append(el("button", {
        cls: ["cx-menu-item", i === index ? "cx-on" : null],
        attrs: { type: "button", role: "option", "aria-selected": String(i === index) },
        children: [
          el("span", { cls: "cx-menu-label", text: item.label }),
          item.hint ? el("span", { cls: "cx-menu-hint", text: item.hint }) : null,
        ].filter(Boolean),
        on: { click: () => { if (onPick) onPick(item); close(); } },
      }));
    });
    live.reposition();
  };

  const onInput = () => {
    const q = field.value || "";
    if (q.length < minChars) { close(); return; }
    index = -1;
    draw(source ? source(q) : []);
  };

  const onKeyDown = (e) => {
    if (!live) return;
    if (e.key === "ArrowDown") { e.preventDefault(); index = Math.min(index + 1, current.length - 1); draw(current); }
    else if (e.key === "ArrowUp") { e.preventDefault(); index = Math.max(index - 1, 0); draw(current); }
    else if (e.key === "Enter" && index >= 0) { e.preventDefault(); if (onPick) onPick(current[index]); close(); }
  };

  if (field) {
    field.addEventListener("input", onInput);
    field.addEventListener("keydown", onKeyDown);
  }

  return {
    node: field || el("span"),
    isOverlay: true,
    close,
    destroy() {
      close();
      if (field) {
        field.removeEventListener("input", onInput);
        field.removeEventListener("keydown", onKeyDown);
      }
    },
  };
});

/** A button whose second half opens a menu. */
define("splitButton", "md", ({ label, onClick, items = [], onPick, tone = "neutral" } = {}) => {
  const main = el("button", { cls: ["cx-btn", "cx-btn-md", `cx-btn-${tone}`, "cx-focusable"],
    text: label, attrs: { type: "button" }, on: { click: onClick } });
  const arrow = el("button", { cls: ["cx-btn", "cx-btn-md", `cx-btn-${tone}`, "cx-split-arrow", "cx-focusable"],
    text: "▾", attrs: { type: "button", "aria-label": `${label} options`, "aria-haspopup": "menu" } });

  let menu = null;
  arrow.addEventListener("click", () => {
    if (menu) { menu.close("toggle"); menu = null; return; }
    menu = floatOn("popover", "cx-popover cx-popover-menu", {
      anchor: arrow, side: "bottom", align: "end", onClose: () => { menu = null; },
    });
    menu.node.append(menuList(items, onPick, () => { if (menu) menu.close("picked"); }));
    menu.reposition();
  });

  const node = el("div", { cls: "cx-split-btn", children: [main, arrow] });
  return { node, destroy() { if (menu) menu.close("destroy"); node.remove(); } };
});
