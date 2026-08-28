// Modals.
//
// v4 built this shell by hand in eleven files, each repeating its own backdrop
// click, close button and Escape handling -- and getting Escape wrong when one
// opened inside another. Here the shell is one thing, and the stacking and
// dismissal are the internals' problem.
//
// Every overlay handle carries isOverlay, because it mounts itself into the
// portal instead of being appended by its caller.

import { define } from "../internals/register.js";
import { el } from "../internals/el.js";
import { uid } from "../internals/ids.js";
import { mount, unmount } from "../internals/portal.js";
import { claim } from "../internals/zlayer.js";
import { push } from "../internals/dismiss.js";
import { trap } from "../internals/focus.js";

const SIZES = { sm: "360px", md: "520px", lg: "720px", xl: "960px" };

function nodeOf(control, where) {
  if (!control) return null;
  if (control.node && control.node.nodeType === 1) return control.node;
  throw new TypeError(`${where} takes a Composer element handle, not a ${typeof control}.`);
}

/**
 * open({ title, subtitle, size, body, actions, onClose, closeOnOutside, stacked })
 * Shared by every modal variant below.
 */
function open({
  title, subtitle, size = "md", body, actions = [], onClose,
  closeOnOutside = true, stacked = false, describedBy,
} = {}) {
  const titleId = uid("modal-title");
  const layer = claim(stacked ? "modal" : "modal");

  const card = el("div", {
    cls: ["cx-modal", `cx-modal-${size}`, stacked ? "cx-modal-stacked" : null],
    attrs: { role: "dialog", "aria-modal": "true", "aria-labelledby": titleId,
             "aria-describedby": describedBy },
  });
  card.style.maxWidth = SIZES[size] || SIZES.md;

  const close = el("button", {
    cls: ["cx-icon-btn", "cx-icon-btn-md", "cx-focusable"],
    text: "✕", attrs: { type: "button", "aria-label": "Close" },
    on: { click: () => handle.close("close-button") },
  });

  card.append(el("header", { cls: "cx-modal-head", children: [
    el("div", { cls: "cx-modal-titles", children: [
      el("h2", { cls: "cx-modal-title", text: title, attrs: { id: titleId } }),
      subtitle ? el("p", { cls: "cx-hint", text: subtitle }) : null,
    ].filter(Boolean) }),
    close,
  ] }));

  const content = el("div", { cls: "cx-modal-body", children: nodeOf(body, "modal") });
  card.append(content);

  if (actions.length) {
    card.append(el("footer", { cls: "cx-modal-foot",
      children: actions.map((a) => nodeOf(a, "modal actions")) }));
  }

  const backdrop = el("div", { cls: "cx-backdrop" });
  const root = el("div", { cls: "cx-modal-layer", children: [backdrop, card] });
  root.style.zIndex = String(layer.z);
  mount(root);

  // Trap after mounting: focusables() must be able to see the card.
  const release = trap(card);
  const dismissal = push({
    nodes: card,
    closeOnOutside,
    onDismiss: (reason) => handle.close(reason),
  });

  let closed = false;
  const handle = {
    node: card,
    isOverlay: true,
    close(reason = "manual") {
      if (closed) return;
      closed = true;
      dismissal.release();
      release();
      layer.release();
      unmount(root);
      if (onClose) onClose(reason);
    },
    destroy() { handle.close("destroy"); },
  };
  return handle;
}

define("modal", "generic", (props) => open(props));

// Opened above an existing modal without closing it. The z rung and the focus
// handoff are the kit's problem; v4 made each caller compute a z-index from its
// parent's.
define("modal", "stacked", (props) => open({ ...props, stacked: true }));

/**
 * A yes/no question. `result` resolves true or false -- including on Escape and
 * on a backdrop click, which both mean "no" rather than leaving the promise
 * pending forever.
 */
define("modal", "dialogue", ({ title, message, tone = "neutral", confirmLabel = "Confirm", cancelLabel = "Cancel", onResult } = {}) => {
  let settle;
  const result = new Promise((resolve) => { settle = resolve; });
  const finish = (value) => { settle(value); if (onResult) onResult(value); };

  const bodyId = uid("modal-msg");
  const handle = open({
    title,
    describedBy: bodyId,
    body: { node: el("p", { cls: "cx-t cx-t-sm", text: message, attrs: { id: bodyId } }) },
    onClose: (reason) => { if (reason !== "confirmed") finish(false); },
    actions: [
      { node: el("button", { cls: ["cx-btn", "cx-btn-lg", "cx-btn-neutral", "cx-focusable"],
          text: cancelLabel, attrs: { type: "button" },
          on: { click: () => handle.close("cancelled") } }) },
      { node: el("button", { cls: ["cx-btn", "cx-btn-lg", tone === "danger" ? "cx-btn-danger" : "cx-btn-primary", "cx-focusable"],
          text: confirmLabel, attrs: { type: "button" },
          on: { click: () => { finish(true); handle.close("confirmed"); } } }) },
    ],
  });
  handle.result = result;
  return handle;
});

/** Ask for one value. Resolves the string, or null if dismissed. */
define("modal", "prompt", ({ title, label, value = "", placeholder, confirmLabel = "Create", cancelLabel = "Cancel", validate, onResult } = {}) => {
  let settle;
  const result = new Promise((resolve) => { settle = resolve; });
  const finish = (v) => { settle(v); if (onResult) onResult(v); };

  const input = el("input", { cls: ["cx-input", "cx-input-md", "cx-focusable"],
    attrs: { type: "text", placeholder, id: uid("prompt") } });
  input.value = value;
  const error = el("p", { cls: "cx-field-error" });

  const submit = () => {
    const problem = validate ? validate(input.value) : null;
    if (problem) { error.textContent = problem; input.focus(); return; }
    finish(input.value);
    handle.close("confirmed");
  };

  input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); submit(); } });

  const handle = open({
    title,
    body: { node: el("div", { cls: "cx-field", children: [
      label ? el("label", { cls: "cx-label", text: label, attrs: { for: input.id } }) : null,
      input, error,
    ].filter(Boolean) }) },
    onClose: (reason) => { if (reason !== "confirmed") finish(null); },
    actions: [
      { node: el("button", { cls: ["cx-btn", "cx-btn-lg", "cx-btn-neutral", "cx-focusable"],
          text: cancelLabel, attrs: { type: "button" }, on: { click: () => handle.close("cancelled") } }) },
      { node: el("button", { cls: ["cx-btn", "cx-btn-lg", "cx-btn-primary", "cx-focusable"],
          text: confirmLabel, attrs: { type: "button" }, on: { click: submit } }) },
    ],
  });
  handle.result = result;
  queueMicrotask(() => { input.focus(); input.select(); });
  return handle;
});

/** Pick one of a list. Resolves the chosen id, or null. */
define("modal", "choice", ({ title, subtitle, items = [], onPick } = {}) => {
  let settle;
  const result = new Promise((resolve) => { settle = resolve; });

  const list = el("div", { cls: "cx-choice-list" });
  for (const item of items) {
    list.append(el("button", {
      cls: ["cx-choice-row", "cx-focusable"],
      attrs: { type: "button", disabled: item.disabled },
      children: [
        el("span", { cls: "cx-choice-label", text: item.label }),
        item.hint ? el("span", { cls: "cx-hint", text: item.hint }) : null,
      ].filter(Boolean),
      on: { click: () => { settle(item.id); if (onPick) onPick(item.id); handle.close("picked"); } },
    }));
  }

  const handle = open({
    title, subtitle, size: "sm",
    body: { node: list },
    onClose: (reason) => { if (reason !== "picked") settle(null); },
  });
  handle.result = result;
  return handle;
});
