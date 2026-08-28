// Telling the user what is going on.
//
// v4 had roughly fifteen badge classes, ten empty states and three spinners.
// Here there are five tones and one shape of each, which is the whole point: a
// status looks the same everywhere, so its colour actually means something.

import { define } from "../internals/register.js";
import { el } from "../internals/el.js";
import { setText } from "../internals/text.js";
import { portal } from "../internals/portal.js";
import { claimFor } from "../internals/zlayer.js";

const TONES = ["neutral", "good", "warn", "danger", "info"];

for (const tone of TONES) {
  define("chip", tone, ({ label, dot = false } = {}) =>
    ({ node: el("span", { cls: ["cx-chip", `cx-chip-${tone}`], children: [
        dot ? el("span", { cls: "cx-chip-dot", attrs: { "aria-hidden": "true" } }) : null,
        el("span", { text: label }),
      ].filter(Boolean) }),
       destroy() { this.node.remove(); } }));
}

define("dot", "default", ({ tone = "neutral", label } = {}) =>
  ({ node: el("span", { cls: ["cx-dot", `cx-dot-${tone}`],
      attrs: { role: label ? "img" : undefined, "aria-label": label, title: label } }),
     destroy() { this.node.remove(); } }));

for (const tone of ["info", "warn", "danger"]) {
  define("banner", tone, ({ text, action } = {}) =>
    ({ node: el("div", { cls: ["cx-banner", `cx-banner-${tone}`],
        attrs: { role: tone === "danger" ? "alert" : "status" }, children: [
          el("span", { cls: "cx-dot", attrs: { "aria-hidden": "true" } }),
          el("span", { cls: "cx-banner-text", text }),
          action && action.node ? action.node : null,
        ].filter(Boolean) }),
       destroy() { this.node.remove(); } }));
}

define("inlineError", "default", ({ text } = {}) =>
  ({ node: el("p", { cls: "cx-inline-error", text, attrs: { role: "alert" } }),
     destroy() { this.node.remove(); } }));

define("progress", "bar", ({ value = 0, max = 100, label } = {}) => {
  const fill = el("div", { cls: "cx-progress-fill" });
  const node = el("div", { cls: "cx-progress",
    attrs: { role: "progressbar", "aria-valuemin": "0", "aria-valuemax": String(max),
             "aria-valuenow": String(value), "aria-label": label },
    children: fill });
  const set = (v) => {
    const pct = max ? Math.min(100, Math.max(0, (v / max) * 100)) : 0;
    fill.style.width = `${pct}%`;
    node.setAttribute("aria-valuenow", String(v));
  };
  set(value);
  return { node, setValue: set, destroy: () => node.remove() };
});

// For work whose total is genuinely unknown. A bar that invents a percentage is
// worse than one that admits it does not know.
define("progress", "indeterminate", ({ label } = {}) =>
  ({ node: el("div", { cls: ["cx-progress", "cx-progress-indeterminate"],
      attrs: { role: "progressbar", "aria-label": label },
      children: el("div", { cls: "cx-progress-fill" }) }),
     destroy() { this.node.remove(); } }));

for (const size of ["md", "sm"]) {
  define("spinner", size, ({ label = "Working" } = {}) =>
    ({ node: el("span", { cls: ["cx-spinner", `cx-spinner-${size}`],
        attrs: { role: "status", "aria-label": label } }),
       destroy() { this.node.remove(); } }));
}

define("emptyState", "default", ({ icon, title, hint, action } = {}) =>
  ({ node: el("div", { cls: "cx-empty", children: [
      icon ? el("div", { cls: "cx-empty-icon", text: icon, attrs: { "aria-hidden": "true" } }) : null,
      el("p", { cls: "cx-empty-title", text: title }),
      hint ? el("p", { cls: "cx-hint", text: hint }) : null,
      action && action.node ? action.node : null,
    ].filter(Boolean) }),
     destroy() { this.node.remove(); } }));

define("dropzone", "default", ({ label, hint, accept, onFiles, multiple = true } = {}) => {
  const input = el("input", { cls: "cx-sr",
    attrs: { type: "file", accept, multiple: multiple || undefined } });
  const node = el("label", { cls: "cx-dropzone", children: [
    input,
    el("span", { cls: "cx-dropzone-label", text: label }),
    hint ? el("span", { cls: "cx-hint", text: hint }) : null,
  ].filter(Boolean) });

  const emit = (files) => { if (onFiles && files && files.length) onFiles([...files]); };
  input.addEventListener("change", () => emit(input.files));

  // dragover must be prevented or the browser navigates to the dropped file,
  // losing whatever was on screen.
  for (const type of ["dragenter", "dragover"]) {
    node.addEventListener(type, (e) => { e.preventDefault(); node.classList.add("cx-over"); });
  }
  for (const type of ["dragleave", "drop"]) {
    node.addEventListener(type, (e) => { e.preventDefault(); node.classList.remove("cx-over"); });
  }
  node.addEventListener("drop", (e) => emit(e.dataTransfer && e.dataTransfer.files));

  return { node, destroy: () => node.remove() };
});

for (const shape of ["line", "block", "grid"]) {
  define("skeleton", shape, ({ count = shape === "grid" ? 6 : 3, label = "Loading" } = {}) => {
    const node = el("div", { cls: ["cx-skeleton", `cx-skeleton-${shape}`],
      attrs: { role: "status", "aria-label": label } });
    for (let i = 0; i < count; i += 1) node.append(el("span", { cls: "cx-skeleton-cell" }));
    return { node, destroy: () => node.remove() };
  });
}

// One toast stack, owned by the kit. Several stacks fighting for the corner is
// how v4's tour toast ended up as the only one anybody could use.
let stack = null;
let stackLayer = null;

function toastStack() {
  if (stack && stack.isConnected) return stack;
  stack = el("div", { cls: "cx-toasts" });
  stackLayer = claimFor("toast", stack);
  stack.style.pointerEvents = "auto";
  portal().appendChild(stack);
  return stack;
}

for (const tone of ["info", "good", "warn", "danger"]) {
  define("toast", tone, ({ text, action, duration = 3200 } = {}) => {
    const node = el("div", { cls: ["cx-toast", `cx-toast-${tone}`],
      attrs: { role: tone === "danger" ? "alert" : "status" }, children: [
        el("span", { cls: "cx-toast-text", text }),
        action && action.node ? action.node : null,
      ].filter(Boolean) });

    toastStack().appendChild(node);
    let timer = duration ? setTimeout(() => handle.destroy(), duration) : null;

    // Hovering pauses the countdown: a toast that vanishes while being read is
    // a toast that failed to say anything.
    node.addEventListener("pointerenter", () => { if (timer) { clearTimeout(timer); timer = null; } });
    node.addEventListener("pointerleave", () => {
      if (duration && !timer) timer = setTimeout(() => handle.destroy(), duration);
    });

    const handle = {
      node,
      setText: (value) => setText(node.querySelector(".cx-toast-text"), value),
      destroy() {
        if (timer) clearTimeout(timer);
        node.remove();
        if (stack && !stack.childElementCount) {
          stackLayer.release();
          stack.remove();
          stack = null;
        }
      },
    };
    return handle;
  });
}

export function _resetToasts() {
  if (stack) { stack.remove(); stack = null; }
  if (stackLayer) { stackLayer.release(); stackLayer = null; }
}
