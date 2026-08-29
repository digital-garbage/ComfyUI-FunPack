// Windows and panels that live above the app.

import { define } from "../internals/register.js";
import { el } from "../internals/el.js";
import { uid } from "../internals/ids.js";
import { mount, unmount } from "../internals/portal.js";
import { claimFor } from "../internals/zlayer.js";
import { push } from "../internals/dismiss.js";
import { trap } from "../internals/focus.js";
import { drag } from "../internals/drag.js";

function nodeOf(control, where) {
  if (!control) return null;
  if (control.node && control.node.nodeType === 1) return control.node;
  throw new TypeError(`${where} takes a Composer element handle, not a ${typeof control}.`);
}

const remember = (id, box) => { try { localStorage.setItem(`cx.win.${id}`, JSON.stringify(box)); } catch { /* private mode */ } };
const recall = (id) => { try { return JSON.parse(localStorage.getItem(`cx.win.${id}`)) || null; } catch { return null; } };

/**
 * An OS-style window: drag by the titlebar, roll up, maximise, resize from the
 * corner, click to front. Position and size are remembered per `id`.
 */
define("floating", "window", ({
  id = uid("win"), title, subtitle, body,
  width = 420, height = 320, x = 80, y = 80,
  resizable = true, rollup = true, onClose,
} = {}) => {
  const saved = recall(id) || {};
  const box = { x: saved.x ?? x, y: saved.y ?? y, width: saved.width ?? width, height: saved.height ?? height };

  const titleId = uid("win-title");
  const node = el("div", { cls: "cx-window",
    attrs: { role: "dialog", "aria-labelledby": titleId } });
  // Bound, because another window coming to the front shifts this one's z.
  const layer = claimFor("floatingWindow", node);

  // Enough of the window to grab: a strip of titlebar wide enough to hit and
  // tall enough to see. Everything that positions the window goes through here,
  // because the one path that did not was the one that lost it.
  const GRABBABLE = 80;
  const TITLEBAR = 32;

  const clampIntoView = () => {
    // Math.max(0, ...) matters on a viewport narrower than the reachable strip:
    // without it the upper bound goes negative and the clamp pushes the window
    // further off rather than back on.
    box.x = Math.min(Math.max(box.x, -box.width + GRABBABLE),
                     Math.max(0, window.innerWidth - GRABBABLE));
    box.y = Math.min(Math.max(box.y, 0), Math.max(0, window.innerHeight - TITLEBAR));
  };

  const apply = () => {
    // Clamped on EVERY apply, not only while dragging. A position is remembered
    // per window id, so a window moved to the right of a wide screen and reopened
    // on a narrow one opened entirely off it -- no titlebar to grab, nothing to
    // drag back, and the only way out was clearing localStorage by hand.
    clampIntoView();
    node.style.left = `${box.x}px`;
    node.style.top = `${box.y}px`;
    node.style.width = `${box.width}px`;
    node.style.height = rolled ? "auto" : `${box.height}px`;
  };

  const titlebar = el("div", { cls: "cx-window-bar", children: [
    el("div", { cls: "cx-window-titles", children: [
      el("span", { cls: "cx-window-title", text: title, attrs: { id: titleId } }),
      subtitle ? el("span", { cls: "cx-hint", text: subtitle }) : null,
    ].filter(Boolean) }),
  ] });

  const controls = el("div", { cls: "cx-window-controls" });
  let rolled = false;
  let maximised = false;
  let restoreBox = null;

  if (rollup) {
    controls.append(el("button", { cls: ["cx-icon-btn", "cx-icon-btn-sm", "cx-focusable"], text: "—",
      attrs: { type: "button", "aria-label": "Roll up" },
      on: { click: () => { rolled = !rolled; node.classList.toggle("cx-rolled", rolled); apply(); } } }));
  }
  controls.append(el("button", { cls: ["cx-icon-btn", "cx-icon-btn-sm", "cx-focusable"], text: "▢",
    attrs: { type: "button", "aria-label": "Maximise" },
    on: { click: () => {
      if (maximised) { Object.assign(box, restoreBox); maximised = false; }
      else {
        restoreBox = { ...box };
        Object.assign(box, { x: 8, y: 8, width: window.innerWidth - 16, height: window.innerHeight - 16 });
        maximised = true;
      }
      apply();
    } } }));
  controls.append(el("button", { cls: ["cx-icon-btn", "cx-icon-btn-sm", "cx-focusable"], text: "✕",
    attrs: { type: "button", "aria-label": "Close" },
    on: { click: () => handle.close("close-button") } }));
  titlebar.append(controls);

  const content = el("div", { cls: "cx-window-body", children: nodeOf(body, "floating.window") });
  node.append(titlebar, content);

  const grip = resizable ? el("div", { cls: "cx-window-grip", attrs: { "aria-hidden": "true" } }) : null;
  if (grip) node.append(grip);

  apply();
  mount(node);

  // Click anywhere in the window brings it to the front of its rung.
  const toFront = () => { layer.raise(); };
  node.addEventListener("pointerdown", toFront);

  // A window open while the viewport shrinks has the same problem as one
  // reopened on a smaller screen.
  const onViewportResize = () => apply();
  window.addEventListener("resize", onViewportResize);

  let start = null;
  const stopDrag = drag(titlebar, {
    onStart: () => { start = { ...box }; toFront(); },
    onMove: ({ dx, dy }) => {
      // apply() does the clamping, so a window can never be dragged somewhere
      // it cannot be dragged back from -- and the rule lives in one place.
      box.x = start.x + dx;
      box.y = start.y + dy;
      apply();
    },
    onEnd: () => remember(id, box),
  });

  const stopResize = grip ? drag(grip, {
    onStart: () => { start = { ...box }; },
    onMove: ({ dx, dy }) => {
      box.width = Math.max(220, start.width + dx);
      box.height = Math.max(120, start.height + dy);
      apply();
    },
    onEnd: () => remember(id, box),
  }) : () => {};

  // Escape closes, but a window does not close on an outside click: it is a
  // window, not a menu, and clicking the app behind it is normal.
  const dismissal = push({ nodes: node, closeOnOutside: false, onDismiss: (r) => handle.close(r) });

  let closed = false;
  const handle = {
    node,
    isOverlay: true,
    toFront,
    close(reason = "manual") {
      if (closed) return;
      closed = true;
      remember(id, box);
      dismissal.release();
      stopDrag(); stopResize();
      window.removeEventListener("resize", onViewportResize);
      layer.release();
      unmount(node);
      if (onClose) onClose(reason);
    },
    destroy() { handle.close("destroy"); },
  };
  return handle;
});

/** A full-screen "please wait" that genuinely blocks. */
define("overlay", "blocking", ({ message, progress, cancel } = {}) => {
  const card = el("div", { cls: "cx-blocking-card", children: [
    el("span", { cls: ["cx-spinner", "cx-spinner-md"], attrs: { "aria-hidden": "true" } }),
    el("p", { cls: "cx-t cx-t-sm", text: message }),
    progress && progress.node ? progress.node : null,
    cancel && cancel.node ? cancel.node : null,
  ].filter(Boolean) });

  const node = el("div", { cls: "cx-blocking",
    attrs: { role: "alertdialog", "aria-busy": "true", "aria-label": message },
    children: card });
  const layer = claimFor("wizard", node);
  mount(node);
  const release = trap(card);

  let closed = false;
  const handle = {
    node, isOverlay: true,
    close() { if (closed) return; closed = true; release(); layer.release(); unmount(node); },
    destroy() { handle.close(); },
  };
  return handle;
});

/** A panel that slides in from an edge. Simple mode uses these over the preview. */
define("slideOver", "md", ({ side = "right", title, body, onClose } = {}) => {
  const titleId = uid("slide-title");

  const panel = el("aside", { cls: ["cx-slideover", `cx-slideover-${side}`],
    attrs: { role: "dialog", "aria-labelledby": titleId } });
  panel.append(el("header", { cls: "cx-slideover-head", children: [
    el("h2", { cls: "cx-panel-title", text: title, attrs: { id: titleId } }),
    el("button", { cls: ["cx-icon-btn", "cx-icon-btn-md", "cx-focusable"], text: "✕",
      attrs: { type: "button", "aria-label": "Close" }, on: { click: () => handle.close("close-button") } }),
  ] }));
  panel.append(el("div", { cls: "cx-slideover-body", children: nodeOf(body, "slideOver.md") }));

  const root = el("div", { cls: "cx-slideover-layer", children: [el("div", { cls: "cx-backdrop" }), panel] });
  const layer = claimFor("modal", root);
  mount(root);

  const release = trap(panel);
  const dismissal = push({ nodes: panel, onDismiss: (r) => handle.close(r) });

  let closed = false;
  const handle = {
    node: panel, isOverlay: true,
    close(reason = "manual") {
      if (closed) return;
      closed = true;
      dismissal.release(); release(); layer.release(); unmount(root);
      if (onClose) onClose(reason);
    },
    destroy() { handle.close("destroy"); },
  };
  return handle;
});
