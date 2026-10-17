// A transparent stage laid over a picture, carrying text and pictures that can be dragged and resized.
//
// The caller gives DATA (where, how big, what colour, what text) and the kit turns it into style. Positions are fractions of the
// picture (0..1); sizes are in the picture's own pixels and drawn at `scale` (screen pixels per picture pixel). A move or a resize is
// reported as a number (the new x and y, or a growth factor), never a style: what it means for the project is the caller's.
import { define } from "../internals/register.js";
import { el } from "../internals/el.js";
import { drag } from "../internals/drag.js";

const clamp01 = (v) => Math.min(1, Math.max(0, v));

define("stage", "overlay", ({ onMove, onResize, onCancel, onLayout, label = "Overlays" } = {}) => {
  const node = el("div", { cls: "cx-stage", attrs: { "aria-label": label } });
  let scale = 1, w = 1, h = 1, busy = false, watching = null;

  function area() {       // the visible thing in the parent that the picture sits in: the monitor, or its "no render" panel
    const parent = node.parentElement;
    return parent ? [...parent.children].find((n) => n !== node && !n.hidden && n.getBoundingClientRect().height > 60) : null;
  }

  function item(o) {
    const root = el("div", { cls: "cx-stage-item", attrs: { "data-id": o.id } });
    const at = (x, y) => { root.style.left = `${x * 100}%`; root.style.top = `${y * 100}%`; };
    root.style.transform = `translate(-50%, -50%) scale(${o.flipH ? -1 : 1}, ${o.flipV ? -1 : 1})`;
    root.style.opacity = String(o.opacity ?? 1);
    at(o.x ?? 0.5, o.y ?? 0.5);
    let resize;
    if (o.kind === "text") {
      const size = (fs) => `${fs * scale}px`, px = (o.fontSize || 42) * scale;
      root.append(String(o.text || "").trim() || "Title");       // the render trims it too
      Object.assign(root.style, { whiteSpace: "pre", fontSize: size(o.fontSize || 42), color: o.color || "#fff", fontFamily: o.family || "arial",
        fontWeight: o.bold ? "700" : "400", fontStyle: o.italic ? "italic" : "normal", textAlign: o.align || "center", lineHeight: String(o.lineSpacing || 1.2),
        textShadow: o.shadow ? `${px * 0.06}px ${px * 0.06}px 0 ${o.shadowColor || "#000"}` : "none",
        webkitTextStroke: o.strokeWidth ? `${o.strokeWidth * scale}px ${o.strokeColor || "#000"}` : "0" });
      if (o.bg) Object.assign(root.style, { background: `color-mix(in srgb, ${o.bg.color || "#000"} ${Math.round((o.bg.opacity ?? 0.5) * 100)}%, transparent)`, padding: `${px * 0.22}px ${px * 0.4}px`, borderRadius: `${px * 0.1}px` });
      resize = (f) => { root.style.fontSize = size(Math.max(8, Math.round((o.fontSize || 42) * f))); };
    } else {
      const img = el("img", { attrs: { src: o.src, draggable: "false" } });
      const sized = (f) => { img.style.width = `${Math.max(8, Math.round(o.widthPx * f)) * scale}px`; img.style.height = o.heightPx ? `${Math.max(8, Math.round(o.heightPx * f)) * scale}px` : ""; };
      sized(1);
      root.append(img);
      resize = sized;
    }
    // The corner handle: a picture grows by its width, a text by its size, from where the corner is dragged.
    const grip = el("span", { cls: "cx-stage-grip" });
    root.append(grip);
    let factor = null, w0 = 1;
    drag(grip, {
      onStart: ({ event }) => { event.stopPropagation(); busy = true; w0 = root.getBoundingClientRect().width || 1; },       // measured once: the factor is against the size at the start
      onMove: ({ dx }) => { factor = Math.max(0.1, (w0 + dx * 2) / w0); resize(factor); },       // centred: both sides grow
      onEnd: ({ cancelled }) => { busy = false; const f = factor; factor = null; if (f && !cancelled) { if (onResize) onResize(o.id, f); } else if (onCancel) onCancel(); },
    });
    let to = null;
    drag(root, {
      onStart: () => { busy = true; },
      onMove: ({ dx, dy }) => { to = { x: clamp01((o.x ?? 0.5) + dx / w), y: clamp01((o.y ?? 0.5) + dy / h) }; at(to.x, to.y); },
      onEnd: ({ cancelled }) => { busy = false; const t = to; to = null; if (t && !cancelled) { if (onMove) onMove(o.id, t); } else if (onCancel) onCancel(); },
    });
    return root;
  }

  return {
    node,
    /** True while a move or resize is under the pointer: do not redraw then. */
    get busy() { return busy; },
    /** Fit the stage over the picture, which has the aspect `pw` x `ph`, inside its parent -> false when there is no room yet. */
    fit(pw, ph) {
      const a = area(), parent = node.parentElement;
      if (!a) return false;
      if (parent && watching !== parent && typeof ResizeObserver === "function") {       // the parent changing size is a reason to fit again
        if (watching) watching.ro.disconnect();
        const ro = new ResizeObserver(() => { if (onLayout) onLayout(); });
        ro.observe(parent);
        watching = { ro, parent };
      }
      const s = a.getBoundingClientRect(), p0 = parent.getBoundingClientRect(), k = Math.min(s.width / pw, s.height / ph);
      w = pw * k; h = ph * k; scale = k;
      Object.assign(node.style, { left: `${s.left - p0.left + (s.width - w) / 2}px`, top: `${s.top - p0.top + (s.height - h) / 2}px`, width: `${w}px`, height: `${h}px` });
      return w > 0 && h > 0;
    },
    get size() { return { w, h }; },
    /** Draw `items` (see the top of this file) at the scale `fit` found. */
    setItems(items = []) { node.replaceChildren(...items.map(item)); },
    clear() { node.replaceChildren(); },
    destroy() { if (watching) watching.ro.disconnect(); node.remove(); },
  };
});
