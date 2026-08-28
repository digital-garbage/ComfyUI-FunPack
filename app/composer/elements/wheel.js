// The radial picker.
//
// Nothing in v4 resembles this. It is for the handful of choices made over and
// over during editing, where a menu costs a read-and-aim every time: press,
// flick toward the wedge you already know the position of, release. The muscle
// memory is the feature -- the same choice always lives at the same angle.

import { define } from "../internals/register.js";
import { el, svg } from "../internals/el.js";
import { mount, unmount } from "../internals/portal.js";
import { claim } from "../internals/zlayer.js";
import { push } from "../internals/dismiss.js";

const SIZE = 240;          // diameter in px
const DEAD_ZONE = 34;      // centre radius that means "cancel"
const MIN = 2;
const MAX = 12;

const point = (cx, cy, radius, degrees) => {
  const rad = (degrees - 90) * (Math.PI / 180);
  return [cx + Math.cos(rad) * radius, cy + Math.sin(rad) * radius];
};

/**
 * An annular wedge: the actual area that picks this item.
 *
 * Highlighting the wedge rather than a box around the label is what makes the
 * target legible -- the lit shape IS the region your pointer has to be in, so
 * aiming stops being a guess about where one item's zone ends.
 */
export function wedgePath(index, count, { cx = SIZE / 2, cy = SIZE / 2, rIn = DEAD_ZONE, rOut = SIZE / 2 - 3 } = {}) {
  const slice = 360 / count;
  const start = index * slice - slice / 2;
  const end = index * slice + slice / 2;
  const largeArc = slice > 180 ? 1 : 0;

  const [x1, y1] = point(cx, cy, rOut, start);
  const [x2, y2] = point(cx, cy, rOut, end);
  const [x3, y3] = point(cx, cy, rIn, end);
  const [x4, y4] = point(cx, cy, rIn, start);

  const f = (n) => Number(n.toFixed(2));
  return `M ${f(x1)} ${f(y1)} A ${rOut} ${rOut} 0 ${largeArc} 1 ${f(x2)} ${f(y2)} ` +
         `L ${f(x3)} ${f(y3)} A ${rIn} ${rIn} 0 ${largeArc} 0 ${f(x4)} ${f(y4)} Z`;
}

/** Which sector a point falls in, or -1 for the dead zone. */
export function sectorAt(dx, dy, count, deadZone = DEAD_ZONE) {
  const distance = Math.hypot(dx, dy);
  if (distance < deadZone) return -1;
  // Angles run clockwise from straight up, so item 0 is at 12 o'clock and the
  // arrangement matches how people describe it ("the one at the top").
  const angle = (Math.atan2(dx, -dy) + Math.PI * 2) % (Math.PI * 2);
  const slice = (Math.PI * 2) / count;
  return Math.floor((angle + slice / 2) % (Math.PI * 2) / slice) % count;
}

define("wheel", "picker", ({ items = [], onPick, onClose, x, y, cancelOnCentre = true } = {}) => {
  if (items.length < MIN || items.length > MAX) {
    throw new RangeError(`wheel.picker takes ${MIN}-${MAX} items; got ${items.length}. More than that and nobody can aim.`);
  }

  const layer = claim("popover");
  const centre = {
    x: x ?? Math.round(window.innerWidth / 2),
    y: y ?? Math.round(window.innerHeight / 2),
  };

  const node = el("div", { cls: "cx-wheel", attrs: { role: "menu", "aria-label": "Quick pick" } });
  node.style.zIndex = String(layer.z);
  node.style.left = `${centre.x - SIZE / 2}px`;
  node.style.top = `${centre.y - SIZE / 2}px`;
  node.style.width = `${SIZE}px`;
  node.style.height = `${SIZE}px`;

  let active = -1;

  // The wedges are drawn behind the labels, so the highlight is the pickable
  // region itself rather than a box around some text.
  const canvas = svg("svg", { class: "cx-wheel-canvas", viewBox: `0 0 ${SIZE} ${SIZE}`,
                              width: SIZE, height: SIZE, "aria-hidden": "true" });
  const shapes = items.map((_, i) => {
    const path = svg("path", { class: "cx-wheel-wedge", d: wedgePath(i, items.length) });
    canvas.append(path);
    return path;
  });
  node.append(canvas);

  const wedges = items.map((item, i) => {
    const angle = (360 / items.length) * i;
    const wedge = el("button", {
      cls: ["cx-wheel-item", "cx-focusable"],
      attrs: { type: "button", role: "menuitem", title: item.label },
      children: [
        item.icon ? el("span", { cls: "cx-wheel-icon", text: item.icon, attrs: { "aria-hidden": "true" } }) : null,
        el("span", { cls: "cx-wheel-label", text: item.label }),
      ].filter(Boolean),
    });
    // Placed on a circle rather than drawn as a pie slice: a round label that
    // stays upright is far easier to read than text bent around a wedge.
    const radius = SIZE / 2 - 34;
    const rad = (angle - 90) * (Math.PI / 180);
    wedge.style.left = `${SIZE / 2 + Math.cos(rad) * radius}px`;
    wedge.style.top = `${SIZE / 2 + Math.sin(rad) * radius}px`;
    wedge.addEventListener("click", () => commit(i));
    node.append(wedge);
    return wedge;
  });

  const hub = el("div", { cls: "cx-wheel-hub", attrs: { "aria-hidden": "true" },
    children: el("span", { cls: "cx-hint", text: cancelOnCentre ? "release here to cancel" : "" }) });
  node.append(hub);

  function highlight(index) {
    active = index;
    wedges.forEach((w, i) => w.classList.toggle("cx-on", i === index));
    shapes.forEach((s, i) => s.classList.toggle("cx-on", i === index));
    hub.classList.toggle("cx-on", index === -1);
  }

  function commit(index) {
    if (index >= 0 && items[index] && onPick) onPick(items[index]);
    handle.close(index >= 0 ? "picked" : "cancelled");
  }

  // Press-and-flick: the pointer never has to land on the item, only point at
  // it, which is what makes this faster than a menu.
  //
  // `armed` exists because the click that OPENS the wheel also delivers a
  // pointerup to this listener, at wherever the opening control happened to be
  // -- which committed whatever wedge that direction pointed at, instantly.
  // Requiring a pointermove first means a release only counts once the pointer
  // has actually gone somewhere.
  let armed = false;

  const at = (event) => sectorAt(
    event.clientX - centre.x, event.clientY - centre.y, items.length,
    cancelOnCentre ? DEAD_ZONE : 0);

  const onMove = (event) => { armed = true; highlight(at(event)); };
  const onUp = (event) => {
    if (!armed) return;
    armed = false;
    const index = at(event);
    if (index >= 0) commit(index);
  };

  const onKeyDown = (event) => {
    const n = items.length;
    if (/^[1-9]$/.test(event.key)) {
      const i = Number(event.key) - 1;
      if (i < n) { event.preventDefault(); commit(i); }
    } else if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      event.preventDefault(); highlight(((active < 0 ? -1 : active) + 1 + n) % n);
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      event.preventDefault(); highlight(((active < 0 ? 1 : active) - 1 + n) % n);
    } else if (event.key === "Enter" && active >= 0) {
      event.preventDefault(); commit(active);
    }
  };

  mount(node);
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);
  window.addEventListener("keydown", onKeyDown);

  const dismissal = push({ nodes: node, onDismiss: (reason) => handle.close(reason) });

  let closed = false;
  const handle = {
    node,
    isOverlay: true,
    get active() { return active; },
    highlight,
    close(reason = "manual") {
      if (closed) return;
      closed = true;
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("keydown", onKeyDown);
      dismissal.release();
      layer.release();
      unmount(node);
      if (onClose) onClose(reason);
    },
    destroy() { handle.close("destroy"); },
  };
  return handle;
});
