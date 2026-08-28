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
const DEAD_ZONE = 42;      // centre radius that means "cancel"; matches the hub exactly
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
export function wedgePath(index, count, {
  cx = SIZE / 2, cy = SIZE / 2, rIn = DEAD_ZONE, rOut = SIZE / 2 - 3,
  startDeg = -(360 / count) / 2, spanDeg = 360,
} = {}) {
  const slice = spanDeg / count;
  const start = startDeg + index * slice;
  const end = start + slice;
  const largeArc = slice > 180 ? 1 : 0;

  const [x1, y1] = point(cx, cy, rOut, start);
  const [x2, y2] = point(cx, cy, rOut, end);
  const [x3, y3] = point(cx, cy, rIn, end);
  const [x4, y4] = point(cx, cy, rIn, start);

  const f = (n) => Number(n.toFixed(2));
  return `M ${f(x1)} ${f(y1)} A ${rOut} ${rOut} 0 ${largeArc} 1 ${f(x2)} ${f(y2)} ` +
         `L ${f(x3)} ${f(y3)} A ${rIn} ${rIn} 0 ${largeArc} 0 ${f(x4)} ${f(y4)} Z`;
}

/**
 * Which sector of an arc a point falls in; -1 for the dead zone or outside it.
 *
 * Angles run clockwise from straight up, so 0 is 12 o'clock and the arrangement
 * matches how people describe it ("the one at the top"). Everything radial in
 * the kit goes through here, so a full wheel and an edge panel cannot disagree
 * about which item a direction means.
 */
export function sectorInArc(dx, dy, count, { startDeg = 0, spanDeg = 360, deadZone = DEAD_ZONE } = {}) {
  if (Math.hypot(dx, dy) < deadZone) return -1;
  const angle = ((Math.atan2(dx, -dy) * 180) / Math.PI + 360) % 360;
  const rel = (angle - startDeg + 720) % 360;
  if (rel > spanDeg) return -1;              // outside a partial arc
  return Math.min(Math.floor(rel / (spanDeg / count)), count - 1);
}

/** The full circle: item 0 centred at the top. */
export function sectorAt(dx, dy, count, deadZone = DEAD_ZONE) {
  return sectorInArc(dx, dy, count, { startDeg: -(360 / count) / 2, spanDeg: 360, deadZone });
}

// Which half of the circle faces inward from each edge, and where a panel
// pinned there sits. An edge panel opens AWAY from its edge, which is the whole
// reason it works with a thumb: the arc lands where the thumb already is.
export const EDGES = {
  right:  { startDeg: 180, spanDeg: 180, anchor: (w, h, at) => [w, at ?? h / 2] },
  left:   { startDeg: 0,   spanDeg: 180, anchor: (w, h, at) => [0, at ?? h / 2] },
  bottom: { startDeg: 270, spanDeg: 180, anchor: (w, h, at) => [at ?? w / 2, h] },
  top:    { startDeg: 90,  spanDeg: 180, anchor: (w, h, at) => [at ?? w / 2, 0] },
};

function buildWheel({
  items, onPick, onClose, centre, arc, cancelOnCentre, cls, hubText,
}) {
  if (items.length < MIN || items.length > MAX) {
    throw new RangeError(`A picker takes ${MIN}-${MAX} items; got ${items.length}. More than that and nobody can aim.`);
  }

  const layer = claim("popover");
  const node = el("div", { cls, attrs: { role: "menu", "aria-label": "Quick pick" } });
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
    const path = svg("path", { class: "cx-wheel-wedge", d: wedgePath(i, items.length, arc) });
    canvas.append(path);
    return path;
  });
  node.append(canvas);

  const slice = arc.spanDeg / items.length;
  const wedges = items.map((item, i) => {
    const angle = arc.startDeg + slice * (i + 0.5);
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

  // A glyph, not a sentence: the hub is small and round, and wrapped text in a
  // circle reads badly at any size. The label survives as the accessible name.
  const hub = el("div", {
    cls: "cx-wheel-hub",
    attrs: cancelOnCentre ? { role: "img", "aria-label": hubText, title: hubText } : { "aria-hidden": "true" },
    children: cancelOnCentre ? el("span", { cls: "cx-wheel-hub-glyph", text: "✕" }) : null,
  });
  node.append(hub);

  function highlight(index) {
    active = index;
    wedges.forEach((w, i) => w.classList.toggle("cx-on", i === index));
    shapes.forEach((sh, i) => sh.classList.toggle("cx-on", i === index));
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

  const at = (event) => sectorInArc(
    event.clientX - centre.x, event.clientY - centre.y, items.length,
    { ...arc, deadZone: cancelOnCentre ? DEAD_ZONE : 0 });

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
}

define("wheel", "picker", ({ items = [], onPick, onClose, x, y, cancelOnCentre = true } = {}) =>
  buildWheel({
    items, onPick, onClose, cancelOnCentre,
    cls: "cx-wheel",
    hubText: "Release here to cancel",
    centre: {
      x: x ?? Math.round(window.innerWidth / 2),
      y: y ?? Math.round(window.innerHeight / 2),
    },
    arc: { startDeg: -(360 / Math.max(items.length, 1)) / 2, spanDeg: 360 },
  }));

/**
 * Half a wheel, pinned to a screen edge -- the Samsung edge-panel shape.
 *
 * A control living at the edge has nowhere to open a full wheel: half of it
 * would be off-screen, and the items that landed there would be unreachable.
 * This puts the centre ON the edge and fans the items into the screen, so the
 * arc lands exactly where the thumb already is and every item is in reach.
 *
 * `at` is the position along that edge (y for left/right, x for top/bottom);
 * it defaults to the middle.
 */
define("wheel", "half", ({ items = [], edge = "right", at, onPick, onClose, cancelOnCentre = true } = {}) => {
  const spec = EDGES[edge];
  if (!spec) throw new RangeError(`Unknown edge "${edge}". Known: ${Object.keys(EDGES).join(", ")}.`);
  const [cx, cy] = spec.anchor(window.innerWidth, window.innerHeight, at);
  return buildWheel({
    items, onPick, onClose, cancelOnCentre,
    cls: `cx-wheel cx-wheel-half cx-wheel-${edge}`,
    hubText: "Release here to cancel",
    centre: { x: cx, y: cy },
    arc: { startDeg: spec.startDeg, spanDeg: spec.spanDeg },
  });
});
