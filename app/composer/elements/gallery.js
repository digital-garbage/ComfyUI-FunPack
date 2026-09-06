// Grids of things.
//
// The adaptive one is v4's media bin generalised: it sizes off its CONTAINER,
// not the viewport, so a 280px panel lays out like a 280px panel whether the
// window is 1200px or 3000px wide.
//
// Three of these show the SAME items in different shapes -- grid, list, icons
// -- so they share one item shape and one selection behaviour. A bin that
// forgot what was selected when the view changed would be three galleries, not
// one bin in three views.

import { define } from "../internals/register.js";
import { el } from "../internals/el.js";
import { applyDensity, recallCols, rememberCols, normaliseCols } from "../internals/density.js";

const EMPTY = "Nothing here yet";

/** One item's picture: a real thumbnail, or the glyph that stands in for one. */
function thumbOf(item, cls = "cx-cell-thumb", marks = true) {
  const thumb = el("div", { cls });
  const glyph = () => el("span", { cls: "cx-cell-glyph", text: item.icon || "▦", attrs: { "aria-hidden": "true" } });
  if (item.thumb) {
    // Decorative: the caption or the row's name carries it, so an alt would
    // just make a screen reader say everything twice.
    const img = el("img", { cls: "cx-cell-img", attrs: { src: item.thumb, alt: "", loading: "lazy" } });
    // A thumbnail that does not arrive falls back to the glyph rather than to
    // the browser's broken-image icon, which reads as a damaged FILE. In a bin
    // of fifty, one server hiccup would otherwise look like lost work.
    img.addEventListener("error", () => img.replaceWith(glyph()));
    thumb.append(img);
  } else {
    thumb.append(glyph());
  }
  if (marks && item.badge) thumb.append(el("span", { cls: "cx-cell-badge", text: item.badge }));
  if (marks && item.duration) thumb.append(el("span", { cls: "cx-cell-duration", text: item.duration }));
  return thumb;
}

/**
 * The part every view of a collection has: items, which are selected, and what
 * a click on one means. `cellFor` is the only difference between them.
 */
function collection({ items = [], empty = EMPTY, selection = [], onActivate, onContext } = {},
                    host, cellFor) {
  const chosen = new Set(selection);

  function draw() {
    host.replaceChildren();
    if (!items.length) {
      host.append(el("p", { cls: "cx-list-empty", text: empty }));
      return;
    }
    for (const item of items) {
      const cell = cellFor(item, chosen.has(item.id));
      cell.addEventListener("click", () => { if (onActivate) onActivate(item); });
      if (onContext) {
        cell.addEventListener("contextmenu", (e) => { e.preventDefault(); onContext(item, e); });
      }
      host.append(cell);
    }
  }
  draw();

  return {
    draw,
    get value() { return [...chosen]; },
    setValue(next) { chosen.clear(); for (const v of next || []) chosen.add(v); draw(); },
    setItems(next) { items = next || []; draw(); },
  };
}

const option = (on, extra = {}) => ({ type: "button", role: "option", "aria-selected": String(on), ...extra });

define("gallery", "adaptive", (props = {}) => {
  const { id = "gallery", cols } = props;
  const grid = el("div", { cls: "cx-gallery", attrs: { role: "listbox", "aria-multiselectable": "true" } });
  applyDensity(grid, cols === undefined ? recallCols(id) : cols);

  const api = collection(props, grid, (item, on) => {
    const cell = el("button", { cls: ["cx-cell", on ? "cx-on" : null, "cx-focusable"], attrs: option(on) });
    cell.append(thumbOf(item), el("span", { cls: "cx-cell-name", text: item.label }));
    return cell;
  });

  const node = el("div", { cls: "cx-gallery-wrap", children: grid });
  return {
    node,
    get value() { return api.value; },
    setValue: api.setValue,
    setItems: api.setItems,
    setCols(next) {
      const n = normaliseCols(next);
      applyDensity(grid, n);
      rememberCols(id, n);
    },
    destroy: () => node.remove(),
  };
});

/**
 * The same items as tiles with no captions: as many as fit, the name on hover
 * and to a screen reader. For a bin holding dozens of results, where what is
 * being read is the picture and not its filename.
 */
define("gallery", "icons", (props = {}) => {
  const grid = el("div", { cls: ["cx-gallery", "cx-gallery-icons"],
    attrs: { role: "listbox", "aria-multiselectable": "true" } });

  const api = collection(props, grid, (item, on) => {
    // title AND aria-label: without a caption the tile has no accessible name
    // at all, and a title alone is not one on a control.
    const cell = el("button", { cls: ["cx-cell", "cx-cell-icon", on ? "cx-on" : null, "cx-focusable"],
      attrs: option(on, { title: item.label, "aria-label": item.label }) });
    // No badge or duration: 9px of text over a 48px tile covers the picture the
    // tile exists to show, and this view is chosen to see many pictures at once.
    cell.append(thumbOf(item, "cx-cell-thumb", false));
    return cell;
  });

  const node = el("div", { cls: "cx-gallery-wrap", children: grid });
  return {
    node,
    get value() { return api.value; },
    setValue: api.setValue,
    setItems: api.setItems,
    destroy: () => node.remove(),
  };
});

/** The same items as rows: a small picture, the whole name, and what it is. */
define("gallery", "list", (props = {}) => {
  const body = el("div", { cls: "cx-media-list", attrs: { role: "listbox" } });

  const api = collection(props, body, (item, on) => el("button", {
    cls: ["cx-media-row", on ? "cx-on" : null, "cx-focusable"],
    attrs: option(on),
    children: [
      // No badge or duration over a 40px picture -- they would cover it. The
      // row has room to say both in words instead.
      thumbOf(item, "cx-media-row-thumb", false),
      el("span", { cls: "cx-media-row-name", text: item.label }),
      item.hint ? el("span", { cls: "cx-media-row-hint", text: item.hint }) : null,
    ].filter(Boolean),
  }));

  const node = el("div", { cls: "cx-gallery-wrap", children: body });
  return {
    node,
    get value() { return api.value; },
    setValue: api.setValue,
    setItems: api.setItems,
    destroy: () => node.remove(),
  };
});

/** Large clickable cards: the wizard's "what are you making?" choices. */
define("gallery", "cards", ({ items = [], value, onActivate } = {}) => {
  let current = value;
  const node = el("div", { cls: "cx-cards", attrs: { role: "radiogroup" } });
  const cards = items.map((item) => {
    const card = el("button", {
      cls: ["cx-card", "cx-focusable"],
      attrs: { type: "button", role: "radio", "aria-checked": String(item.id === current) },
      children: [
        item.icon ? el("span", { cls: "cx-card-icon", text: item.icon, attrs: { "aria-hidden": "true" } }) : null,
        el("span", { cls: "cx-card-title", text: item.label }),
        item.hint ? el("span", { cls: "cx-hint", text: item.hint }) : null,
      ].filter(Boolean),
    });
    card.classList.toggle("cx-on", item.id === current);
    card.addEventListener("click", () => {
      current = item.id;
      for (const { item: i, card: c } of cards) {
        c.classList.toggle("cx-on", i.id === current);
        c.setAttribute("aria-checked", String(i.id === current));
      }
      if (onActivate) onActivate(item);
    });
    node.append(card);
    return { item, card };
  });
  return { node, get value() { return current; }, destroy: () => node.remove() };
});

/**
 * A single scrolling row: filmstrips, segment strips, recent items.
 *
 * Selectable like its siblings, and through the same `collection` -- a strip is
 * the one shape whose whole job is "which of these am I on", and one that could
 * only be clicked THROUGH left the caller drawing its own current-item marker
 * over a row that did not have the concept.
 */
define("gallery", "strip", ({ onReorder, ...props } = {}) => {
  const node = el("div", { cls: "cx-strip", attrs: { role: "listbox", "aria-label": props.label } });

  const api = collection(props, node, (item, on) => {
    const cell = el("button", {
      cls: ["cx-strip-cell", on ? "cx-on" : null, item.rating ? "cx-rated" : null,
            item.excluded ? "cx-excluded" : null, "cx-focusable"],
      attrs: option(on, { title: item.label, "aria-label": item.label,
                          "data-rating": item.rating || undefined,
                          // The item's OWN id, not just its position -- a
                          // reorder tracked by index goes stale the moment
                          // anything redraws the strip mid-drag (a scene
                          // removed by keyboard while the mouse is still
                          // held down, for one), since a redraw can happen on
                          // its own input channel independent of the drag.
                          "data-id": item.id,
                          // Draggable whether or not `onReorder` is given: a
                          // drag with nowhere to report to is inert, not wired
                          // -- the delegated listeners below are the only thing
                          // that acts on it, and they no-op without a handler.
                          draggable: "true" }) });
    // As wide as what it stands for is long, which is what turns a row of equal
    // boxes into something that reads as time. Bounded at both ends: one very
    // long clip beside short ones must not squeeze the rest to a sliver.
    if (item.weight) cell.style.flexGrow = String(Math.min(6, Math.max(1, Number(item.weight) || 1)));
    // The same fallback the grid has: a thumbnail that does not arrive shows the
    // glyph, not the browser's broken-image icon, which reads as a damaged file.
    cell.append(thumbOf(item, "cx-strip-face", false));
    if (item.badge) cell.append(el("span", { cls: "cx-cell-badge", text: item.badge }));
    return cell;
  });

  // Delegated on the container rather than per-cell: cells are torn down and
  // rebuilt on every setItems/setValue, and listeners bound to them would have
  // to be reattached every time. One set here survives all of that.
  //
  // The dragged item's id lives on `dataTransfer`, not in a closure variable.
  // A closure var tracks by NOTHING the browser understands, so nothing keeps
  // it in step with the actual drag session: `dragend` fires on the ORIGINAL
  // source element, and a redraw mid-drag (an unrelated remove/undo firing on
  // its own input channel while the mouse is still held down) can detach that
  // element from the tree -- its `dragend` then never bubbles here, and a
  // closure var set at dragstart is never cleared, ready to be picked up by
  // the next unrelated drop that lands on this strip from somewhere else
  // entirely. `dataTransfer` has no such gap: it belongs to the actual OS-level
  // drag session, unaffected by which DOM node currently receives the events,
  // and a drop from any OTHER session simply never had this MIME type set.
  const MIME = "text/x-funpack-reorder";
  const cellOf = (e) => e.target.closest(".cx-strip-cell");
  node.addEventListener("dragstart", (e) => {
    const cell = cellOf(e);
    if (!cell || !e.dataTransfer) return;
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData(MIME, cell.dataset.id || "");
    cell.classList.add("cx-dragging");
  });
  node.addEventListener("dragend", (e) => { cellOf(e)?.classList.remove("cx-dragging"); });
  node.addEventListener("dragover", (e) => {
    // `getData` only returns the real value on "drop" (a browser security
    // restriction during drag) -- `types` is what dragover can actually read.
    if (e.dataTransfer && e.dataTransfer.types.includes(MIME)) e.preventDefault();
  });
  node.addEventListener("drop", (e) => {
    if (!e.dataTransfer) return;
    const draggedId = e.dataTransfer.getData(MIME);
    if (!draggedId) return;
    e.preventDefault();
    const cell = cellOf(e);
    const targetId = cell ? cell.dataset.id : null;
    if (onReorder && targetId && targetId !== draggedId) onReorder(draggedId, targetId);
  });

  return {
    node,
    get value() { return api.value; },
    setValue: api.setValue,
    setItems: api.setItems,
    destroy: () => node.remove(),
  };
});

/**
 * A scale over something laid out in a row: where each part begins.
 *
 * The marks are given rather than computed from an interval -- what a person
 * looks for on a timeline is where one scene becomes the next, and at 24 frames
 * a second a tick per second is a picket fence.
 */
define("ruler", "default", ({ marks = [], total = 1, label } = {}) => {
  const node = el("div", { cls: "cx-ruler", attrs: { role: "presentation", "aria-label": label } });

  function set(next = [], span = 1) {
    node.replaceChildren();
    for (const mark of next) {
      const tick = el("span", { cls: "cx-ruler-mark", text: mark.label, attrs: { title: mark.hint } });
      // Percent, so the ruler tracks the row above it at any width.
      tick.style.insetInlineStart = `${Math.min(100, Math.max(0, (mark.at || 0) * 100))}%`;
      node.append(tick);
    }
    node.dataset.total = String(span);
  }

  set(marks, total);
  return { node, set, destroy: () => node.remove() };
});

/**
 * The result, big. One image or one video, or an empty state before there is
 * either -- because "nothing has been generated" and "the picture failed to
 * load" have to look different, and a bare <img> with a broken src looks like
 * neither.
 */
define("viewer", "media", ({ src, kind = "image", empty = "Nothing yet", onError } = {}) => {
  const stage = el("div", { cls: "cx-viewer-stage" });
  let current = null;
  let currentKind = kind;
  let currentFile = null;
  let mediaEl = null;

  function show(next, nextKind = kind, file = null) {
    current = next || null;
    currentKind = nextKind;
    currentFile = current ? file : null;
    mediaEl = null;
    stage.replaceChildren();
    if (!current) {
      stage.append(el("p", { cls: "cx-list-empty", text: empty }));
      return;
    }
    const media = nextKind === "video"
      ? el("video", { cls: "cx-viewer-media", attrs: { src: current, controls: "", loop: "", playsinline: "" } })
      : el("img", { cls: "cx-viewer-media", attrs: { src: current, alt: "" } });
    mediaEl = media;
    // A failed load is reported rather than left as a broken icon: the file is
    // produced by a run, so it not arriving is a fault worth naming.
    media.addEventListener("error", () => {
      stage.replaceChildren(el("p", { cls: "cx-list-empty", text: "This result could not be loaded." }));
      mediaEl = null;
      // Nothing is actually showing any more -- `file` saying otherwise is
      // what let "Save to bin" silently bin a dead reference from this exact
      // state, the one place the panel itself says there is nothing here.
      currentFile = null;
      if (onError) onError(current);
    });
    stage.append(media);
  }

  show(src, kind);
  const node = el("div", { cls: "cx-viewer", children: stage });
  return {
    node,
    get value() { return current; },
    get kind() { return currentKind; },
    // The file this came from -- {filename, subfolder, type} -- when the
    // caller had one to give. Null for anything shown by URL alone, which is
    // not enough to name a file back to the server with.
    get file() { return currentFile; },
    setValue: (next) => show(next, kind),
    setSource: (next, nextKind, file) => show(next, nextKind || kind, file),
    /**
     * The video frame on screen right now, as a PNG blob -- null if nothing is
     * playing or there is no frame yet to draw (a video that has not loaded
     * its first frame has no dimensions to capture at).
     */
    captureFrame() {
      if (currentKind !== "video" || !mediaEl || !mediaEl.videoWidth) return Promise.resolve(null);
      const canvas = document.createElement("canvas");
      canvas.width = mediaEl.videoWidth;
      canvas.height = mediaEl.videoHeight;
      canvas.getContext("2d").drawImage(mediaEl, 0, 0);
      return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    },
    destroy: () => node.remove(),
  };
});
