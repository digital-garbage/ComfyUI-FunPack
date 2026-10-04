// The cutting-room timeline: a ruler in seconds over lanes of clips, a sticky gutter naming each lane, and a
// playhead through all of it. Draws and reports -- what a clip IS, and what a drag MEANS, belong to the feature.
//
//   lanes: [{ id, label, kind: "video" | "audio" | "overlay", reorder?, clips: [clip] }]
//   clip:  { id, start, dur, title?, head?: [text], badge?, body?, thumbs?: [url],
//            actions?: [{ label, title, onClick }], selected?, focus?, excluded?, ghost?, rating?, trim? }
//   callbacks: onSeek(sec) onSelect(id, { additive, range }) onTrim(id, "in"|"out", deltaSec) onReorder(id, index) onMove(id, deltaSec)

import { define } from "../internals/register.js";
import { el } from "../internals/el.js";
import { drag } from "../internals/drag.js";
import { snapDelta } from "../internals/snap.js";

const NICE = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
const MIN_TICK_PX = 64;
const MIN_CLIP_PX = 12;
const TAIL_SECONDS = 2;          // room past the last clip to drop or seek into
const MOVE_THRESHOLD_PX = 12;    // a press that travels less than this is a click, not a reorder

const tickEvery = (px) => NICE.find((s) => s * px >= MIN_TICK_PX) || NICE[NICE.length - 1];
const clock = (s) => {
  const whole = Math.max(0, Math.floor(s));
  return `${String(Math.floor(whole / 60)).padStart(2, "0")}:${String(whole % 60).padStart(2, "0")}`;
};

define("timeline", "stage", ({ pxPerSecond = 80, label = "Timeline", lanes: initial = [], onSeek, onSelect, onTrim, onReorder, onMove } = {}) => {
  let px = pxPerSecond;
  let lanes = initial;
  let playhead = 0;
  let disposers = [];

  const gutter = el("div", { cls: "cx-nle-gutter" });
  const ruler = el("div", { cls: "cx-nle-ruler" });
  const laneBox = el("div", { cls: "cx-nle-lanes" });
  const head = el("div", { cls: "cx-nle-playhead", attrs: { "aria-hidden": "true" } });
  const content = el("div", { cls: "cx-nle-content", children: [ruler, laneBox, head] });
  const stage = el("div", { cls: "cx-nle-stage", children: [gutter, content] });
  const node = el("div", { cls: "cx-nle", attrs: { role: "application", "aria-label": label }, children: [stage] });

  const spanOf = () => Math.max(1, ...lanes.flatMap((l) => l.clips.map((c) => c.start + c.dur))) + TAIL_SECONDS;
  const timeAt = (clientX) => Math.max(0, (clientX - content.getBoundingClientRect().left) / px);

  function drawRuler(span) {
    ruler.replaceChildren();
    const every = tickEvery(px);
    for (let s = 0; s <= span; s += every) {
      const tick = el("span", { cls: "cx-nle-tick", text: clock(s) });
      tick.style.insetInlineStart = `${s * px}px`;
      ruler.append(tick);
    }
  }

  // Every other clip's edges, the playhead and zero: what a dragged edge sticks to (Alt holds it free).
  const SNAP_PX = 10;
  const anchorsFor = (clip) => [0, playhead, ...lanes.flatMap((l) => l.clips.filter((c) => c.id !== clip.id).flatMap((c) => [c.start, c.start + c.dur]))];
  const snapped = (clip, edges, dxSec, event) => (event && event.altKey ? dxSec : snapDelta(dxSec, edges, anchorsFor(clip), SNAP_PX / px));

  function clipNode(lane, clip) {
    const parts = [];
    if (clip.head || clip.actions) {
      parts.push(el("div", { cls: "cx-nle-clip-head", children: [
        ...(clip.head || []).map((t) => el("span", { cls: "cx-nle-clip-tag", text: t })),
        el("span", { cls: "cx-nle-clip-actions", children: (clip.actions || []).map((a) => el("button", {
          cls: ["cx-nle-clip-action", "cx-focusable"], text: a.label,
          attrs: { type: "button", title: a.title, "data-action": "" },
          on: { click: (e) => { e.stopPropagation(); a.onClick(); } },
        })) }),
      ] }));
    }
    if (clip.thumbs && clip.thumbs.length) {
      parts.push(el("div", { cls: "cx-nle-strip", children: clip.thumbs.map((url) => el("img", { cls: "cx-nle-thumb", attrs: { src: url, alt: "", loading: "lazy" } })) }));
    }
    if (clip.title) parts.push(el("div", { cls: "cx-nle-clip-title", text: clip.title }));
    if (clip.body) parts.push(el("div", { cls: "cx-nle-clip-body", text: clip.body }));
    if (clip.trim) {
      for (const edge of ["in", "out"]) parts.push(el("span", { cls: ["cx-nle-trim", `cx-nle-trim-${edge}`], attrs: { "data-trim": edge } }));
    }
    const cell = el("div", {
      cls: ["cx-nle-clip", `cx-nle-clip-${lane.kind}`, clip.selected ? "cx-on" : null, clip.focus ? "cx-focus" : null,
        clip.excluded ? "cx-excluded" : null, clip.ghost ? "cx-ghost" : null, clip.rating ? "cx-rated" : null],
      attrs: { role: "option", tabindex: "0", "aria-selected": String(Boolean(clip.selected)), "aria-label": clip.title || clip.id,
        "data-id": clip.id, "data-rating": clip.rating || undefined },
      children: parts,
    });
    cell.style.insetInlineStart = `${clip.start * px}px`;
    cell.style.width = `${Math.max(MIN_CLIP_PX, clip.dur * px)}px`;
    cell.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect && onSelect(clip.id, {}); } });

    // Trim handles first, then the body: a press on a handle must not also start a reorder.
    cell.querySelectorAll("[data-trim]").forEach((h) => disposers.push(drag(h, {
      onStart: ({ event }) => event.stopPropagation(),
      onMove: ({ dx, event }) => {
        const edge = h.dataset.trim;
        dx = snapped(clip, [edge === "out" ? clip.start + clip.dur : clip.start], dx / px, event) * px;
        const dur = Math.max(0.1, clip.dur + (edge === "out" ? dx : -dx) / px);
        cell.style.width = `${Math.max(MIN_CLIP_PX, dur * px)}px`;
        if (edge === "in") cell.style.insetInlineStart = `${(clip.start + (clip.dur - dur)) * px}px`;
      },
      onEnd: ({ dx, cancelled, event }) => {
        dx = snapped(clip, [h.dataset.trim === "out" ? clip.start + clip.dur : clip.start], dx / px, event) * px;
        if (!cancelled && onTrim && dx) onTrim(clip.id, h.dataset.trim, dx / px);
        draw();
      },
    })));

    let moved = false;
    disposers.push(drag(cell, {
      onStart: ({ event }) => { if (event.target.closest("[data-action],[data-trim]")) { moved = null; return; } moved = false; },
      onMove: ({ dx, event }) => {
        if (moved === null) return;
        if (Math.abs(dx) >= MOVE_THRESHOLD_PX && (lane.reorder || lane.move)) moved = true;
        if (moved && lane.move) dx = snapped(clip, [clip.start, clip.start + clip.dur], dx / px, event) * px;
        if (moved) cell.style.transform = `translateX(${dx}px)`;
      },
      onEnd: ({ dx, event, cancelled }) => {
        cell.style.transform = "";
        if (moved === null || cancelled) return;
        if (!moved) {
          if (onSelect) onSelect(clip.id, { additive: event.metaKey || event.ctrlKey, range: event.shiftKey });
          return;
        }
        if (lane.move) { if (onMove) onMove(clip.id, snapped(clip, [clip.start, clip.start + clip.dur], dx / px, event)); draw(); return; }          // a free lane: the clip goes where it was dropped, in seconds
        // Dropped where its middle now sits among the lane's other clips.
        const mid = clip.start + clip.dur / 2 + dx / px;
        const others = lane.clips.filter((c) => c.id !== clip.id);
        const index = others.filter((c) => c.start + c.dur / 2 < mid).length;
        if (onReorder) onReorder(clip.id, index);
        draw();
      },
    }));
    return cell;
  }

  function draw() {
    disposers.forEach((d) => d());
    disposers = [];
    const span = Math.max(spanOf(), ((node.clientWidth || 0) - 96) / px);       // the ruler runs to the window's edge, as an editor's does
    content.style.width = `${span * px}px`;
    drawRuler(span);
    gutter.replaceChildren(el("div", { cls: "cx-nle-gutter-ruler" }));
    laneBox.replaceChildren();
    for (const lane of lanes) {
      gutter.append(el("div", { cls: ["cx-nle-gutter-lane", `cx-nle-lane-${lane.kind}`], text: lane.label }));
      const row = el("div", { cls: ["cx-nle-lane", `cx-nle-lane-${lane.kind}`], attrs: { role: "listbox", "aria-label": lane.label }, children: lane.clips.map((c) => clipNode(lane, c)) });
      laneBox.append(row);
    }
    head.style.insetInlineStart = `${Math.max(0, playhead) * px}px`;
  }

  // The ruler and every lane's empty space seek; a clip's own press selects instead.
  content.addEventListener("click", (e) => {
    if (e.target.closest(".cx-nle-clip") || !onSeek) return;
    onSeek(Math.min(spanOf(), timeAt(e.clientX)));
  });

  draw();
  // The ruler follows the width it is given.
  let observer = null;
  if (typeof ResizeObserver !== "undefined") { observer = new ResizeObserver(() => draw()); observer.observe(node); }
  return {
    node,
    get span() { return spanOf(); },
    get pxPerSecond() { return px; },
    timeAt,
    setLanes(next = []) { lanes = next; draw(); },
    setZoom(next) { px = Math.max(4, next); draw(); },
    /** `follow`: scroll along when the playhead leaves the visible part (while playing). */
    setPlayhead(seconds, follow = false) {
      playhead = seconds; head.style.insetInlineStart = `${Math.max(0, seconds) * px}px`;
      if (follow) { const x = Math.max(0, seconds) * px + gutter.offsetWidth, w = node.clientWidth; if (x > node.scrollLeft + w - 24 || x < node.scrollLeft + gutter.offsetWidth) node.scrollLeft = Math.max(0, x - w * 0.2); }
    },
    destroy() { disposers.forEach((d) => d()); if (observer) observer.disconnect(); node.remove(); },
  };
});
