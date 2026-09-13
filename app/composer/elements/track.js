// A single time-based track: clips laid out by real duration, a ruler in
// seconds, and a playhead -- the shape of an actual video timeline, not a
// filmstrip of equal-sized thumbnails with a tick per item.
//
// One track, on purpose: v5 generates one clip per scene, in order, with no
// separate audio/overlay layers to line up against it. A second track is a
// different, larger feature (waveforms, compositing, independent trims) --
// this is only ever asked to place things that already play back to back.

import { define } from "../internals/register.js";
import { el } from "../internals/el.js";
import { wireDragReorder } from "../internals/dragReorder.js";

/** A clip's face: the result so far, or the glyph that stands in for one --
 *  the same visual language cx-cell-thumb uses elsewhere in the kit, but
 *  filling the clip's own fixed-height box rather than a 16:10 box of its
 *  own: a clip's width tracks real duration, and that box can run to
 *  hundreds of pixels wide. */
function faceOf(item) {
  const thumb = el("div", { cls: "cx-track-face" });
  const glyph = () => el("span", { cls: "cx-cell-glyph", text: item.icon || "▦", attrs: { "aria-hidden": "true" } });
  if (item.thumb) {
    const img = el("img", { cls: "cx-cell-img", attrs: { src: item.thumb, alt: "", loading: "lazy" } });
    // A thumbnail that fails to load falls back to the glyph, not the
    // browser's broken-image icon -- which would read as a damaged file.
    img.addEventListener("error", () => img.replaceWith(glyph()));
    thumb.append(img);
  } else {
    thumb.append(glyph());
  }
  if (item.badge) thumb.append(el("span", { cls: "cx-cell-badge", text: item.badge }));
  return thumb;
}

// A tick every 1/2/5/10... seconds, whichever is the smallest that still
// leaves room to read the label beside it. Fixed intervals rather than a tick
// per clip: what a person reads off a ruler is elapsed TIME, and a tick per
// scene is the scene-index ruler this replaces.
const NICE_INTERVALS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
const MIN_TICK_PX = 56;

function tickInterval(pxPerSecond) {
  for (const s of NICE_INTERVALS) if (s * pxPerSecond >= MIN_TICK_PX) return s;
  return NICE_INTERVALS[NICE_INTERVALS.length - 1];
}

function formatTime(seconds) {
  const whole = Math.max(0, Math.round(seconds));
  const m = Math.floor(whole / 60);
  const s = whole % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

// A floor, not a cap: a scene under a second would otherwise render as a
// sliver nothing could click or even see next to a normal one. Everything
// else stays exactly proportional -- this only rescues the degenerate case.
const MIN_CLIP_PX = 24;

/**
 * track.default({ items, pxPerSecond, selection, playhead, label,
 *                  onSeek, onReorder })
 *
 * `items`: [{ id, label, thumb, icon, badge, rating, excluded, start,
 *             duration }] -- start/duration in SECONDS, contiguous (each
 * item's start is the previous one's start + duration), which is what lets
 * one click handler double as both "seek" and "select": every point on the
 * track belongs to exactly one clip. The track's own span is read off the
 * items themselves (the last one's start + duration) -- nothing else here
 * knows the project's length.
 *
 * `onSeek(seconds, item)` fires on any click on the track -- the ruler, the
 * empty background, or a clip. `item` is whichever clip's span the point
 * (or, for a clip itself, the clip clicked) falls in.
 */
define("track", "default", ({
  items = [], pxPerSecond: initialPxPerSecond = 40, selection = [], playhead = 0,
  label, onSeek, onReorder,
} = {}) => {
  let pxPerSecond = initialPxPerSecond;
  const chosen = new Set(selection);
  let current = items;
  const spanOf = () => current.reduce((max, i) => Math.max(max, i.start + i.duration), 0);

  const node = el("div", { cls: "cx-track", attrs: { role: "presentation", "aria-label": label } });
  const inner = el("div", { cls: "cx-track-inner" });
  const ruler = el("div", { cls: "cx-track-ruler", attrs: { role: "presentation" } });
  const clips = el("div", { cls: "cx-track-clips", attrs: { role: "listbox", "aria-label": label } });
  const head = el("div", { cls: "cx-track-playhead", attrs: { "aria-hidden": "true" } });
  inner.append(ruler, clips, head);
  node.append(inner);

  function drawRuler(span) {
    ruler.replaceChildren();
    const interval = tickInterval(pxPerSecond);
    for (let t = 0; t <= span; t += interval) {
      const tick = el("span", { cls: "cx-track-tick", text: formatTime(t) });
      tick.style.insetInlineStart = `${t * pxPerSecond}px`;
      ruler.append(tick);
    }
  }

  function drawClips() {
    clips.replaceChildren();
    for (const item of current) {
      const on = chosen.has(item.id);
      const cell = el("button", {
        cls: ["cx-track-clip", on ? "cx-on" : null, item.rating ? "cx-rated" : null,
              item.excluded ? "cx-excluded" : null, "cx-focusable"],
        attrs: {
          type: "button", role: "option", "aria-selected": String(on), title: item.label,
          "aria-label": item.label, "data-rating": item.rating || undefined,
          "data-id": item.id,
          // Draggable whether or not `onReorder` is given, matching the
          // strip: a drag with nowhere to report to is inert, not wired.
          draggable: "true",
        },
        children: [faceOf(item)],
      });
      cell.style.width = `${Math.max(MIN_CLIP_PX, item.duration * pxPerSecond)}px`;
      clips.append(cell);
    }
  }

  function setPlayhead(seconds) {
    head.style.insetInlineStart = `${Math.max(0, seconds) * pxPerSecond}px`;
  }

  function draw() {
    const span = spanOf();
    inner.style.width = `${span * pxPerSecond}px`;
    drawRuler(span);
    drawClips();
    setPlayhead(playhead);
  }

  wireDragReorder(clips, ".cx-track-clip", onReorder);

  // One handler for the whole track: a click's OWN position is turned into a
  // time, the same way a scrub bar would, whether it landed on the ruler, the
  // empty background, or a clip -- a clip covers most of the track's area, so
  // reading its id instead of where on it the click landed would make the
  // ruler the only place a real, exact-second seek ever reaches. Keyboard
  // activation (Enter/Space on a focused clip) still resolves correctly: the
  // browser reports the click at that button's own on-screen position, which
  // is well inside the right clip's span even if not at its exact start.
  inner.addEventListener("click", (e) => {
    if (!onSeek) return;
    const rect = inner.getBoundingClientRect();
    const seconds = Math.min(spanOf(), Math.max(0, (e.clientX - rect.left) / pxPerSecond));
    const item = current.find((i) => seconds < i.start + i.duration) || current[current.length - 1] || null;
    onSeek(seconds, item);
  });

  draw();

  return {
    node,
    setItems(next = []) { current = next; draw(); },
    setValue(next = []) { chosen.clear(); for (const id of next) chosen.add(id); drawClips(); },
    setPlayhead(seconds) { playhead = seconds; setPlayhead(seconds); },
    setZoom(next) { pxPerSecond = next; draw(); },
    destroy: () => node.remove(),
  };
});
