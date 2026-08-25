// Timeline peek: collapse the timeline to a strip that opens when you point at it.
//
// A browser preference, not a project one — how much of the screen you want the timeline to
// take is a property of the screen you are sitting at, which is why it lives beside the
// colour scheme rather than in settings that travel with a project.
//
// The header is NOT part of the strip: it carries Generate, the pinned buttons and the
// Composer, and reaching for any of them must not throw the timeline open under the cursor.
// Only the body below it collapses, and only the body reacts to a pointer.
//
// The hover half is CSS. The part that needs script is the DRAG: while a file is being
// dragged from the media browser or the desktop, the pointer is in drag mode and a strip that
// stays shut is a target you cannot hit. dragenter opens it and holds it open for the whole
// drag, so dropping onto a lane works exactly as it does when the timeline is pinned open.
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;      // node --test
  if (root) root.TimelinePeek = api;
}(typeof window !== "undefined" ? window : null, function () {
  const LS_KEY = "funpack_timeline_peek";
  const ATTR = "data-timeline-peek";
  const DRAG_CLASS = "peek-drag-open";

  function stored() {
    try {
      return localStorage.getItem(LS_KEY) === "1";
    } catch (_) {
      return false;                     // private mode / storage denied: the plain layout
    }
  }

  function persist(on) {
    try {
      if (on) localStorage.setItem(LS_KEY, "1");
      else localStorage.removeItem(LS_KEY);
    } catch (_) { /* the attribute below still applies for this session */ }
  }

  // The drag payload decides nothing about whether to open: a drag that turns out to be
  // something the timeline cannot take still needs the strip open to say so. Anything with a
  // dataTransfer counts — files from the desktop, clips from the media browser, a lane's own
  // reorder — which is every drag that could be aimed at a lane.
  function isDrag(event) {
    return !!(event && event.dataTransfer);
  }

  function apply(on, doc) {
    const d = doc || (typeof document !== "undefined" ? document : null);
    if (!d || !d.documentElement) return !!on;
    if (on) d.documentElement.setAttribute(ATTR, "1");
    else {
      d.documentElement.removeAttribute(ATTR);
      // Leaving the preference while a drag happened to be open would strand the class on
      // the zone, and it only means anything alongside the attribute.
      const zone = d.getElementById && d.getElementById("timeline-zone");
      if (zone) zone.classList.remove(DRAG_CLASS);
    }
    return !!on;
  }

  function get() { return stored(); }

  function set(on) {
    persist(!!on);
    const result = apply(!!on);
    if (on) measure();          // turning it on is the first time the strip height matters
    return result;
  }

  // Depth counting, not a boolean: dragging across a lane fires dragleave for the element
  // being left AFTER dragenter for the one being entered, so a plain flag closes the strip
  // mid-drag as the pointer crosses from one clip to the next.
  function makeDragTracker() {
    let depth = 0;
    return {
      enter() { depth += 1; return depth > 0; },
      leave() { depth = Math.max(0, depth - 1); return depth > 0; },
      end() { depth = 0; return false; },
      get depth() { return depth; },
    };
  }

  // The header's height, measured rather than assumed: .zone-head WRAPS when the zone is
  // narrow (34px min-height, 138px in practice on a normal window). The zone itself sizes to
  // its contents now, so this is only what anything anchored to the timeline's height reads
  // to find the closed edge — a constant there would float a toast over the strip or leave a
  // gap under it.
  function measure(doc) {
    const d = doc || (typeof document !== "undefined" ? document : null);
    if (!d || !d.getElementById) return 0;
    const zone = d.getElementById("timeline-zone");
    const head = zone && zone.querySelector(".zone-head");
    if (!head || !head.getBoundingClientRect) return 0;
    const h = Math.ceil(head.getBoundingClientRect().height);
    if (h > 0) d.documentElement.style.setProperty("--timeline-peek-h", h + "px");
    return h;
  }

  function install(doc) {
    const d = doc || document;
    const zone = d.getElementById("timeline-zone");
    if (!zone) return () => {};
    const track = makeDragTracker();
    // Measured now and on resize: the header re-wraps as the window changes width, and a
    // stale strip height would either clip it or leave a gap under it.
    measure(d);
    const onResize = () => measure(d);
    if (typeof window !== "undefined" && window.addEventListener) {
      window.addEventListener("resize", onResize);
    }
    const paint = (open) => zone.classList.toggle(DRAG_CLASS, !!open);

    const onEnter = (e) => { if (isDrag(e)) paint(track.enter()); };
    const onLeave = (e) => { if (isDrag(e)) paint(track.leave()); };
    const onEnd = () => paint(track.end());

    zone.addEventListener("dragenter", onEnter);
    zone.addEventListener("dragleave", onLeave);
    zone.addEventListener("drop", onEnd);
    // A drag abandoned outside the window fires neither drop nor dragleave on the zone, and
    // the strip would stay open until the next pointer move. dragend on the document covers
    // the drag that started here; the window's own dragleave covers one that came from
    // outside it.
    d.addEventListener("dragend", onEnd);
    d.addEventListener("drop", onEnd);
    return () => {
      zone.removeEventListener("dragenter", onEnter);
      zone.removeEventListener("dragleave", onLeave);
      zone.removeEventListener("drop", onEnd);
      d.removeEventListener("dragend", onEnd);
      d.removeEventListener("drop", onEnd);
      if (typeof window !== "undefined" && window.removeEventListener) {
        window.removeEventListener("resize", onResize);
      }
    };
  }

  return { get, set, apply, install, measure, isDrag, makeDragTracker, LS_KEY, ATTR, DRAG_CLASS };
}));
