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
      if (zone) { zone.classList.remove(DRAG_CLASS); zone.classList.remove(OPEN_CLASS); }
    }
    return !!on;
  }

  function get() { return stored(); }

  function set(on) {
    persist(!!on);
    const result = apply(!!on);
    if (on) { ensureStrip(); measure(); }   // first time the strip and its height matter
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

  const STRIP_ID = "timeline-peek-strip";
  const OPEN_CLASS = "peek-open";
  const STRIP_LABEL = "Timeline — hover to show";

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

  // A real element, not a ::before on the body. Showing a sliver of the closed timeline read
  // as a broken timeline — a lane label and a cut-off clip — so the closed state shows a
  // labelled bar instead and the body is hidden outright. It also has to be its own element
  // because the pointer must be able to be ON the strip without being on the header.
  function ensureStrip(doc) {
    const d = doc || document;
    const zone = d.getElementById("timeline-zone");
    const body = d.getElementById("timeline-body");
    if (!zone || !body) return null;
    let strip = d.getElementById(STRIP_ID);
    if (!strip) {
      strip = d.createElement("div");
      strip.id = STRIP_ID;
      strip.className = "tl-peek-strip";
      strip.textContent = STRIP_LABEL;
      body.parentNode.insertBefore(strip, body);
    }
    return strip;
  }

  // Which part of the zone the pointer is over decides everything: the header is never a
  // trigger (reaching for Generate must not open the timeline), the strip and the timeline
  // itself both are — the body has to keep it open or it would shut the moment it opened
  // under the cursor.
  function opensOnPointer(target, doc) {
    const d = doc || document;
    const zone = d.getElementById("timeline-zone");
    if (!zone || !target || !target.closest) return false;
    if (!zone.contains(target)) return false;
    return !target.closest(".zone-head");
  }

  // The closed body is display:none, so the lanes have no width to lay out against while it
  // is shut. Telling the app to re-measure on the way open costs one resize handler and
  // avoids clips drawn at a stale width for the first frame after it appears.
  let reflowQueued = false;
  function markReflow(doc) {
    if (reflowQueued || typeof window === "undefined" || !window.dispatchEvent) return;
    reflowQueued = true;
    (window.requestAnimationFrame || setTimeout)(() => {
      reflowQueued = false;
      try { window.dispatchEvent(new Event("resize")); } catch (_) { /* older engines */ }
    });
  }

  function install(doc) {
    const d = doc || document;
    const zone = d.getElementById("timeline-zone");
    if (!zone) return () => {};
    const track = makeDragTracker();
    ensureStrip(d);
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

    // Hover is driven from script rather than :hover because the two halves of "open" are
    // different elements — the strip you point at and the timeline that replaces it — and a
    // CSS rule on either one alone flickers as the other takes the pointer.
    const openIf = (e) => {
      const want = opensOnPointer(e.target, d);
      zone.classList.toggle(OPEN_CLASS, want);
      if (want) markReflow(d);
    };
    const closeAll = () => zone.classList.remove(OPEN_CLASS);
    zone.addEventListener("mouseover", openIf);
    zone.addEventListener("mouseleave", closeAll);

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
      zone.removeEventListener("mouseover", openIf);
      zone.removeEventListener("mouseleave", closeAll);
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

  return { get, set, apply, install, measure, ensureStrip, opensOnPointer, isDrag,
           makeDragTracker, LS_KEY, ATTR, DRAG_CLASS, OPEN_CLASS, STRIP_ID, STRIP_LABEL };
}));
