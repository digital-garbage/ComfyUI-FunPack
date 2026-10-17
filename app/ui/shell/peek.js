// "Auto-hide": the timeline sits as its title bar plus a labelled strip, and opens to full height when the pointer rests on
// the strip or the timeline. As v4: the title bar never opens it (reaching for Generate or Rate must not throw it open), a
// pass over the strip on the way somewhere does not flash it (a short beat first), and a drag opens it at once so a lane
// can be hit. It changes the frame's own layout, so it lives with the frame; the feature is only the button.
const BEAT_MS = 220;

export function createPeek(doc = document) {
  const root = doc.documentElement, win = doc.defaultView;
  const zone = () => doc.querySelector(".fp-main > :nth-child(2)");
  const strip = Object.assign(doc.createElement("div"), { className: "fp-peek-strip", textContent: "Timeline · hover to show" });
  let on = false, timer = null, dragging = false, stops = [];

  const place = () => { const z = zone(), body = z && z.querySelector(".cx-panel-body"); if (body && strip.parentNode !== z) z.insertBefore(strip, body); };
  // the closed height (title bar + strip, which may wrap) is what the row above leaves room for
  const sizer = typeof win.ResizeObserver === "function" ? new win.ResizeObserver(() => {
    const z = zone();
    if (z && !root.classList.contains("fp-peek-open")) root.style.setProperty("--peek-h", z.offsetHeight + "px");
  }) : null;
  const open = () => { clearTimeout(timer); timer = null; root.classList.add("fp-peek-open"); };
  const shut = () => { clearTimeout(timer); timer = null; if (!dragging) root.classList.remove("fp-peek-open"); };
  const over = (e) => {
    const t = e.target, z = zone();
    if (!t || !t.closest || !z) return;
    if (z.contains(t) && !t.closest(".cx-panel-head")) { if (!timer && !root.classList.contains("fp-peek-open")) timer = setTimeout(open, BEAT_MS); }
    else if (t.closest(".fp-main, .cx-frame-head")) shut();         // elsewhere on the frame; a menu or popover over it leaves it be
  };
  const left = (e) => { if (!e.relatedTarget) shut(); };            // out of the window
  // a drag opens it at once and holds it open until the drop
  const dragIn = () => { dragging = true; open(); };
  const dragOut = (e) => { if (!e.relatedTarget) { dragging = false; shut(); } };
  const dragDone = () => { dragging = false; shut(); };
  const listen = (type, fn) => { doc.addEventListener(type, fn); return () => doc.removeEventListener(type, fn); };

  function set(next) {
    next = Boolean(next);
    if (next === on) return;
    on = next;
    clearTimeout(timer); timer = null; dragging = false;
    root.classList.remove("fp-peek-open");
    root.classList.toggle("fp-peek", on);
    if (on) {
      place();
      const z = zone(); if (sizer && z) sizer.observe(z);
      stops = [listen("pointerover", over), listen("pointerout", left), listen("dragenter", dragIn), listen("dragleave", dragOut), listen("drop", dragDone), listen("dragend", dragDone)];
    } else {
      stops.forEach((f) => f()); stops = [];
      if (sizer) sizer.disconnect();
      strip.remove();
    }
  }
  return { set, get on() { return on; }, destroy: () => set(false) };
}
