// "Auto-hide": the timeline sits as its title bar plus a labelled strip, and opens to full height when the pointer rests on
// the strip or the timeline. As v4: the title bar never opens it (reaching for Generate or Rate must not throw it open), a
// pass over the strip on the way somewhere does not flash it (a short beat first), and a drag opens it at once so a lane
// can be hit. A preference of this screen, kept in this browser.
import { composer as c } from "../../composer/composer.js";

const KEY = "funpack_timeline_peek", BEAT_MS = 220;

export default {
  id: "peek",
  mount: "timeline.status",
  setup({ host }) {
    const root = document.documentElement;
    const zone = () => document.querySelector(".fp-main > :nth-child(2)");
    let on = false, timer = null, dragging = false;
    try { on = localStorage.getItem(KEY) === "1"; } catch { /* the plain layout */ }
    const strip = Object.assign(document.createElement("div"), { className: "fp-peek-strip", textContent: "Timeline · hover to show" });
    const place = () => { const z = zone(), body = z && z.querySelector(".cx-panel-body"); if (body && strip.parentNode !== z) z.insertBefore(strip, body); };
    // the closed height (title bar + strip, which may wrap) is what the row above leaves room for
    const sizer = typeof ResizeObserver === "function" ? new ResizeObserver(() => {
      const z = zone();
      if (z && !root.classList.contains("fp-peek-open")) root.style.setProperty("--peek-h", z.offsetHeight + "px");
    }) : null;
    const paint = () => { root.classList.toggle("fp-peek", on); place(); const z = zone(); if (sizer && z) sizer.observe(z); };
    paint();
    const tab = c.button.sm({ label: "Auto-hide", tone: "neutral", pressed: on, title: "Show the timeline only while the pointer is on it", onClick: () => {
      on = !on; paint(); tab.setPressed && tab.setPressed(on);
      try { on ? localStorage.setItem(KEY, "1") : localStorage.removeItem(KEY); } catch { /* lasts until reload */ }
    } });
    host.append(tab.node);
    const open = () => { clearTimeout(timer); timer = null; root.classList.add("fp-peek-open"); };
    const shut = () => { clearTimeout(timer); timer = null; if (!dragging) root.classList.remove("fp-peek-open"); };
    const over = (e) => {
      if (!on) return;
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
    document.addEventListener("pointerover", over); document.addEventListener("pointerout", left);
    document.addEventListener("dragenter", dragIn); document.addEventListener("dragleave", dragOut);
    document.addEventListener("drop", dragDone); document.addEventListener("dragend", dragDone);
    return () => {
      document.removeEventListener("pointerover", over); document.removeEventListener("pointerout", left);
      document.removeEventListener("dragenter", dragIn); document.removeEventListener("dragleave", dragOut);
      document.removeEventListener("drop", dragDone); document.removeEventListener("dragend", dragDone);
      clearTimeout(timer); if (sizer) sizer.disconnect();
      root.classList.remove("fp-peek", "fp-peek-open"); tab.node.remove(); strip.remove();
    };
  },
};
