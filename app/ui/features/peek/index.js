// "Auto-hide": the timeline sits as a strip (its title bar) and opens to full height while the pointer is on it.
// A preference of this screen, kept in this browser. It also opens for the length of a drag, so dropping onto a lane still works.
import { composer as c } from "../../composer/composer.js";

const KEY = "funpack_timeline_peek";

export default {
  id: "peek",
  mount: "timeline.status",
  setup({ host }) {
    const root = document.documentElement;
    let on = false;
    try { on = localStorage.getItem(KEY) === "1"; } catch { /* the plain layout */ }
    const paint = () => root.classList.toggle("fp-peek", on);
    paint();
    const tab = c.button.sm({ label: "Auto-hide", tone: "neutral", pressed: on, title: "Show the timeline only while the pointer is on it", onClick: () => {
      on = !on; paint(); tab.setPressed && tab.setPressed(on);
      try { on ? localStorage.setItem(KEY, "1") : localStorage.removeItem(KEY); } catch { /* lasts until reload */ }
    } });
    host.append(tab.node);
    // A drag in progress holds it open.
    const open = () => root.classList.add("fp-peek-open"), shut = () => root.classList.remove("fp-peek-open");
    const out = (e) => { if (!e.relatedTarget) shut(); };
    document.addEventListener("dragenter", open); document.addEventListener("dragleave", out);
    document.addEventListener("drop", shut); document.addEventListener("dragend", shut);
    return () => {
      document.removeEventListener("dragenter", open); document.removeEventListener("dragleave", out);
      document.removeEventListener("drop", shut); document.removeEventListener("dragend", shut);
      root.classList.remove("fp-peek", "fp-peek-open"); tab.node.remove();
    };
  },
};
