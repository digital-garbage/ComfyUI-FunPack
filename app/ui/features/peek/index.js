// "Auto-hide": the button that makes the timeline show only while the pointer is on it. The behaviour is the frame's (shell/peek.js).
// A preference of this screen, kept in this browser.
import { composer as c } from "../../composer/composer.js";

const KEY = "funpack_timeline_peek";

export default {
  id: "peek",
  mount: "timeline.status",
  needs: ["peek"],
  setup({ host, app }) {
    let on = false;
    try { on = localStorage.getItem(KEY) === "1"; } catch { /* the plain layout */ }
    app.peek.set(on);
    const tab = c.button.sm({ label: "Auto-hide", tone: "neutral", pressed: on, title: "Show the timeline only while the pointer is on it", onClick: () => {
      on = !on; app.peek.set(on); tab.setPressed(on);
      try { on ? localStorage.setItem(KEY, "1") : localStorage.removeItem(KEY); } catch { /* lasts until reload */ }
    } });
    host.append(tab.node);
    return () => { app.peek.set(false); tab.destroy(); };
  },
};
