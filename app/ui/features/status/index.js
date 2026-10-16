// The two chips at the far end of the menubar: whether the project is saved, and whether ComfyUI answers.
import { composer as c } from "../../composer/composer.js";
import { inPlace } from "../../shell/place.js";

export default {
  id: "status",
  mount: "menubar.right",
  needs: ["project", "api"],
  setup({ host, app }) {
    const put = inPlace(host);
    let live = true;
    const draw = () => put((app.project.unsaved ? c.chip.warn({ label: "unsaved" }) : c.chip.neutral({ label: "saved" })).node,
      (live ? c.chip.good({ label: "ComfyUI live", dot: true }) : c.chip.danger({ label: "ComfyUI offline", dot: true })).node);
    const ping = async () => { try { live = Boolean((await app.api.health()).ok); } catch { live = false; } draw(); };
    draw(); ping();
    const timers = [setInterval(() => { const now = Boolean(app.project.unsaved); if (now !== saved) { saved = now; draw(); } }, 600), setInterval(ping, 10000)];
    let saved = Boolean(app.project.unsaved);
    const off = app.on(() => { saved = Boolean(app.project.unsaved); draw(); });
    return () => { timers.forEach(clearInterval); off(); };
  },
};
