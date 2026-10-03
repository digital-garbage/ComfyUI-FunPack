// The welcome screen: shown the first time this browser opens the app, and from Help ▸ Welcome tour.
import { composer as c } from "../../composer/composer.js";

const SEEN = "funpack_welcomed";
const card = (title, hint) => c.field.default({ label: title, hint, control: c.button.sm({ label: title, tone: "ghost", disabled: true }) });

export default {
  id: "welcome",
  mount: "menubar.right",
  needs: ["project"],
  setup({ app }) {
    let win = null;
    const seen = () => { try { return localStorage.getItem(SEEN) === "1"; } catch { return true; } };
    const markSeen = () => { try { localStorage.setItem(SEEN, "1"); } catch { /* private window */ } };
    const open = () => {
      if (win) return;
      const recent = app.project.project;
      const done = () => { markSeen(); if (win) win.close("done"); };
      win = c.modal.generic({
        title: "Welcome to FunPack", size: "lg",
        body: c.region.stack({ gap: "md", children: [
          c.hint.default({ text: "Multi-scene video on a real timeline." }),
          c.button.lg({ label: "Begin", tone: "primary", onClick: done }),
          ...(recent ? [c.button.sm({ label: `Continue with “${recent.name}”`, tone: "ghost", onClick: done })] : []),
          c.button.sm({ label: "Load an existing project", tone: "ghost", disabled: true }),
          c.toolbar.default({ items: [card("Update", "Check this install for a newer version"), card("Switch branch", "Pick another branch"),
            card("Restart ComfyUI", "Reload the server without updating"), card("Rollback update", "Undo the last update")] }),
        ] }),
        onClose: () => { markSeen(); win = null; },
      });
    };
    const timer = seen() ? 0 : setTimeout(open, 600);                 // after the first paint, so the screen it sits over is already there
    const off = app.on((what) => { if (what === "welcome.open") open(); });
    return () => { clearTimeout(timer); off(); };
  },
};
