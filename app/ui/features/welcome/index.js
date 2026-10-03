// The welcome screen: shown the first time this browser opens the app, and from Help ▸ Welcome tour.
import { composer as c } from "../../composer/composer.js";
import { list } from "../../shell/project.js";

const SEEN = "funpack_welcomed";
const card = (title, hint, onClick) => c.field.default({ label: title, hint, control: c.button.sm({ label: title, tone: "ghost", onClick }) });

export default {
  id: "welcome",
  mount: "menubar.right",
  needs: ["project", "maintenance"],
  setup({ app }) {
    let win = null;
    const seen = () => { try { return localStorage.getItem(SEEN) === "1"; } catch { return true; } };
    const markSeen = () => { try { localStorage.setItem(SEEN, "1"); } catch { /* private window */ } };
    const load = async () => {
      let all; try { all = await list(); } catch (err) { return c.toast.warn({ text: err.message }); }
      if (!all.length) return c.toast.warn({ text: "There are no saved projects yet." });
      const id = await c.modal.choice({ title: "Load a project", items: all.map((r) => ({ id: r.id, label: r.name })) }).result;
      if (!id) return;
      try { await app.project.open(id); markSeen(); if (win) win.close("done"); } catch (err) { c.toast.warn({ text: err.message }); }
    };
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
          c.button.sm({ label: "Load an existing project", tone: "ghost", onClick: load }),
          c.toolbar.default({ items: [card("Update", "Check this install for a newer version", () => app.maintenance.update()), card("Switch branch", "Pick another branch", () => app.maintenance.switchBranch()),
            card("Restart ComfyUI", "Reload the server without updating", () => app.maintenance.restart()), card("Rollback update", "Undo the last update", () => app.maintenance.rollback())] }),
        ] }),
        onClose: () => { markSeen(); win = null; },
      });
    };
    const timer = seen() ? 0 : setTimeout(open, 600);                 // after the first paint, so the screen it sits over is already there
    const off = app.on((what) => { if (what === "welcome.open") open(); });
    return () => { clearTimeout(timer); off(); };
  },
};
