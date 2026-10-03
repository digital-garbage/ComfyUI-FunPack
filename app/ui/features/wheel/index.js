// The picker wheel: press the middle mouse button anywhere and every offered action fans out under the pointer; release on one to run it.
import { composer as c } from "../../composer/composer.js";
import { offer } from "../../shell/actions.js";

export default {
  id: "wheel",
  mount: "menubar.menus",
  needs: ["project"],
  setup({ app }) {
    let wheel = null;
    const offs = [
      offer(app, { id: "undo", label: "Undo", icon: "↶", run: () => app.project.undo() }),
      offer(app, { id: "redo", label: "Redo", icon: "↷", run: () => app.project.redo() }),
      offer(app, { id: "models", label: "Models & Pipeline", icon: "▤", run: () => app.openSettings && app.openSettings("models") }),
      offer(app, { id: "engine", label: "Engine settings", icon: "⚙", run: () => app.openSettings && app.openSettings("engine") }),
    ];
    const onDown = (e) => {
      if (e.button !== 1 || wheel || document.querySelector('[role="dialog"]')) return;       // nothing behind a dialog
      e.preventDefault();
      const items = (app.actions || []).map((a) => ({ id: a.id, label: a.label, icon: a.icon }));
      wheel = c.wheel.picker({ items, x: e.clientX, y: e.clientY, onClose: () => { wheel = null; },
        onPick: (id) => { const a = (app.actions || []).find((x) => x.id === id); if (a) Promise.resolve().then(() => a.run()).catch((err) => c.toast.warn({ text: err.message })); } });
    };
    document.addEventListener("mousedown", onDown, true);
    return () => { document.removeEventListener("mousedown", onDown, true); if (wheel) wheel.close && wheel.close(); offs.forEach((f) => f()); };
  },
};
