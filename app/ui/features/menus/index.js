// The File and Edit menus of the menubar.
import { composer as c } from "../../composer/composer.js";
import { list, remove } from "../../shell/project.js";

export default {
  id: "menus",
  mount: "menubar.menus",
  needs: ["project"],
  setup({ host, app }) {
    const p = app.project;
    const ask = (title, value, confirmLabel) => c.modal.prompt({ title, label: "Name", value, confirmLabel }).result;

    const actions = {
      new: async () => { const name = await ask("New project", "Untitled"); if (name) await p.newProject(name.trim() || "Untitled"); },
      rename: async () => { const name = await ask("Rename project", p.project.name, "Rename"); if (name) p.rename(name); },
      delete: async () => {
        const open = p.project;
        if (!(await c.modal.dialogue({ title: "Delete project", message: `Delete "${open.name}" for good?`, tone: "danger", confirmLabel: "Delete" }).result)) return;
        await p.flush();
        const next = (await list()).find((r) => r.id !== open.id);     // asked before deleting, so a failure leaves the project alone
        await remove(open.id);
        if (next) await p.open(next.id); else await p.newProject("Untitled");
      },
      undo: () => p.undo(),
      redo: () => p.redo(),
    };
    const menus = {
      File: () => [{ id: "new", label: "New project…" }, { id: "rename", label: "Rename…", disabled: !p.project },
        { separator: true }, { id: "delete", label: "Delete project…", danger: true, disabled: !p.project }],
      Edit: () => [{ id: "undo", label: "Undo", disabled: !p.canUndo }, { id: "redo", label: "Redo", disabled: !p.canRedo }],
    };
    for (const [label, items] of Object.entries(menus)) {
      const button = c.button.menu({ label, tone: "ghost", onClick: () => c.menu.dropdown({ anchor: button, items: items(), onPick: (id) => Promise.resolve(actions[id]()).catch((err) => c.toast.warn({ text: err.message })) }) });
      host.append(button.node);
    }
  },
};
