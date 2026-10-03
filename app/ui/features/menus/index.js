// The menubar: File, Edit, View, Help. What is not built yet is listed and greyed ("soon"), so the bar reads as v4's.
import { composer as c } from "../../composer/composer.js";
import { list, remove } from "../../shell/project.js";
import { segments } from "../../shell/scenes.js";
import * as edits from "../../shell/edits.js";

const soon = (label, hint = "soon") => ({ id: "-", label, hint, disabled: true });
const sceneIds = (p) => segments(p).filter((s) => s.kind === "scene").map((s) => s.id);

export default {
  id: "menus",
  mount: "menubar.menus",
  needs: ["project", "selection"],
  setup({ host, app }) {
    const p = app.project, sel = app.selection;
    const ask = (title, value, confirmLabel) => c.modal.prompt({ title, label: "Name", value, confirmLabel }).result;
    const move = (by) => p.edit((pr) => { const order = sceneIds(pr), at = order.indexOf(sel.focus); return at >= 0 && edits.reorder(pr, sel.focus, at + by, order); });

    const actions = {
      new: async () => { const name = await ask("New project", "Untitled"); if (name) await p.newProject(name.trim() || "Untitled"); },
      delete: async () => {
        const open = p.project;
        if (!(await c.modal.dialogue({ title: "Delete project", message: `Delete "${open.name}" for good?`, tone: "danger", confirmLabel: "Delete" }).result)) return;
        await p.flush();
        const next = (await list()).find((r) => r.id !== open.id);     // asked before deleting, so a failure leaves the project alone
        await remove(open.id);
        if (next) await p.open(next.id); else await p.newProject("Untitled");
      },
      undo: () => p.undo(), redo: () => p.redo(),
      add: () => { const sc = p.edit((pr) => edits.addScene(pr)); if (sc) p.select(sc.id); },
      del: () => sel.ids.forEach((id) => p.edit((pr) => edits.removeScene(pr, id))),
      left: () => move(-1), right: () => move(1),
      reset: () => app.resetLayout(),
      welcome: () => app.say("welcome.open"),
      exclude: () => { const sc = p.selected; if (sc) p.setScene(sc.id, "excluded", !sc.excluded); },
    };
    const menus = {
      File: async () => {
        const recent = await list().catch(() => []);
        return [{ id: "new", label: "New Project", hint: "⌘N" }, soon("Project Setup Wizard…", "theme · model · tour"), { separator: true },
          { heading: "Open recent" }, ...(recent.length ? recent.slice(0, 8).map((r) => ({ id: `open:${r.id}`, label: r.name })) : [{ id: "-", label: "No projects", disabled: true }]),
          { separator: true }, soon("Save Project File…", "⬇"), soon("Load Project File…"), { separator: true }, soon("Import Media…"),
          { id: "delete", label: "Delete Current Project", danger: true, disabled: !p.project }];
      },
      Edit: () => [{ id: "undo", label: "Undo", hint: "⌘Z", disabled: !p.canUndo }, { id: "redo", label: "Redo", hint: "⇧⌘Z", disabled: !p.canRedo }, { separator: true },
        { id: "add", label: "Add Scene", hint: "+", disabled: !p.project }, { id: "del", label: "Delete Scene", disabled: !sel.ids.length }, { separator: true },
        { id: "left", label: "Move Clip Left", hint: "timeline cut", disabled: !sel.focus }, { id: "right", label: "Move Clip Right", hint: "timeline cut", disabled: !sel.focus }, { separator: true },
        { id: "exclude", label: "Toggle Exclude", disabled: !sel.focus }],
      View: () => [soon("Refresh Preview", ""), { id: "reset", label: "Reset Layout" }],
      Help: () => [soon("Restart tour", ""), soon("Skip to FAQ", ""), soon("Exit tour", ""), { separator: true }, { id: "welcome", label: "Welcome tour…", hint: "?" }],
    };
    const pick = (id) => (id.startsWith("open:") ? p.open(id.slice(5)) : actions[id] && actions[id]());
    for (const [label, items] of Object.entries(menus)) {
      const button = c.button.menu({ label, tone: "ghost", onClick: async () => c.menu.dropdown({ anchor: button, items: await items(),
        onPick: (id) => Promise.resolve(pick(id)).catch((err) => c.toast.warn({ text: err.message })) }) });
      host.append(button.node);
    }
  },
};
