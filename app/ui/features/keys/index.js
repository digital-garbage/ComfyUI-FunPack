// v4's editing keys: undo/redo, new project, add and delete a scene.
import * as edits from "../../shell/edits.js";

export default {
  id: "keys",
  mount: "menubar.menus",
  needs: ["project", "selection", "keys"],
  setup({ app }) {
    const p = app.project, sel = app.selection;
    const off = [
      app.keys.bind("mod+z", () => p.undo()),
      app.keys.bind("shift+mod+z", () => p.redo()),
      app.keys.bind("mod+y", () => p.redo()),
      app.keys.bind("+", () => { if (!p.project) return false; const sc = p.edit((pr) => edits.addScene(pr)); if (sc) p.select(sc.id); }),
      app.keys.bind("delete", () => sel.ids.length ? sel.ids.forEach((id) => p.edit((pr) => edits.removeScene(pr, id))) : false),
    ];
    return () => off.forEach((f) => f());
  },
};
