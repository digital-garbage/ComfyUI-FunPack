// v4's editing keys: undo/redo, delete the selected scene, +/- zoom the timeline. Scene keys do nothing in Simple mode, where no timeline shows.
import * as edits from "../../shell/edits.js";

export default {
  id: "keys",
  mount: "menubar.menus",
  needs: ["project", "selection", "keys", "mode"],
  setup({ app }) {
    const p = app.project, sel = app.selection;
    const scene = (fn) => () => (app.mode.now === "simple" ? false : fn());
    const remove = scene(() => sel.ids.length ? sel.ids.forEach((id) => p.edit((pr) => edits.removeScene(pr, id))) : false);
    const off = [
      app.keys.bind("mod+z", () => p.undo()),
      app.keys.bind("shift+mod+z", () => p.redo()),
      app.keys.bind("mod+y", () => p.redo()),
      app.keys.bind("delete", remove), app.keys.bind("backspace", remove),
      ...["+", "="].map((k) => app.keys.bind(k, scene(() => app.say("zoom.in")))),
      app.keys.bind("-", scene(() => app.say("zoom.out"))),
    ];
    return () => off.forEach((f) => f());
  },
};
