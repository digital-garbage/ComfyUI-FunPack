// v4's editing keys: undo/redo, delete the selected scene, +/- zoom the timeline, space/K/L play and pause, J back a second, S split. Scene keys do nothing in Simple mode, where no timeline shows.
import * as edits from "../../shell/edits.js";
import { segments } from "../../shell/scenes.js";

export default {
  id: "keys",
  mount: "menubar.menus",
  needs: ["project", "selection", "keys", "mode"],
  setup({ app }) {
    const p = app.project, sel = app.selection;
    const scene = (fn) => () => (app.mode.now === "simple" ? false : fn());
    const frame = () => (p.project && p.project.frame_rate) || 25;
    const edge = (which) => {       // I / O: the playhead to the picked clip's in / out point
      const s = p.project && segments(p.project).find((x) => x.kind === "scene" && x.id === sel.focus);
      if (!s) return false;
      app.playhead.set(which === "start" ? s.start : s.start + s.dur);
    };
    const remove = scene(() => sel.ids.length ? sel.ids.forEach((id) => p.edit((pr) => edits.removeScene(pr, id))) : false);
    const off = [
      app.keys.bind("mod+z", () => p.undo()),
      app.keys.bind("shift+mod+z", () => p.redo()),
      app.keys.bind("mod+y", () => p.redo()),
      app.keys.bind("delete", remove), app.keys.bind("backspace", remove),
      ...["+", "="].map((k) => app.keys.bind(k, scene(() => app.say("zoom.in")))),
      app.keys.bind("-", scene(() => app.say("zoom.out"))),
      app.keys.bind(" ", (e) => (e.target.closest && e.target.closest("button, a, [role=slider]") ? false : app.say("play.toggle"))), app.keys.bind("k", () => app.say("play.pause")), app.keys.bind("l", () => app.say("play.start")),
      app.keys.bind("j", () => app.playhead.set(app.playhead.at - 1)),
      app.keys.bind("s", scene(() => app.say("split"))),
      app.keys.bind("arrowleft", scene(() => app.playhead.set(app.playhead.at - 1 / frame()))), app.keys.bind("arrowright", scene(() => app.playhead.set(app.playhead.at + 1 / frame()))),
      app.keys.bind("i", scene(() => edge("start"))), app.keys.bind("o", scene(() => edge("end"))),
      app.keys.bind("mod+,", () => (app.openSettings ? app.openSettings() : false)),
    ];
    return () => off.forEach((f) => f());
  },
};
