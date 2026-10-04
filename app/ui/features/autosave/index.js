// Autosave delay: how long after an edit the project is written, never while a key was pressed a moment ago (shell/project.js).
import { composer as c } from "../../composer/composer.js";

const CHOICES = [1, 3, 10, 30, 60];

export default {
  id: "autosave",
  mount: "menubar.menus",
  needs: ["project"],
  setup({ app }) {
    const p = app.project;
    app.editorSettings.push(() => c.settingsRow.default({ label: "Save after", hint: "Seconds after an edit before the project is written. Waits while you are typing.",
      control: c.select.md({ label: "Save after", value: String(p.pref("autosave_sec", 3)),
        options: CHOICES.map((n) => ({ value: String(n), label: `${n} s` })), onChange: (v) => p.setPref("autosave_sec", Number(v)) }) }));
  },
};
