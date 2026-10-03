// Editor: how the editor behaves. Each feature that has a preference puts a section on app.editorSettings; nothing is named here.
import { composer as c } from "../../composer/composer.js";

export const editor = (app) => function mount() {
  const parts = app.editorSettings.map((section) => section());
  return parts.length ? c.region.stack({ gap: "lg", children: parts }) : c.emptyState.default({ icon: "☰", title: "Nothing to set", hint: "No editor feature has a preference to show." });
};
