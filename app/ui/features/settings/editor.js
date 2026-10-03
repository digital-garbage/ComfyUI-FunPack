// Editor: how the editor behaves. Each feature that has a preference puts a section on app.editorSettings; nothing is named here.
import { composer as c } from "../../composer/composer.js";

export const editor = (app) => function mount() {
  const parts = app.editorSettings.map((section) => section());
  if (parts.length && !app.project.project) parts.unshift(c.banner.info({ text: "Open a project first: these preferences are kept with the project, so nothing here can be changed without one." }));
  return parts.length ? c.region.stack({ gap: "lg", children: parts }) : c.emptyState.default({ icon: "☰", title: "Nothing to set", hint: "No editor feature has a preference to show." });
};
