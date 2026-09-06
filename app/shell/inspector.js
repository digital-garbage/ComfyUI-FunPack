// The inspector: what the PROJECT is, and what THIS SCENE is.
//
// Two tabs because they are two different things and v4 proved that mixing them
// is how a per-scene value quietly becomes the project's. The rule this file
// exists to make visible: a regenerate reads the PROJECT, and a crop is the
// scene's own -- so Length appears in both places and means something different
// in each, and both say so.

import { composer } from "../composer/composer.js";

const RATINGS = [
  { value: "perfect", label: "Perfect" },
  { value: "good", label: "Good" },
  { value: "wrong", label: "Wrong" },
  { value: "awful", label: "Awful" },
];

export function createInspector({ project, onRename } = {}) {
  const host = composer.region.stack({ gap: "sm", fill: true });
  let tab = "scene";

  const tabs = composer.tabs.underline({
    value: tab, label: "Inspector",
    tabs: [{ value: "scene", label: "Scene" }, { value: "project", label: "Project" }],
    onChange: (next) => { tab = next; draw(); },
  });

  const rows = composer.region.stack({ gap: "sm", label: "Inspector rows" });
  host.set([tabs, rows]);

  function sceneRows() {
    const scene = project.selected;
    if (!scene) {
      return [composer.emptyState.default({
        icon: "▭", title: "No scene", hint: "Add one on the timeline." })];
    }
    const at = project.scenes.indexOf(scene) + 1;
    return [
      composer.label.section({ text: `Scene ${at}` }),
      composer.settingsRow.default({
        label: "Length on the timeline",
        // The rule, said where it applies: this is a crop, and a regenerate
        // does not read it.
        hint: "Frames this clip plays for. Regenerating uses the project's length.",
        control: composer.number.md({
          value: scene.length ?? project.video.length ?? 1, min: 1, label: "Length",
          onChange: (v) => project.setScene(scene.id, "length", v),
        }),
      }),
      composer.settingsRow.default({
        label: "Rating",
        hint: "What you thought of it. Kept with the scene.",
        control: composer.buttonGroup.md({
          label: "Rating", value: scene.rating || undefined, items: RATINGS,
          onChange: (v) => project.setScene(scene.id, "rating", v),
        }),
      }),
      composer.settingsRow.default({
        label: "Result",
        hint: scene.result ? "This scene has been generated." : "Not generated yet.",
      }),
    ];
  }

  function projectRows() {
    const open = project.project;
    if (!open) return [composer.hint.default({ text: "No project is open." })];
    return [
      composer.label.section({ text: "Project" }),
      composer.settingsRow.default({
        label: "Name",
        control: composer.input.md({
          value: open.name, label: "Project name",
          onCommit: (v) => { if (onRename) onRename(v); },
        }),
      }),
      composer.settingsRow.default({
        label: "Scenes", hint: `${project.scenes.length} on the timeline`,
      }),
      composer.hint.default({
        text: "Size and length live in the Constructor: they are what every scene "
            + "is generated at.",
      }),
    ];
  }

  function draw() {
    tabs.setValue(tab);
    // Wrapped in a group, which is what draws the card AND is the container the
    // stacking rule measures: settings rows outside one keep their 180px control
    // column whatever the width, and a four-button group is then clipped.
    const [head, ...body] = tab === "scene" ? sceneRows() : projectRows();
    rows.set([head, composer.group.default({ rows: body })]);
  }

  draw();

  return {
    node: host.node,
    draw,
    /** Which tab is showing. Tests and the wheel both ask. */
    get tab() { return tab; },
    show(next) { tab = next; draw(); },
    destroy() { host.destroy(); },
  };
}
