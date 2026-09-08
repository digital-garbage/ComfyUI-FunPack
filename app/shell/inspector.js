// The inspector: what the PROJECT is, and what THIS SCENE is.
//
// Two tabs because they are two different things and v4 proved that mixing them
// is how a per-scene value quietly becomes the project's. The rule this file
// exists to make visible: a regenerate reads the PROJECT, and a crop is the
// scene's own -- so Length appears in both places and means something different
// in each, and both say so.

import { composer } from "../composer/composer.js";

// Two, because two is all anything downstream reads: what learns from these
// takes the sign of a rating and never its size. A finer scale would be the UI
// claiming a precision nothing uses.
const RATINGS = [
  { value: "liked", label: "Liked" },
  { value: "disliked", label: "Disliked" },
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
  // A STABLE host for whatever the pipeline declares at "project.negative" --
  // stable because offer() is wired to it once, at startup, while `rows` is
  // torn down and rebuilt by every draw() below. A field appended into a node
  // that gets replaced on the next scene selection would not survive it.
  const negative = composer.region.stack({ gap: "sm", label: "Negative prompt" });
  host.set([tabs, rows, negative]);
  // Every `.cx-stack` (composer/elements/layout.css) allows itself to shrink
  // below its own content's height -- `host` is also a FILL stack (flex:1 1
  // auto), so it competes for room with its OWN siblings in layout.js's outer
  // Properties column (the Generation section, the Settings collapsible) the
  // same way `rows`/`negative` compete with each other inside it. Without
  // flex-shrink:0 at every one of these levels, once the Project tab grew
  // past a couple of settings rows (Prompt craft, Variables), the column
  // squeezed whichever stack ran out of room below its own content instead
  // of growing past it -- and the spilled-over content landed on top of
  // whatever sits next in the flow, up to and including swallowing that
  // sibling's own clicks. flex-grow stays (fill:true still means "take any
  // LEFTOVER room" when this tab's content is short); only shrinking below
  // content is refused.
  host.node.style.flexShrink = "0";
  rows.node.style.flexShrink = "0";
  negative.node.style.flexShrink = "0";

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
        hint: "Liked or not. Press it again to take it back.",
        control: composer.buttonGroup.md({
          label: "Rating", value: scene.rating || undefined, items: RATINGS,
          // Pressing the one that is on clears it. Without a way back, a
          // mis-click is a permanent opinion -- and "no opinion" is a real
          // answer that nothing downstream should be given a guess about.
          onChange: (v) => {
            const next = v === scene.rating ? null : v;
            project.setScene(scene.id, "rating", next);
            draw();
          },
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
    const vars = project.variables;
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
      composer.label.section({ text: "Prompt craft" }),
      composer.field.default({
        label: "Anchor",
        hint: "Prepended to every scene's prompt at generation.",
        control: composer.textarea.md({
          value: project.anchor, rows: 2, placeholder: "cinematic, shared subject/setting",
          onCommit: (v) => project.setAnchor(v),
        }),
      }),
      composer.field.default({
        label: "Postfix",
        hint: "Appended to every scene's prompt, while the toggle below is on.",
        control: composer.textarea.md({
          value: project.postfix, rows: 2, placeholder: "4k, high detail",
          onCommit: (v) => project.setPostfix(v),
        }),
      }),
      composer.toggle.default({
        label: "Postfix enabled", checked: project.postfixEnabled,
        onChange: (v) => project.setPostfixEnabled(v),
      }),
      composer.label.section({ text: "Variables" }),
      composer.hint.default({ text: "$name in the anchor, a scene, or the postfix is replaced with its value." }),
      // `vars` (project.variables) is mutated IN PLACE and handed back to the
      // same setVariables call, rather than rebuilt via .map() from this
      // closure's own index -- setVariables's changed() triggers boot.js's
      // `inspector.draw()`, which tears down and rebuilds every field here
      // (including the one NOT being edited) before a second, still-pending
      // commit on this same row can fire. A rebuilt-`next`-array version reads
      // its OWN stale `vars` snapshot at that point and would silently
      // overwrite whatever the first commit just wrote; mutating the shared
      // object means the second commit's own edit survives no matter which
      // node the redraw already detached.
      ...vars.map((v, i) => composer.settingsRow.default({
        label: `$${v.name || "…"}`,
        control: composer.toolbar.default({ label: `$${v.name}`, items: [
          composer.input.md({
            value: v.name, placeholder: "name",
            onCommit: (name) => { v.name = name.replace(/^\$+/, ""); project.setVariables(vars); },
          }),
          composer.input.md({
            value: v.value, placeholder: "value",
            onCommit: (value) => { v.value = value; project.setVariables(vars); },
          }),
          composer.button.sm({ label: "✕", tone: "danger",
            onClick: () => { vars.splice(i, 1); project.setVariables(vars); } }),
        ] }),
      })),
      composer.button.md({ label: "+ Add variable",
        onClick: () => project.setVariables([...vars, { name: "", value: "" }]) }),
    ];
  }

  function draw() {
    tabs.setValue(tab);
    // Whatever mounted at "project.negative" belongs to the project, not the
    // scene -- shown only on the tab that means "the project".
    negative.node.hidden = tab !== "project";
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
    /** Where layout.js offers "project.negative" -- a project-level role,
     *  not a per-run one, so it lives beside Name/Scenes, not the prompt. */
    negativeHost: negative,
  };
}
