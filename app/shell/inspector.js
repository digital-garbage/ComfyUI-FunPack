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

/**
 * A "✕" that removes something the moment it is pressed, not the moment a
 * browser would normally call it clicked -- for a MOUSE press. Between a
 * text field's blur-commit and its own click, a browser blurs the field
 * FIRST -- and a commit here triggers a full, non-keyed redraw of this whole
 * list (draw() -> node.replaceChildren()). Wired only to `onClick`, pressing
 * ✕ right after typing in ANOTHER row's field destroys this exact button
 * mid-gesture: the field's blur/commit/redraw runs before the click the
 * mousedown started ever reaches it, so the press does nothing and says
 * nothing. `mousedown` fires first, and calling `onRemove` there -- after
 * `preventDefault()` stops the browser's own focus-move, and so the blur it
 * would otherwise cause -- removes the row before a redraw gets the chance
 * to pull the rug out from under this button itself.
 *
 * `onClick` stays wired too, and is what a KEYBOARD Enter/Space reaches --
 * no mousedown precedes those, only a synthesised `click`. A real mouse
 * click still fires both: `onRemove` must tolerate being called twice for
 * the one press (its own reference-lookup guard makes the second call, on
 * an already-removed row, a safe no-op -- see the callers below).
 */
function removeButton(onRemove) {
  const btn = composer.button.sm({ label: "✕", tone: "danger", onClick: onRemove });
  btn.node.addEventListener("mousedown", (e) => { e.preventDefault(); onRemove(); });
  return btn;
}

export function createInspector({ project, onRename, onPickSourceImage, onPickReference } = {}) {
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
      composer.label.section({ text: "Reference media" }),
      composer.settingsRow.default({
        label: "Resolution source",
        hint: scene.source_image
          ? "Sets this scene's aspect ratio. The project's own Width/Height set the actual resolution; its pixels are not used."
          : "Pick an image in the Media bin to set this scene's aspect ratio.",
        control: composer.toolbar.default({ label: "Resolution source", items: [
          composer.button.sm({ label: scene.source_image ? "Change" : "Pick…",
            onClick: () => onPickSourceImage && onPickSourceImage(scene.id) }),
          ...(scene.source_image ? [removeButton(
            () => project.setSourceImage(scene.id, null))] : []),
        ] }),
      }),
      composer.label.section({ text: "References" }),
      ...scene.references.map((mediaId, i) => composer.settingsRow.default({
        label: `Reference ${i + 1}`,
        control: composer.toolbar.default({ label: `Reference ${i + 1}`, items: [
          composer.hint.default({ text: mediaId }),
          removeButton(() => project.setReferences(
            scene.id, scene.references.filter((_, j) => j !== i))),
        ] }),
      })),
      composer.button.md({ label: "+ Add reference",
        onClick: () => onPickReference && onPickReference(scene.id) }),
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
      // Every write below reads `project.variables` FRESH at commit time
      // (never the `vars` snapshot this render closed over) and hands
      // setVariables a brand new array -- necessary for two reasons:
      //
      // 1. setVariables's own remember() clones `project` for undo BEFORE
      //    applying the new list, so that clone must still show the OLD
      //    values. An earlier version mutated the shared row object in
      //    place and passed the SAME array back -- by the time remember()
      //    cloned it, the "before" snapshot already held the new value, and
      //    Undo after any variable edit silently did nothing.
      // 2. setVariables's changed() triggers boot.js's `inspector.draw()`,
      //    rebuilding every field here (including ones not being edited)
      //    before a second, still-pending commit on the same row can fire.
      //    Reading `project.variables` fresh (rather than the `vars` this
      //    closure captured at render time) means that second commit builds
      //    its own edit on top of whatever the first commit already saved.
      //
      // Rows are addressed by the ROW OBJECT ITSELF (found by reference in
      // the current array), never by the index this render closed over.
      // Removing row0 shrinks the array, so a commit still in flight for a
      // LATER row would write to the wrong slot under its old index -- and
      // that is not just two clicks in quick succession: removeButton's own
      // mousedown handler triggers a synchronous redraw that detaches
      // whatever field still has an uncommitted edit, and a browser fires
      // `blur` on a detached-while-focused element even with the mousedown's
      // default prevented. That blur commits REENTRANTLY, inside the very
      // setVariables() call that is still removing a row, indexed against an
      // array that has already changed shape underneath it -- including when
      // the row being removed IS the one with the uncommitted edit, where the
      // reentrant commit would otherwise silently overwrite whatever row now
      // sits at that same numeric slot. Looking the row up by reference makes
      // a commit for a row no longer in the array a no-op instead.
      ...vars.map((row) => {
        const commit = (patch) => {
          const current = project.variables;
          const at = current.indexOf(row);
          if (at === -1) return;                 // this row is already gone
          project.setVariables(current.map((x, j) => (j === at ? { ...x, ...patch } : x)));
        };
        const remove = () => {
          const current = project.variables;
          if (!current.includes(row)) return;     // already removed (see removeButton)
          project.setVariables(current.filter((x) => x !== row));
        };
        return composer.settingsRow.default({
          label: `$${row.name || "…"}`,
          control: composer.toolbar.default({ label: `$${row.name}`, items: [
            composer.input.md({
              value: row.name, placeholder: "name",
              onCommit: (name) => commit({ name: name.replace(/^\$+/, "") }),
            }),
            composer.input.md({
              value: row.value, placeholder: "value",
              onCommit: (value) => commit({ value }),
            }),
            removeButton(remove),
          ] }),
        });
      }),
      composer.button.md({ label: "+ Add variable",
        onClick: () => project.setVariables([...project.variables, { name: "", value: "" }]) }),
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
