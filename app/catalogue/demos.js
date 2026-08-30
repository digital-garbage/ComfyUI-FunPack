// One demo per element: the props the catalogue renders it with, and the props
// the rule tests attack it with.
//
// It lives in one place because two things need it and they must not drift: the
// catalogue shows what an element looks like, and rule_content_is_data feeds
// every string in here a payload to prove it comes back as text. A registered
// element with no entry here fails the suite -- that is how coverage keeps up
// with the registry instead of lagging behind it.

// A demo is either a plain props object, or a function given `composer` -- the
// containers take other elements' handles, and those have to be built.
export const DEMOS = {
  "header.display": { text: "Cutting Room" },
  "header.xl": { text: "Scene 3 — rooftop, dusk" },
  "header.lg": { text: "Sampling" },
  "header.md": { text: "Second pass" },
  "header.sm": { text: "Guidance" },

  "text.lg": { text: "The chain sampler runs each scene in order, carrying the last frame forward." },
  "text.md": { text: "The chain sampler runs each scene in order, carrying the last frame forward." },
  "text.sm": { text: "The chain sampler runs each scene in order, carrying the last frame forward." },
  "text.xs": { text: "The chain sampler runs each scene in order, carrying the last frame forward." },

  "hint.default": { text: "Locks frame timing to the audio stream." },
  "label.field": { text: "Lock strength", for: "demo-field" },
  "label.section": { text: "Sampling" },
  "code.inline": { text: "bong_tangent" },
  "code.block": { text: "steps: 7\nshift: 3.0\nscheduler: bong_tangent", label: "Pass settings" },

  "button.xl": { label: "Generate", tone: "primary" },
  "button.lg": { label: "Save project" },
  "button.md": { label: "Cancel" },
  "button.sm": { label: "Reset" },

  "iconButton.md": { icon: "✕", label: "Close" },
  "iconButton.sm": { icon: "✎", label: "Rename" },
  "iconButton.micro": { icon: "✕", label: "Remove" },


  "input.md": { value: "rooftop, dusk", placeholder: "Scene prompt", label: "Prompt" },
  "input.sm": { value: "121", label: "Frames" },
  "search.md": { placeholder: "Search shortcuts", label: "Search" },
  "number.md": { value: 3, min: 1, max: 10, step: 0.5, precision: 1, unit: "x", label: "Shift" },
  "stepper.md": { value: 7, min: 1, max: 40, step: 1, label: "Steps" },
  "textarea.md": { value: "A slow push in on the rooftop as the light goes.", rows: 3, label: "Scene prompt" },
  "filterList.md": {
    value: "bong",
    items: [
      { id: "bong", label: "bong_tangent", hint: "validated" },
      { id: "beta", label: "beta57", hint: "" },
      { id: "karras", label: "karras", hint: "2.3 era" },
    ],
  },

  "checkbox.default": { checked: true, label: "Enabled" },
  "checkboxRow.default": { label: "Sync to audio clock", hint: "Locks frame timing to the audio stream.", checked: true },
  "checklist.default": {
    label: "Passes",
    values: ["high"],
    items: [
      { value: "high", label: "High noise" },
      { value: "low", label: "Low noise" },
      { value: "detail", label: "Detail tail" },
    ],
  },
  "color.swatch": { label: "Overlay colour", value: "#f3a93c" },
  "radioGroup.default": {
    label: "Alignment",
    value: "beat",
    options: [
      { value: "beat", label: "Beat grid", hint: "Follows the tempo" },
      { value: "onset", label: "Onset", hint: "Follows transients" },
      { value: "flat", label: "Flat" },
    ],
  },
  "select.md": {
    label: "Scheduler", value: "bong_tangent",
    options: [
      { value: "bong_tangent", label: "bong_tangent" },
      { value: "beta57", label: "beta57" },
      { value: "karras", label: "karras" },
    ],
  },
  "select.sm": {
    label: "Sort", value: "recent",
    options: [{ value: "recent", label: "Most recent" }, { value: "name", label: "Name" }],
  },
  "segmented.md": {
    label: "Mode", value: "i2v",
    options: [{ value: "t2v", label: "Text" }, { value: "i2v", label: "Image" }, { value: "v2v", label: "Video" }],
  },
  "segmented.sm": {
    label: "Columns", value: "2",
    options: [{ value: "0", label: "Auto" }, { value: "2", label: "2" }, { value: "3", label: "3" }],
  },
  "toggle.default": { label: "Second pass", hint: "Runs both schedules in full.", checked: true },

  "slider.md": { value: 0.65, min: 0, max: 1, step: 0.05, label: "Lock strength" },
  "slider.sm": { value: 0.3, min: 0, max: 1, step: 0.05, label: "Blend" },
  "slider.readout": { value: 0.65, min: 0, max: 1, step: 0.05, unit: "x", label: "Lock strength" },
  "slider.macro": {
    value: 0.4, min: 0, max: 1, step: 0.05, label: "Variability",
    presets: [{ label: "Chill", value: 0.2 }, { label: "Spicy", value: 0.5 }, { label: "Chaos", value: 0.9 }],
  },
  "range.md": { from: 0.3, to: 0.7, min: 0, max: 1, step: 0.05, label: "Sigma window" },

  "buttonGroup.md": {
    label: "Alignment",
    value: "beat",
    items: [
      { value: "beat", label: "Beat grid" },
      { value: "onset", label: "Onset" },
      { value: "flat", label: "Flat" },
    ],
  },

  // --- layout: these compose other elements, so their demos are functions ---
  "field.default": (c) => ({
    label: "Lock strength", hint: "Locks frame timing to the audio stream.",
    control: c.slider.readout({ value: 0.65, min: 0, max: 1, step: 0.05, unit: "x" }),
  }),
  "field.row": (c) => ({
    fields: [
      c.field.default({ label: "Width", control: c.number.md({ value: 1216, step: 32 }) }),
      c.field.default({ label: "Height", control: c.number.md({ value: 704, step: 32 }) }),
    ],
  }),
  "settingsRow.default": (c) => ({
    label: "Second pass", hint: "Runs both schedules in full.",
    control: c.toggle.default({ checked: true }),
  }),
  "group.default": (c) => ({
    label: "Sampling",
    rows: [
      c.settingsRow.default({ label: "Steps", control: c.stepper.md({ value: 7, min: 1, max: 40 }) }),
      c.settingsRow.default({ label: "Shift", hint: "Structure versus fine detail.",
        control: c.number.md({ value: 3, min: 1, max: 10, step: 0.5, precision: 1 }) }),
      c.settingsRow.default({ label: "Reset engine", danger: true,
        control: c.button.md({ label: "Reset", tone: "danger" }) }),
    ],
  }),
  "panel.default": (c) => ({
    title: "Scene 3",
    actions: [c.iconButton.sm({ icon: "✎", label: "Rename" }), c.iconButton.sm({ icon: "✕", label: "Remove" })],
    body: c.text.sm({ text: "A slow push in on the rooftop as the light goes." }),
  }),
  "toolbar.default": (c) => ({
    label: "Timeline",
    items: [
      c.button.sm({ label: "Split" }), c.button.sm({ label: "Duplicate" }),
      c.button.sm({ label: "Delete", tone: "danger" }),
    ],
  }),
  "actionBar.sticky": (c) => ({
    note: "Nothing is saved until you press Save.",
    actions: [c.button.lg({ label: "Cancel" }), c.button.lg({ label: "Save", tone: "primary" })],
  }),
  "list.rows": (c) => ({
    reorder: true,
    addLabel: "Add LoRA",
    onAdd: () => {},
    onRemove: () => {},
    items: [
      { label: "detail_v3.safetensors",
        control: c.number.md({ value: 0.8, min: -2, max: 2, step: 0.05, precision: 2, label: "Weight" }) },
      { label: "motion_lift.safetensors",
        control: c.number.md({ value: 0.4, min: -2, max: 2, step: 0.05, precision: 2, label: "Weight" }) },
    ],
  }),
  "tabs.underline": { value: "compose", tabs: [
    { value: "compose", label: "Compose" }, { value: "shortcuts", label: "Shortcuts" }, { value: "files", label: "Files" },
  ] },
  "tabs.dock": { value: "assets", tabs: [
    { value: "assets", label: "Assets" }, { value: "preview", label: "Preview" }, { value: "props", label: "Properties" },
  ] },
  "splitPane.h": (c) => ({
    size: 40, label: "Assets and preview",
    panes: [
      c.panel.default({ title: "Assets", body: c.hint.default({ text: "Media bin" }) }),
      c.panel.default({ title: "Preview", body: c.hint.default({ text: "Program monitor" }) }),
    ],
  }),
  "frame.app": (c) => ({
    main: c.panel.default({ title: "Workspace", body: c.hint.default({ text: "Everything that fills the window" }) }),
    footer: c.actionBar.sticky({ note: "Ready", actions: [c.button.lg({ label: "Generate", tone: "primary" })] }),
  }),
  "workspace.docked": (c) => ({
    id: "demo",
    leftLabel: "Assets", rightLabel: "Properties",
    left: c.panel.default({ title: "Assets", body: c.hint.default({ text: "Media bin" }) }),
    centre: c.panel.default({ title: "Preview", body: c.hint.default({ text: "Program monitor" }) }),
    right: c.panel.default({ title: "Properties", body: c.hint.default({ text: "Scene settings" }) }),
  }),
  "splitPane.v": (c) => ({
    size: 60, label: "Preview and timeline",
    panes: [
      c.panel.default({ title: "Preview", body: c.hint.default({ text: "Program monitor" }) }),
      c.panel.default({ title: "Timeline", body: c.hint.default({ text: "Scenes" }) }),
    ],
  }),
  "collapsible.default": (c) => ({
    label: "Advanced", hint: "rarely needed",
    body: c.checkboxRow.default({ label: "Bypass this node", hint: "Passes inputs straight through." }),
  }),
  "sidebar.rail": { value: "models", items: [
    { value: "general", icon: "◐", label: "General" },
    { value: "models", icon: "▦", label: "Models" },
    { value: "engine", icon: "⚙", label: "Engine" },
  ] },

  // --- status ---
  "chip.neutral": { label: "experimental" },
  "chip.good": { label: "validated", dot: true },
  "chip.warn": { label: "needs audio", dot: true },
  "chip.danger": { label: "missing model", dot: true },
  "chip.info": { label: "flow-match", dot: true },
  "dot.default": { tone: "good", label: "Ready" },
  "banner.info": { text: "This project was made with an older engine." },
  "banner.warn": { text: "No audio stream: the audio clock will not run." },
  "banner.danger": { text: "The diffusion model could not be loaded." },
  "inlineError.default": { text: "Steps must be between 1 and 40." },
  "progress.bar": { value: 62, max: 100, label: "Rendering" },
  "progress.indeterminate": { label: "Preparing" },
  "spinner.md": { label: "Working" },
  "spinner.sm": { label: "Working" },
  "emptyState.default": { icon: "▦", title: "No media yet", hint: "Drop files here, or import from the menu." },
  "dropzone.default": { label: "Drop video or images", hint: "mp4, png, jpg" },
  "skeleton.line": { count: 3, label: "Loading scenes" },
  "skeleton.block": { count: 2, label: "Loading panels" },
  "skeleton.grid": { count: 6, label: "Loading media" },
  "toast.info": { text: "Project saved.", duration: 0 },
  "toast.good": { text: "Render finished.", duration: 0 },
  "toast.warn": { text: "Audio track is silent.", duration: 0 },
  "toast.danger": { text: "Generation failed.", duration: 0 },

  // --- overlays: these mount themselves into the portal ---
  "modal.generic": (c) => ({
    title: "Export settings", subtitle: "Written into the picture's metadata.",
    body: c.checkboxRow.default({ label: "Include prompt", hint: "Anyone with the file can read it." }),
    actions: [c.button.lg({ label: "Cancel" }), c.button.lg({ label: "Export", tone: "primary" })],
  }),
  "modal.stacked": (c) => ({
    title: "Pick a model", size: "sm",
    body: c.filterList.md({ items: [{ id: "a", label: "ltx-2.5-distilled" }, { id: "b", label: "minimax-h3" }] }),
  }),
  "modal.dialogue": { title: "Delete scene 3?", message: "This removes the scene and its render. It cannot be undone.",
                      tone: "danger", confirmLabel: "Delete", cancelLabel: "Keep" },
  "modal.prompt": { title: "New project", label: "Name", value: "Untitled", confirmLabel: "Create" },
  "modal.choice": { title: "Start from", subtitle: "You can change this later.", items: [
    { id: "blank", label: "Empty project", hint: "Nothing on the timeline" },
    { id: "image", label: "From an image", hint: "Adds it as the first anchor" },
  ] },

  "popover.anchored": (c) => ({
    anchor: c.button.md({ label: "Anchor" }),
    body: c.hint.default({ text: "Anchored, flipped and clamped by the kit." }),
  }),
  "menu.dropdown": (c) => ({
    anchor: c.button.md({ label: "Scene ▾" }),
    items: [
      { id: "split", label: "Split at playhead", hint: "S" },
      { id: "dup", label: "Duplicate" },
      { separator: true },
      { id: "del", label: "Remove", danger: true },
    ],
  }),
  "menu.context": { x: 200, y: 200, items: [
    { id: "rename", label: "Rename" }, { id: "export", label: "Export" },
  ] },
  "tooltip.default": (c) => ({ anchor: c.iconButton.md({ icon: "✎", label: "Rename" }),
                               text: "Rename this scene", trigger: false }),
  "autocomplete.default": (c) => ({
    input: c.input.md({ value: "", placeholder: "Type a shortcut" }),
    source: () => [{ label: "rooftop", hint: "location" }, { label: "rain", hint: "weather" }],
  }),
  "splitButton.md": { label: "Generate", items: [
    { id: "all", label: "All scenes" }, { id: "one", label: "This scene only" },
  ] },

  "floating.window": (c) => ({
    id: "demo-window", title: "Composer", subtitle: "prompt",
    width: 320, height: 200, x: 40, y: 40,
    body: c.textarea.md({ value: "A slow push in on the rooftop.", rows: 3, label: "Prompt" }),
  }),
  "overlay.blocking": { message: "Restarting ComfyUI…" },
  "slideOver.md": (c) => ({ side: "right", title: "Properties",
    body: c.hint.default({ text: "Simple mode slides this over the preview." }) }),

  // --- gallery + wheel ---
  "gallery.adaptive": { id: "demo-gallery", cols: 3, items: [
    { id: "1", label: "rooftop_dusk_01.mp4", badge: "video", duration: "0:04" },
    { id: "2", label: "anchor_frame.png", badge: "image" },
    { id: "3", label: "rooftop_dusk_02.mp4", badge: "video", duration: "0:05" },
    { id: "4", label: "reference_face.png", badge: "ref" },
    { id: "5", label: "score.wav", badge: "audio", duration: "1:12" },
    { id: "6", label: "rooftop_dusk_03.mp4", badge: "video", duration: "0:04" },
  ] },
  "gallery.cards": { value: "image", items: [
    { id: "blank", icon: "▦", label: "Empty project", hint: "Nothing on the timeline" },
    { id: "image", icon: "◐", label: "From an image", hint: "Adds it as the first anchor" },
    { id: "import", icon: "🎞", label: "Import a workflow", hint: "Reads an existing graph" },
  ] },
  "gallery.strip": { label: "Segments", items: [
    { id: "1", icon: "▦", label: "Scene 1" }, { id: "2", icon: "▦", label: "Scene 2" },
    { id: "3", icon: "▦", label: "Scene 3" }, { id: "4", icon: "▦", label: "Scene 4" },
  ] },
  "wheel.half": { edge: "right", items: [
    { icon: "✂", label: "Split" }, { icon: "⧉", label: "Duplicate" },
    { icon: "★", label: "Rate" }, { icon: "◐", label: "Anchor" },
  ] },
  "wheel.picker": { items: [
    { icon: "✂", label: "Split" }, { icon: "⧉", label: "Duplicate" },
    { icon: "★", label: "Rate" }, { icon: "⇄", label: "Reverse" },
    { icon: "◐", label: "Anchor" }, { icon: "✕", label: "Remove" },
  ] },
};

/** Extra variants worth seeing in the catalogue but not worth a second demo. */
export const VARIANTS = {
  "button.xl": [
    { label: "Generate", tone: "primary" },
    { label: "Neutral" },
    { label: "Delete", tone: "danger" },
    { label: "Ghost", tone: "ghost" },
    { label: "Disabled", disabled: true },
    { label: "Working", tone: "primary", busy: true },
  ],
  "button.md": [
    { label: "Cancel" },
    { label: "Primary", tone: "primary" },
    { label: "Danger", tone: "danger" },
    { label: "Ghost", tone: "ghost" },
  ],
};
