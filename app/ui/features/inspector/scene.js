// The Scene tab: this clip's prompt, where it starts from, how long it runs.
import { composer as c } from "../../composer/composer.js";
import { effFrames, effFps, genUnitId, isSubclip } from "../../shell/scenes.js";

const SOURCES = [{ value: "image", label: "Image · i2v anchor" }, { value: "carry", label: "From generated frame" }, { value: "video", label: "Video clip" }];
const MODES = [{ value: "project", label: "Project default" }, { value: "timeline", label: "Timeline trim" }, { value: "custom", label: "Custom" }];
const inert = (label, extra = {}) => c.button.sm({ label, tone: "ghost", disabled: true, ...extra });

/** A number-or-mode pair: "Project default / Timeline trim / Custom", and the number only when Custom. */
const lengthField = (p, sc, label, modeKey, valueKey, shown) => {
  const mode = sc[modeKey] || "project";
  return c.field.default({ label, control: c.region.stack({ gap: "xs", children: [
    c.select.md({ label, options: MODES, value: mode, onChange: (v) => p.setScene(sc.id, modeKey, v) }),
    mode === "custom" ? c.number.md({ label: `${label} (custom)`, min: 1, max: 16384, precision: 0, value: sc[valueKey] ?? shown,
      onChange: (v) => p.setScene(sc.id, valueKey, v) }) : null,
  ] }) });
};

export function sceneRows(p, app) {
  const sc = p.selected, open = p.project;
  if (!sc) return [c.emptyState.default({ icon: "▭", title: "No scene", hint: "Add one on the timeline." })];
  const cuts = p.scenes.filter((s) => genUnitId(s) === genUnitId(sc)).length > 1;
  return [
    c.banner.info({ text: "Generating with FunPack Studio + Chain Sampler", action: { label: "Engine settings →", onClick() {} } }),
    ...(cuts ? [c.hint.default({ text: "This scene has editorial cuts — Generate regens the whole uncut scene." })] : []),
    ...(isSubclip(sc) ? [c.hint.default({ text: "This is a cut: its prompt and source belong to the first part." })] : [
      c.field.default({ label: "Prompt", control: c.textarea.md({ label: "Prompt", value: sc.text || "", rows: 4, onInput: (v) => p.setText(sc.id, v) }) }),
      c.field.default({ label: "Source", control: c.select.md({ label: "Source", options: SOURCES, value: (sc.source || {}).type || "carry",
        onChange: (v) => p.setScene(sc.id, "source", { ...(sc.source || {}), type: v }) }) }),
    ]),
    c.label.section({ text: "Reference media (v5 pipeline)" }),
    c.field.default({ label: "Resolution source", hint: "Sets this scene's aspect ratio for generation — the project's own Width/Height set the actual resolution; this image's pixels are not used.",
      control: c.toolbar.default({ items: [c.text.sm({ text: "◇  — choose resolution source —" })], trailing: [inert("Browse")] }) }),
    c.field.default({ label: "References", control: c.toolbar.default({ items: [c.text.sm({ text: "◇  + Add reference" })], trailing: [inert("Browse"), inert("✕")] }) }),
    c.field.row({ fields: [
      lengthField(p, sc, "Frames", "frames_mode", "frames", effFrames(sc, open)),
      lengthField(p, sc, "FPS", "fps_mode", "fps", effFps(sc, open)),
    ] }),
    c.button.sm({ label: "Generate this scene", tone: "primary", onClick: () => app.say("generate.selected") }),
    c.checkbox.default({ label: "Exclude from full generation", checked: Boolean(sc.excluded), onChange: (v) => p.setScene(sc.id, "excluded", v) }),
  ];
}
