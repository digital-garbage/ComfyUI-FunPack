// The Properties zone: what this scene is, and what the whole project makes.
import { composer as c } from "../../composer/composer.js";
import { effFrames, effFps, isSubclip, DRIVEN } from "../../shell/scenes.js";

const SOURCES = [{ value: "carry", label: "Continue from previous" }, { value: "image", label: "Image" }, { value: "video", label: "Video" }];

const sceneRows = (p) => {
  const sc = p.selected, open = p.project;
  if (!sc) return [c.emptyState.default({ icon: "▭", title: "No scene", hint: "Add one on the timeline." })];
  return [
    c.label.section({ text: `Scene ${p.scenes.indexOf(sc) + 1}` }),
    ...(isSubclip(sc) ? [c.hint.default({ text: "This is a cut of a longer clip: its prompt and source belong to the first part." })] : [
    c.textarea.md({ label: "Scene prompt", value: sc.text || "", rows: 6, autoGrow: true, onInput: (v) => p.setText(sc.id, v) }),
    c.settingsRow.default({ label: "Source", control: c.select.md({ label: "Source", options: SOURCES, value: (sc.source || {}).type || "carry",
      onChange: (v) => p.setScene(sc.id, "source", { ...(sc.source || {}), type: v }) }) })]),
    c.settingsRow.default({ label: "Frames", hint: `${effFrames(sc, open)} frames at ${effFps(sc, open)} fps. Trim on the timeline to change.` }),
    c.toggle.default({ label: "Leave out of the cut", checked: Boolean(sc.excluded), onChange: (v) => p.setScene(sc.id, "excluded", v) }),
  ];
};

// What the pipeline lets the project decide (width, length, rate...): one whole-number control per slot input
// that declares the role. A role that `drives` length or rate edits the number the timeline draws.
const videoRows = (p, slots) => (slots || []).flatMap((slot) => (slot.roles || []).filter((r) => r.at === "project.video" && r.input).map((role) => {
  const field = DRIVEN[role.drives];
  const now = field ? p.project[field] : p.video[role.input] ?? (slot.inputs || {})[role.input];
  if (!Number.isInteger(now)) return null;                  // the project file keeps whole numbers only
  const label = role.label || role.input;
  return { key: field || role.input, row: c.settingsRow.default({ label, control: c.number.md({ label, min: 1, max: 16384, step: 1, precision: 0, value: now,
    onChange: (v) => (field ? p.setField(field, v) : p.setVideo(role.input, v)) }) }) };
}).filter(Boolean)).filter((r, i, all) => all.findIndex((x) => x.key === r.key) === i).map((r) => r.row);   // one row per thing edited

const projectRows = (p, slots) => {
  const open = p.project;
  if (!open) return [c.emptyState.default({ icon: "▭", title: "No project", hint: "Open or create one." })];
  const rows = videoRows(p, slots);
  const num = (label, key, fallback, min) => c.settingsRow.default({ label, control: c.number.md({ label, min, max: 16384, step: 1, precision: 0, value: open[key] || fallback,
    onChange: (v) => p.setField(key, v) }) });
  return [
    c.label.section({ text: "Project" }),
    ...(rows.length ? rows : [num("Frames per scene", "num_frames_per_scene", 97, 9), num("Frame rate", "frame_rate", 25, 1)]),
    c.label.section({ text: "Negative prompt" }),
    c.textarea.md({ label: "Negative prompt", value: p.negative, rows: 3, autoGrow: true, onInput: (v) => p.setNegative(v) }),
  ];
};

export default {
  id: "inspector",
  mount: "inspector",
  needs: ["project"],
  setup({ host, app }) {
    const p = app.project;
    let tab = "scene", key = "";
    const body = c.region.stack({ gap: "sm", label: "Properties" });
    const tabs = c.segmented.sm({ label: "Properties", options: [{ value: "project", label: "Project" }, { value: "scene", label: "Scene" }], value: tab,
      onChange: (v) => { tab = v; draw(true); } });
    host.append(tabs.node, body.node);
    function draw(force) {      // not on every keystroke: that would rebuild the box being typed in
      const sc = p.selected, open = p.project;
      const next = `${tab}|${open && open.id}|${p.selectedId}|${tab === "scene" && sc && open ? [effFrames(sc, open), effFps(sc, open), p.scenes.indexOf(sc)] : ""}`;       // a trim or reorder changes what the panel says
      if (!force && next === key) return;
      key = next;
      body.set((tab === "scene" ? sceneRows : projectRows)(p, app.pipeline && app.pipeline.slots()));
    }
    draw(true);
    const offSlots = app.pipeline && app.pipeline.subscribe ? app.pipeline.subscribe(() => draw(true)) : null;     // the pipeline arrives after the project
    const offApp = app.on((what) => draw(what === "open"));
    return () => { offApp(); if (offSlots) offSlots(); };
  },
};
