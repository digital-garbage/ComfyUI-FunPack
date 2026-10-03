// The Project tab: what the whole project makes.
import { composer as c } from "../../composer/composer.js";
import { DRIVEN } from "../../shell/scenes.js";

const SIZES = [{ value: "", label: "The first clip's own size" }, { value: "project", label: "The project's size" }];
const STARTS = [{ value: "image", label: "From an image" }, { value: "prompt", label: "From a prompt" }];

// What the pipeline lets the project decide (width, length, rate...): one whole-number control per slot input
// that declares the role. A role that `drives` length or rate edits the number the timeline draws.
const videoControls = (p, slots) => (slots || []).flatMap((slot) => (slot.roles || []).filter((r) => r.at === "project.video" && r.input).map((role) => {
  const field = DRIVEN[role.drives];
  const now = field ? p.project[field] : p.video[role.input] ?? (slot.inputs || {})[role.input];
  if (!Number.isInteger(now)) return null;                  // the project file keeps whole numbers only
  const label = role.label || role.input;
  return { key: field || role.input, label, field: c.field.default({ label, control: c.number.md({ label, min: 1, max: role.drives === "fps" ? 1000 : 16384, step: 1, precision: 0, value: now,
    onChange: (v) => (field ? p.setField(field, v) : p.setVideo(role.input, v)) }) }) };
}).filter(Boolean)).filter((r, i, all) => all.findIndex((x) => x.key === r.key) === i);

const num = (p, label, key, fallback) => ({ key, label, field: c.field.default({ label, control: c.number.md({ label, min: 1, max: key === "frame_rate" ? 1000 : 16384, step: 1, precision: 0,
  value: p.project[key] || fallback, onChange: (v) => p.setField(key, v) }) }) });

export function projectRows(p, slots) {
  const open = p.project;
  if (!open) return [c.emptyState.default({ icon: "▭", title: "No project", hint: "Open or create one." })];
  const controls = videoControls(p, slots);
  const pick = (...keys) => keys.map((k) => controls.find((x) => x.key === k)).filter(Boolean);
  const frames = pick("num_frames_per_scene")[0] || num(p, "Frames / scene", "num_frames_per_scene", 97);
  const fps = pick("frame_rate")[0] || num(p, "FPS", "frame_rate", 25);
  const size = controls.filter((x) => x !== frames && x !== fps);
  const per = (open.num_frames_per_scene || 97) / (open.frame_rate || 25);
  return [
    c.label.section({ text: "Project" }),
    c.field.default({ label: "Project name", control: c.input.md({ label: "Project name", value: open.name || "", onCommit: (v) => p.rename(v) }) }),
    c.label.section({ text: "Video settings" }),
    c.field.row({ fields: [frames.field, fps.field] }),
    c.hint.default({ text: `≈ ${per.toFixed(2)} s per shot` }),
    ...(size.length ? [c.field.row({ fields: size.map((x) => x.field) })] : []),
    c.field.default({ label: "Final render size", control: c.select.md({ label: "Final render size", options: SIZES, value: open.export_size_from || "",
      onChange: (v) => p.setField("export_size_from", v) }) }),
    c.field.default({ label: "Start shots", control: c.select.md({ label: "Start shots", options: STARTS, value: "image", onChange() {}, disabled: true }) }),
    c.label.section({ text: "Prompt" }),
    c.field.default({ label: "Anchor", control: c.textarea.md({ label: "Anchor", value: p.anchor, rows: 2, onInput: (v) => p.setAnchor(v) }) }),
    c.field.default({ label: "Negative prompt", control: c.textarea.md({ label: "Negative prompt", value: p.negative, rows: 2, onInput: (v) => p.setNegative(v) }) }),
    c.field.default({ label: "Postfix", control: c.textarea.md({ label: "Postfix", value: p.postfix, rows: 2, onInput: (v) => p.setPostfix(v) }) }),
    c.collapsible.default({ label: "ADVANCED PROJECT SETTINGS", body: c.collapsible.default({ label: "Split preview (generation prompt)",
      body: c.hint.default({ text: "ComfyUI offline — preview paused" }) }) }),
  ];
}
