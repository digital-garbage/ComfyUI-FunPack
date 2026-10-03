// The Properties zone: what this scene is, and what the whole project makes.
import { composer as c } from "../../composer/composer.js";
import { effFrames, effFps } from "../../shell/scenes.js";

const SOURCES = [{ value: "carry", label: "Continue from previous" }, { value: "image", label: "Image" }, { value: "video", label: "Video" }];

const sceneRows = (p) => {
  const sc = p.selected, open = p.project;
  if (!sc) return [c.emptyState.default({ icon: "▭", title: "No scene", hint: "Add one on the timeline." })];
  return [
    c.label.section({ text: `Scene ${p.scenes.indexOf(sc) + 1}` }),
    c.textarea.md({ label: "Scene prompt", value: sc.text || "", rows: 6, autoGrow: true, onInput: (v) => p.setText(sc.id, v) }),
    c.settingsRow.default({ label: "Source", control: c.select.md({ label: "Source", options: SOURCES, value: (sc.source || {}).type || "carry",
      onChange: (v) => p.setScene(sc.id, "source", { ...(sc.source || {}), type: v }) }) }),
    c.settingsRow.default({ label: "Frames", hint: `${effFrames(sc, open)} frames at ${effFps(sc, open)} fps. Trim on the timeline to change.` }),
    c.toggle.default({ label: "Leave out of the cut", checked: Boolean(sc.excluded), onChange: (v) => p.setScene(sc.id, "excluded", v) }),
  ];
};

const projectRows = (p) => {
  const open = p.project;
  if (!open) return [c.emptyState.default({ icon: "▭", title: "No project", hint: "Open or create one." })];
  const num = (label, key, fallback, min) => c.settingsRow.default({ label, control: c.number.md({ label, min, value: open[key] || fallback,
    onChange: (v) => p.setField(key, v) }) });
  return [
    c.label.section({ text: "Project" }),
    num("Frames per scene", "num_frames_per_scene", 97, 9), num("Frame rate", "frame_rate", 25, 1),
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
      const next = `${tab}|${p.project && p.project.id}|${p.selectedId}`;
      if (!force && next === key) return;
      key = next;
      body.set((tab === "scene" ? sceneRows : projectRows)(p));
    }
    draw(true);
    return app.on((what) => draw(what === "open"));
  },
};
