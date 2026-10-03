// Stand-ins for the parts of v4's screen whose features are not ported yet: same place, same look,
// inert. A real feature that mounts at one of these points replaces its row here (delete the row).
import { composer as c } from "../../composer/composer.js";

const btn = (label, tone = "ghost", extra = {}) => c.button.sm({ label, tone, disabled: true, ...extra });
const menu = (label) => c.button.menu({ label, tone: "ghost" });          // inert, but not dimmed: a menu is never greyed in v4
const dockTab = (label) => c.button.sm({ label, tone: "neutral", pressed: true, disabled: true });

const items = {
  "menubar.mode": () => [c.segmented.sm({ options: [{ value: "simple", label: "Simple" }, { value: "editor", label: "Editor" }], value: "editor" })],
  "menubar.menus": () => ["File", "Edit", "View", "Help"].map(menu),
  "menubar.right": () => [c.chip.neutral({ label: "saved" }), c.chip.good({ label: "ComfyUI live", dot: true })],

  preview: () => [
    c.viewer.media({ empty: "No render yet. Use Generate in the timeline header." }),
    c.toolbar.default({ items: [btn("⏹"), btn("▶"), c.text.sm({ text: "00:00:00:00" }), btn("📌 Save frame")] }),
  ],


  "timeline.actions": () => [btn("⧉ Render", "render"), btn("⚡ Auto Montage")],
  "timeline.status": () => [btn("⏱ Sampler", "neutral"), c.text.sm({ text: "0 clips" }),
    dockTab("Assets"), dockTab("Preview"), dockTab("Properties"), btn("◆ Composer", "neutral")],
  timeline: () => [
    c.toolbar.default({ items: [btn("⤓ Export"), btn("Save to media bin"), btn("⊟ Separate audio"), btn("Remove audio")],
      trailing: [c.text.sm({ text: "J/K/L · S split · I/O in/out · +/- zoom" })] }),
  ],
};

export default Object.entries(items).map(([mount, build]) => ({
  id: `placeholder:${mount}`, mount,
  setup: ({ host }) => build().forEach((handle) => host.append(handle.node)),
}));
