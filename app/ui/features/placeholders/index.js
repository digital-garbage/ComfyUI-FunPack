// Stand-ins for the parts of v4's screen whose features are not ported yet: same place, same look,
// inert. A real feature that mounts at one of these points replaces its row here (delete the row).
import { composer as c } from "../../composer/composer.js";

const btn = (label, tone = "ghost", extra = {}) => c.button.sm({ label, tone, disabled: true, ...extra });
const menu = (label) => c.button.menu({ label, tone: "ghost" });          // inert, but not dimmed: a menu is never greyed in v4

const items = {
  "menubar.right": () => [c.chip.neutral({ label: "saved" }), c.chip.good({ label: "ComfyUI live", dot: true })],

  preview: () => [
    c.progress.bar({ value: 0, label: "Playback position" }),
    c.toolbar.default({ items: [c.iconButton.sm({ icon: "⏹", label: "Stop", disabled: true }), c.iconButton.sm({ icon: "▶", label: "Play", disabled: true }),
      c.text.sm({ text: "00:00:00:00" }), c.select.sm({ label: "Save frame to", disabled: true, options: [{ value: "", label: "— save to Media bin —" }], value: "" }),
      btn("📌 Save frame")] }),
  ],

  "timeline.actions": () => [btn("⚡ Auto Montage")],
  "timeline.toolbar": () => [btn("⊟ Separate audio"), btn("Remove audio")],
};

export default Object.entries(items).map(([mount, build]) => ({
  id: `placeholder:${mount}`, mount,
  setup: ({ host }) => build().forEach((handle) => host.append(handle.node)),
}));
