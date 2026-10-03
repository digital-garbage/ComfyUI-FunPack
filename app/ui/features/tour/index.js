// The guided tour: a walk along the screen, one callout per part, on whatever project is open (it changes nothing).
// Help ▸ Welcome tour / Restart tour start it, Exit tour ends it, Skip to FAQ jumps to the last callout.
import { composer as c } from "../../composer/composer.js";

const zone = (n) => () => document.querySelectorAll("section.cx-zone")[n];
const button = (text) => () => [...document.querySelectorAll("button")].find((b) => b.textContent.trim().startsWith(text));
const STEPS = [
  { at: () => document.querySelector(".cx-frame-main"), side: "bottom", title: "Everything starts here", text: "The menu bar above has File, Edit, View and Help. Settings (top right) holds the models, the engine and the updates." },
  { at: zone(0), side: "right", title: "Assets", text: "Your projects and the media bin. Drop images, clips and audio here; right-click a tile to use it as a reference, rename or delete it." },
  { at: zone(1), side: "left", title: "Preview", text: "The monitor plays the cut wherever the playhead is. Save frame puts the current picture in the media bin." },
  { at: zone(2), side: "left", title: "Properties", text: "Project settings (size, length, postfix) and the picked scene's prompt, source, references and length." },
  { at: zone(3), side: "top", title: "Timeline", text: "One clip per scene. Drag the edges to trim, press S to split, Space to play. Lanes below carry separated sound." },
  { at: button("◆ Composer"), side: "top", title: "Composer", text: "Write the whole story in one box, cut at the cut word; keep templates, $variables and shortcuts; chat with the prompt enhancer." },
  { at: button("▶ Generate"), side: "top", title: "Generate", text: "Makes every scene that has no render yet. “Generate this scene” in Properties makes just the picked one; rate a result so FunPack learns your taste." },
  { at: () => document.querySelector(".cx-frame-main"), side: "bottom", title: "FAQ", text: "Nothing happens? Check the ComfyUI chip in the status bar and the Log window. Wrong look? Settings ▸ Appearance. Lost a layout? View ▸ Reset Layout. Lost a project? File ▸ Open recent. That is the tour: Help ▸ Restart tour runs it again." },
];

export default {
  id: "tour",
  mount: "menubar.menus",
  needs: ["project"],
  setup({ app }) {
    let at = -1, pop = null;
    const stop = () => { if (pop) { const p = pop; pop = null; p.close(); } at = -1; };
    function show(i) {
      if (pop) { const p = pop; pop = null; p.close(); }
      let target;
      while (i >= 0 && i < STEPS.length && !((target = STEPS[i].at()) && target.getBoundingClientRect().width)) i += i >= at ? 1 : -1;      // a part that is hidden right now is skipped
      if (i < 0 || i >= STEPS.length) return stop();
      at = i;
      const s = STEPS[i], last = i === STEPS.length - 1;
      const body = c.region.stack({ gap: "sm", children: [c.label.section({ text: `${i + 1} / ${STEPS.length} · ${s.title}` }), c.text.sm({ text: s.text }),
        c.toolbar.default({ items: [c.button.sm({ label: "Back", tone: "ghost", disabled: i === 0, onClick: () => show(i - 1) }), c.button.sm({ label: last ? "Done" : "Next", tone: "primary", onClick: () => (last ? stop() : show(i + 1)) })],
          trailing: [c.button.sm({ label: "Exit tour", tone: "ghost", onClick: stop })] })] });
      pop = c.popover.anchored({ anchor: target, body, side: s.side, align: "center", gap: 10, closeOnOutside: false, onClose: () => { if (pop) { pop = null; at = -1; } } });
    }
    app.has.add("tour");
    const off = app.on((what) => {
      if (what === "tour.start") show(0);
      else if (what === "tour.faq") show(STEPS.length - 1);
      else if (what === "tour.stop") stop();
    });
    return () => { off(); app.has.delete("tour"); };
  },
};
