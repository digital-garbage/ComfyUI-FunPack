// The app's regions, laid out as v4 had them: a menubar, then Assets | Preview | Properties side by
// side with the Timeline under all three. Every region offers mount points; no feature is named here.
//   menubar.mode | menubar.menus | menubar.right                      the bar
//   timeline.toolbar | timeline.toolbar.end (the row over the stage; the stage itself is `timeline`)
//   assets.projects | assets.media (stacked in the Assets zone)
//   <zone> (body) | <zone>.actions (after the title) | <zone>.status (far end)   for assets, preview, inspector, timeline
import { composer } from "../composer/composer.js";
import { offer } from "./mounts.js";

export function build(root) {
  const group = (mount) => {
    const bar = composer.toolbar.default({});
    offer(mount, bar.node);
    return bar;
  };
  const titles = {};
  const zone = (title, mount, bodyOf) => {
    const body = bodyOf ? bodyOf() : composer.region.stack({ gap: "sm", fill: true });
    if (!bodyOf) offer(mount, body.node);
    const panel = composer.panel.zone({ title, body, actions: [group(`${mount}.actions`)], status: [group(`${mount}.status`)] });
    titles[mount] = panel.setTitle;
    return panel;
  };

  // Assets stacks two sections, as v4 did: the project list, then the media bin below it.
  const assets = () => {
    const sheets = ["projects", "media"].map((name) => {
      const sheet = composer.region.stack({ gap: "sm" });
      offer(`assets.${name}`, sheet.node);
      return sheet;
    });
    return composer.region.stack({ gap: "md", fill: true, children: sheets });
  };
  // The timeline zone: a toolbar (what acts on the cut | zoom and hints), then the stage.
  const timeline = () => {
    const bar = composer.toolbar.default({ items: [group("timeline.toolbar")], trailing: [group("timeline.toolbar.end")] });
    const stage = composer.region.stack({ gap: "none", fill: true });
    offer("timeline", stage.node);
    return composer.region.stack({ gap: "sm", fill: true, children: [bar, stage] });
  };
  const row = composer.region.stack({ gap: "none", children: [zone("Assets", "assets", assets), zone("Preview", "preview"), zone("Properties", "inspector")] });
  const main = composer.region.stack({ gap: "none", fill: true, children: [row, zone("Timeline", "timeline", timeline)] });
  row.node.classList.add("fp-row");
  main.node.classList.add("fp-main");

  const header = composer.toolbar.default({
    items: [composer.brand.default({}), group("menubar.mode"), group("menubar.menus")],
    trailing: [group("menubar.right")],
  });
  const frame = composer.frame.app({ header, main });
  root.append(frame.node);
  frame.setZoneTitle = (mount, text) => titles[mount] && titles[mount](text);
  return frame;
}
