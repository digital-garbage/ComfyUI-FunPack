// The app's regions, laid out as v4 had them: a menubar, then Assets | Preview | Properties side by
// side with the Timeline under all three. Every region offers mount points; no feature is named here.
//   menubar.mode | menubar.menus | menubar.right                      the bar
//   <zone> (body) | <zone>.actions (after the title) | <zone>.status (far end)   for assets, preview, inspector, timeline
import { composer } from "../composer/composer.js";
import { offer } from "./mounts.js";

export function build(root) {
  const group = (mount) => {
    const bar = composer.toolbar.default({});
    offer(mount, bar.node);
    return bar;
  };
  const zone = (title, mount) => {
    const body = composer.region.stack({ gap: "sm", fill: true });
    offer(mount, body.node);
    return composer.panel.zone({ title, body, actions: [group(`${mount}.actions`)], status: [group(`${mount}.status`)] });
  };

  const row = composer.region.stack({ gap: "none", children: [zone("Assets", "assets"), zone("Preview", "preview"), zone("Properties", "inspector")] });
  const main = composer.region.stack({ gap: "none", fill: true, children: [row, zone("Timeline", "timeline")] });
  row.node.classList.add("fp-row");
  main.node.classList.add("fp-main");

  const header = composer.toolbar.default({
    items: [composer.brand.default({}), group("menubar.mode"), group("menubar.menus")],
    trailing: [group("menubar.right")],
  });
  const frame = composer.frame.app({ header, main });
  root.append(frame.node);
  return frame;
}
