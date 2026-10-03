// The app's regions, laid out as v4 had them: a menubar, then Assets | Preview | Properties side by
// side with the Timeline under all three. Each region offers a mount point; no feature is named here.
import { composer } from "../composer/composer.js";
import { offer } from "./mounts.js";

export function build(root) {
  const zone = (title, mount) => {
    const body = composer.region.stack({ gap: "sm", fill: true });
    offer(mount, body.node);
    return composer.panel.zone({ title, body });
  };
  const group = (mount) => {
    const bar = composer.toolbar.default({});
    offer(mount, bar.node);
    return bar;
  };

  const row = composer.region.stack({ gap: "none", children: [zone("Assets", "assets"), zone("Preview", "preview"), zone("Properties", "inspector")] });
  const main = composer.region.stack({ gap: "none", fill: true, children: [row, zone("Timeline", "timeline")] });
  row.node.classList.add("fp-row");
  main.node.classList.add("fp-main");

  const header = composer.toolbar.default({ items: [composer.brand.default({}), group("menubar")], trailing: [group("menubar.right")] });
  const frame = composer.frame.app({ header, main });
  root.append(frame.node);
  return frame;
}
