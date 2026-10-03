// The app's regions, and what each offers to features. This is the whole of the layout:
// Assets | Preview over Timeline | Properties, under a menubar. It names no feature.
import { composer } from "../composer/composer.js";
import { offer } from "./mounts.js";

export function build(root) {
  const zone = (title, mount) => {
    const body = composer.region.stack({ gap: "sm", fill: true });
    offer(mount, body.node);
    return composer.panel.zone({ title, body });
  };
  const menubar = composer.region.stack({ gap: "sm" });
  offer("menubar", menubar.node);

  const centre = composer.splitPane.v({ panes: [zone("Preview", "preview"), zone("Timeline", "timeline")], size: 60 });
  const main = composer.workspace.docked({ centre, left: zone("Assets", "assets"), right: zone("Properties", "inspector") });
  const frame = composer.frame.app({ header: menubar, main });
  root.append(frame.node);
  return frame;
}
