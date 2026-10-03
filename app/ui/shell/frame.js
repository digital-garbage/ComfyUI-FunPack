// The app's regions, laid out as v4 had them: a menubar, then Assets | Preview | Properties side by
// side with the Timeline under all three. Every region offers mount points; no feature is named here.
//   menubar.mode | menubar.menus | menubar.right                      the bar
//   assets.projects | assets.media (tabs in the Assets zone)
//   <zone> (body) | <zone>.actions (after the title) | <zone>.status (far end)   for assets, preview, inspector, timeline
import { composer } from "../composer/composer.js";
import { offer } from "./mounts.js";

export function build(root) {
  const group = (mount) => {
    const bar = composer.toolbar.default({});
    offer(mount, bar.node);
    return bar;
  };
  const zone = (title, mount, bodyOf) => {
    const body = bodyOf ? bodyOf() : composer.region.stack({ gap: "sm", fill: true });
    if (!bodyOf) offer(mount, body.node);
    return composer.panel.zone({ title, body, actions: [group(`${mount}.actions`)], status: [group(`${mount}.status`)] });
  };

  // Assets holds two sheets under tabs: the project list and the media bin.
  const assets = () => {
    const sheets = ["projects", "media"].map((name) => {
      const sheet = composer.region.stack({ gap: "sm", fill: true });
      offer(`assets.${name}`, sheet.node);
      sheet.node.hidden = name !== "projects";
      return { name, sheet };
    });
    const tabs = composer.tabs.underline({ label: "Assets", value: "projects", tabs: [{ value: "projects", label: "Projects" }, { value: "media", label: "Media" }],
      onChange: (v) => sheets.forEach(({ name, sheet }) => { sheet.node.hidden = name !== v; }) });
    return composer.region.stack({ gap: "sm", fill: true, children: [tabs, ...sheets.map((x) => x.sheet)] });
  };
  const row = composer.region.stack({ gap: "none", children: [zone("Assets", "assets", assets), zone("Preview", "preview"), zone("Properties", "inspector")] });
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
