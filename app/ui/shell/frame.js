// The app's regions, laid out as v4 had them: a menubar, then Assets | Preview | Properties side by
// side with the Timeline under all three. Every region offers mount points; no feature is named here.
//   menubar.mode | menubar.menus | menubar.right                      the bar
//   preview.simple (under the monitor, Simple mode only)
//   timeline.toolbar | timeline.toolbar.end (the row over the stage; the stage itself is `timeline`)
//   assets.projects | assets.media (stacked in the Assets zone)
//   <zone> (body) | <zone>.actions (after the title) | <zone>.status (far end)   for assets, preview, inspector, timeline
import { composer } from "../composer/composer.js";
import { offer } from "./mounts.js";
import { drag } from "../composer/internals/drag.js";

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
  // Preview: the monitor's stack, then (Simple mode only) the one-prompt bar under it.
  const preview = () => {
    const stack = composer.region.stack({ gap: "sm", fill: true });
    offer("preview", stack.node);
    const bar = composer.region.stack({ gap: "sm" });
    offer("preview.simple", bar.node);
    return composer.region.stack({ gap: "sm", fill: true, children: [stack, bar] });
  };
  // The timeline zone: a toolbar (what acts on the cut | zoom and hints), then the stage.
  const timeline = () => {
    const bar = composer.toolbar.default({ items: [group("timeline.toolbar")], trailing: [group("timeline.toolbar.end")] });
    const stage = composer.region.stack({ gap: "none", fill: true });
    offer("timeline", stage.node);
    return composer.region.stack({ gap: "sm", fill: true, children: [bar, stage] });
  };
  const row = composer.region.stack({ gap: "none", children: [zone("Assets", "assets", assets), zone("Preview", "preview", preview), zone("Properties", "inspector")] });
  const main = composer.region.stack({ gap: "none", fill: true, children: [row, zone("Timeline", "timeline", timeline)] });
  row.node.classList.add("fp-row");
  main.node.classList.add("fp-main");

  const header = composer.toolbar.default({
    items: [composer.brand.default({}), group("menubar.mode"), group("menubar.menus")],
    trailing: [group("menubar.right")],
  });
  const frame = composer.frame.app({ header, main });
  // In Simple mode Assets and Properties slide over the preview on request.
  frame.togglePanel = (which) => { const open = row.node.classList.contains(`fp-show-${which}`); row.node.classList.remove("fp-show-assets", "fp-show-props"); if (!open) row.node.classList.add(`fp-show-${which}`); };
  frame.closePanels = () => row.node.classList.remove("fp-show-assets", "fp-show-props");
  root.append(frame.node);
  // v4's draggable splitters: three 7px bars that set the zone sizes; sizes are remembered.
  const KEY = "funpack_layout", css = document.documentElement.style;
  const size = { "--media-w": [160, 600], "--properties-w": [200, 700], "--timeline-h": [200, 700] };
  const saved = (() => { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } })();
  const apply = () => { for (const k in size) saved[k] ? css.setProperty(k, saved[k] + "px") : css.removeProperty(k); };
  const keep = () => { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch { /* not remembered */ } };
  const splitter = (host, cls, prop, sign, axis) => {
    const bar = document.createElement("div");
    bar.className = `fp-split ${cls}`; bar.setAttribute("role", "separator");
    let from = 0;
    drag(bar, {
      onStart: () => { from = parseFloat(getComputedStyle(document.documentElement).getPropertyValue(prop)); },
      onMove: (m) => { saved[prop] = Math.round(Math.min(size[prop][1], Math.max(size[prop][0], from + sign * m[axis]))); apply(); },
      onEnd: keep,
    });
    host.append(bar);
  };
  splitter(row.node, "fp-split-left", "--media-w", 1, "dx");
  splitter(row.node, "fp-split-right", "--properties-w", -1, "dx");
  splitter(main.node, "fp-split-bottom", "--timeline-h", -1, "dy");
  apply();
  frame.resetLayout = () => { for (const k in size) delete saved[k]; apply(); keep(); };
  frame.setZoneTitle = (mount, text) => titles[mount] && titles[mount](text);
  return frame;
}
