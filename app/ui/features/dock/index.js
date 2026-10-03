// The three dock buttons beside the timeline: each shows or hides its zone.
import { composer as c } from "../../composer/composer.js";

const ZONES = [["assets", "Assets"], ["preview", "Preview"], ["properties", "Properties"]];

export default {
  id: "dock",
  mount: "timeline.status",
  needs: ["dock"],
  setup({ host, app }) {
    const tabs = ZONES.map(([zone, label]) => {
      const tab = c.button.sm({ label, tone: "neutral", pressed: app.dock.shown(zone), onClick: () => app.dock.toggle(zone) });
      host.append(tab.node);
      return [zone, tab];
    });
    return app.dock.on(() => tabs.forEach(([zone, tab]) => tab.setPressed && tab.setPressed(app.dock.shown(zone))));
  },
};
