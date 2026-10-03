// "⏱ Sampler": steps, scheduler and sampler one click away, in a thin window over the timeline. Same settings as Engine ▸ Sampling,
// same values tree, so the two cannot disagree. Absent when no installed module has a sampling setting.
import { offer } from "../../shell/actions.js";
import { composer as c } from "../../composer/composer.js";
import { settingRow, whenSatisfied } from "../settings/field.js";

export default {
  id: "sampler",
  mount: "timeline.status",
  needs: ["pipeline"],
  setup({ host, app }) {
    const ps = app.pipeline;
    let pop = null;
    const sampling = () => ps.activeModules().filter((m) => (m.category || "") === "sampling" && Object.keys(m.settings || {}).length);
    const button = c.button.sm({ label: "⏱ Sampler", tone: "neutral", title: "Steps, scheduler and sampler: one click, no modal.", onClick: () => (pop ? pop.close() : open()) });
    button.node.hidden = true;
    host.append(button.node);

    const body = c.region.stack({ gap: "md" });
    const draw = () => {
      if (ps.loading()) return body.set([c.hint.default({ text: "Loading…" })]);
      if (ps.loadError()) return body.set([c.banner.warn({ text: `Could not load: ${ps.loadError()}` })]);
      const mods = sampling(), values = ps.currentValues();
      const change = async (id, name, value) => { await ps.setModuleValue(id, name, value); draw(); };
      body.set(mods.flatMap((m) => [mods.length > 1 ? c.label.section({ text: m.title || m.id }) : null,
        ...Object.entries(m.settings).filter(([, spec]) => whenSatisfied(values, m.id, spec.when)).map(([name, spec]) => settingRow(m.id, name, spec, values, change))]).filter(Boolean));
    };
    const typing = () => body.node.contains(document.activeElement) && /^(input|textarea)$/i.test(document.activeElement.tagName);
    const off = ps.subscribe(() => { if (pop && !typing()) draw(); });
    const show = () => { button.node.hidden = !sampling().length; if (button.node.hidden && pop) pop.close(); };
    ps.ensureLoaded().then(show);
    const offShow = ps.subscribe(show);

    function open() {
      draw();
      pop = c.popover.anchored({ anchor: button, body, side: "top", align: "end", onClose: () => { pop = null; } });
    }
    const offered = offer(app, { id: "sampler", label: "Sampler", icon: "⏱", run: () => { if (!button.node.hidden) (pop ? pop.close() : open()); } });
    return () => { off(); offShow(); offered(); if (pop) pop.close(); button.node.remove(); };
  },
};
