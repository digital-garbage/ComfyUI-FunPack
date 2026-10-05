// "⏱ Sampler": the pipeline's steps, sampler and scheduler (the inputs its preset marks as sampling) and Engine ▸ Sampling's
// settings, one click away in a thin window over the timeline. Same pipeline and values tree as those pages, so they cannot
// disagree. Absent when the pipeline marks no sampling input and no module has a sampling setting.
import { offer } from "../../shell/actions.js";
import { composer as c } from "../../composer/composer.js";
import { settingRow, whenSatisfied } from "../settings/field.js";
import { widgetControl } from "../settings/models.js";

export default {
  id: "sampler",
  mount: "timeline.status",
  needs: ["pipeline"],
  setup({ host, app }) {
    const ps = app.pipeline;
    let pop = null, specs = {};
    const sampling = () => ps.activeModules().filter((m) => (m.category || "") === "sampling" && Object.keys(m.settings || {}).length && ps.useful(m));
    const marked = () => (ps.slots() || []).flatMap((slot) => (slot.roles || []).filter((r) => r.at === "generation.sampling" && r.input && !Array.isArray((slot.inputs || {})[r.input])).map((role) => ({ slot, role })));
    const describe = async () => {
      const missing = [...new Set(marked().map(({ slot }) => slot.node))].filter((n) => !(n in specs));
      if (missing.length) try { Object.assign(specs, (await app.api.describeNodes(missing)).nodes || {}); } catch { /* the rows just stay out until the next change */ }
    };
    const inputRow = ({ slot, role }) => {
      const w = ((specs[slot.node] || {}).widgets || []).find((x) => x.name === role.input);
      if (!w) return null;
      const set = async (v) => { if (JSON.stringify((slot.inputs || {})[role.input]) !== JSON.stringify(v)) await ps.save({ inputs: { [slot.id]: { [role.input]: v } } }); };
      return c.settingsRow.default({ label: role.label || w.name, hint: w.tooltip || "", control: widgetControl({ ...w, name: role.label || w.name }, (slot.inputs || {})[role.input] ?? w.default, set) });
    };
    const button = c.button.sm({ label: "⏱ Sampler", tone: "neutral", title: "Steps, sampler, scheduler and sampling settings: one click, no modal.", onClick: () => (pop ? pop.close() : open()) });
    button.node.hidden = true;
    host.append(button.node);

    const body = c.region.stack({ gap: "md" });
    const draw = () => {
      if (ps.loading()) return body.set([c.hint.default({ text: "Loading…" })]);
      if (ps.loadError()) return body.set([c.banner.warn({ text: `Could not load: ${ps.loadError()}` })]);
      const mods = sampling(), values = ps.currentValues();
      const change = async (id, name, value) => { await ps.setModuleValue(id, name, value); draw(); };
      body.set([...marked().map(inputRow), ...mods.flatMap((m) => [mods.length > 1 ? c.label.section({ text: m.title || m.id }) : null,
        ...Object.entries(m.settings).filter(([, spec]) => whenSatisfied(values, m.id, spec.when)).map(([name, spec]) => settingRow(m.id, name, spec, values, change))])].filter(Boolean));
    };
    const typing = () => body.node.contains(document.activeElement) && /^(input|textarea)$/i.test(document.activeElement.tagName);
    const off = ps.subscribe(() => { if (pop && !typing()) describe().then(draw); });
    const show = () => { button.node.hidden = !sampling().length && !marked().length; if (button.node.hidden && pop) pop.close(); };
    ps.ensureLoaded().then(show);
    const offShow = ps.subscribe(show);

    function open() {
      draw();
      pop = c.popover.anchored({ anchor: button, body, side: "top", align: "end", onClose: () => { pop = null; } });
      describe().then(() => { if (pop) draw(); });
    }
    const offered = offer(app, { id: "sampler", label: "Sampler", icon: "⏱", run: () => { if (!button.node.hidden) (pop ? pop.close() : open()); } });
    return () => { off(); offShow(); offered(); if (pop) pop.close(); button.node.remove(); };
  },
};
