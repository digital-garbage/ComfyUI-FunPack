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
    let pop = null, specs = {}, unreadable = false;
    const latest = {};              // per input, the value whose save is still on its way: newer than the slot
    const sampling = () => { const ok = ps.usefulness(); return ps.activeModules().filter((m) => (m.category || "") === "sampling" && Object.keys(m.settings || {}).length && ok(m)); };
    const marked = () => (ps.slots() || []).flatMap((slot) => (slot.roles || []).filter((r) => r.at === "generation.sampling" && r.input && !Array.isArray((slot.inputs || {})[r.input])).map((role) => ({ slot, role })));
    const describe = async () => {
      const missing = [...new Set(marked().map(({ slot }) => slot.node))].filter((n) => !(n in specs));
      if (missing.length) try { Object.assign(specs, (await app.api.describeNodes(missing)).nodes || {}); unreadable = false; } catch { unreadable = true; }   // asked again on the next change
    };
    const inputRow = ({ slot, role }) => {
      const w = ((specs[slot.node] || {}).widgets || []).find((x) => x.name === role.input);
      if (!w) return null;
      const set = async (v) => {            // compared with the newest value: the one drawn may be several saves old
        const key = `${slot.id}.${role.input}`;
        const now = (ps.slots() || []).find((s) => s.id === slot.id) || slot;
        if (JSON.stringify(key in latest ? latest[key] : (now.inputs || {})[role.input]) === JSON.stringify(v)) return;
        latest[key] = v;
        await ps.save({ inputs: { [slot.id]: { [role.input]: v } } });
        if (ps.settled) await ps.settled();       // a save queued behind another returns before it lands
        if (latest[key] !== v) return;            // a newer edit of this input is the one that counts
        delete latest[key];                       // landed or not, the slot is the truth again
        const after = (ps.slots() || []).find((s) => s.id === slot.id) || {};
        if (JSON.stringify((after.inputs || {})[role.input]) !== JSON.stringify(v)) {      // not saved: show what will really run, and why
          c.toast.warn({ text: [...(ps.saveNotes ? ps.saveNotes() : [])].find((n) => /could not save/i.test(n)) || `${role.label || role.input} was not saved.` });
          draw();
        }
      };
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
      const rows = marked().map(inputRow);
      const unread = marked().length && !rows.some(Boolean) && (unreadable || marked().every(({ slot }) => slot.node in specs)) ? [c.hint.default({ text: "Steps, sampler and scheduler could not be read from the sampler node: change them in Settings ▸ Models & Pipeline." })] : [];
      body.set([...unread, ...rows, ...mods.flatMap((m) => [mods.length > 1 ? c.label.section({ text: m.title || m.id }) : null,
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
