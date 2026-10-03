// Modules: switch a module off for this project, and bring back the ones the server turned off after a failure.
import { composer as c } from "../../composer/composer.js";

const CATEGORIES = [["continuity", "Continuity"], ["guidance", "Guidance"], ["conditioning", "Conditioning"], ["sampling", "Sampling"], ["post", "Post"], ["system", "System"], ["", "Other"]];

export const modules = (app) => function mount() {
  const ps = app.pipeline;
  const page = c.region.stack({ gap: "md" });

  function draw() {
    if (ps.loading()) return page.set([c.hint.default({ text: "Loading…" })]);
    if (ps.loadError()) return page.set([c.banner.warn({ text: `Could not load: ${ps.loadError()}` })]);
    const control = ps.control();
    const mine = Object.values(ps.modulesById()).filter((m) => control[m.id] && control[m.id].controllable);
    if (!mine.length) return page.set([c.emptyState.default({ icon: "☷", title: "Nothing to switch", hint: "No installed module can be switched off." })]);
    const name = (m) => m.title || m.id;
    const row = (m) => {
      const held = control[m.id].quarantine;
      if (held) {
        return c.settingsRow.default({ label: `${name(m)} — failed`, hint: `Off since ${held.when || "a failure"}: ${held.reason || "an error"}`,
          control: c.button.sm({ label: "Turn back on", tone: "ghost", onClick: async () => {
            try { await app.api.releaseModule(m.id); } catch (err) { c.toast.warn({ text: `Could not turn ${name(m)} back on: ${err.message}` }); }
            await ps.refreshControl(); draw();
          } }) });
      }
      return c.toggle.default({ label: name(m), hint: ps.isOff(m.id) ? "Off for this project" : undefined, checked: !ps.isOff(m.id),
        onChange: (on) => ps.setOff(m.id, !on).then(draw) });
    };
    page.set([c.hint.default({ text: "A module you turn off here is off for this project only, and disappears from Engine settings. One that failed while generating is off for every project until its code is repaired or you turn it back on." }),
      ...CATEGORIES.flatMap(([key, label]) => { const list = mine.filter((m) => (m.category || "") === key); return list.length ? [c.label.section({ text: label }), ...list.map(row)] : []; })]);
  }
  draw();
  ps.ensureLoaded().then(() => ps.refreshControl()).then(draw);
  const off = ps.subscribe(draw);
  return { node: page.node, destroy: () => { off(); page.node.remove(); } };
};
