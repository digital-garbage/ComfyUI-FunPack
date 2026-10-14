// Modules: what could be installed, what is in use, and what is off (for this project, after a failure, or by choice).
import { composer as c } from "../../composer/composer.js";
import { RESTART_HINT } from "../addons/index.js";

const CATEGORIES = [["continuity", "Continuity"], ["guidance", "Guidance"], ["conditioning", "Conditioning"], ["sampling", "Sampling"], ["post", "Post"], ["system", "System"], ["", "Other"]];

export const modules = (app) => function mount() {
  const ps = app.pipeline;
  const page = c.region.stack({ gap: "md" });

  let addons = [];
  const loadAddons = () => app.api.addons().then((a) => { addons = a; draw(); }).catch(() => {});
  // The project remembers which add-ons it uses, so a new rental can offer them again on load.
  const remember = (id, on) => { const now = app.project.pref("addons", []) || [], next = on ? [...new Set([...now, id])] : now.filter((i) => i !== id); app.project.setPref("addons", next); };
  const flip = async (a, on) => {
    try { addons = await app.api.setAddon(a.id, on); remember(a.id, on); } catch (err) { c.toast.warn({ text: `Could not change ${a.title}: ${err.message}` }); }
    draw();
  };
  const addonRow = (a) => {
    const pending = a.state === "pending";
    const hint = [a.hint, pending ? RESTART_HINT : null, a.failed ? `Could not install: ${a.failed}\nSwitch it off and on to try again.` : null].filter(Boolean).join("\n");
    return c.toggle.default({ label: a.title, hint, checked: a.state === "installed" || pending, onChange: (on) => flip(a, on) });
  };

  function draw() {
    if (ps.loading()) return page.set([c.hint.default({ text: "Loading…" })]);
    if (ps.loadError()) return page.set([c.banner.warn({ text: `Could not load: ${ps.loadError()}` })]);
    const control = ps.control();
    const ok = ps.usefulness(), mine = Object.values(ps.modulesById()).filter((m) => control[m.id] && control[m.id].controllable && (control[m.id].quarantine || ok(m)));
    const offered = addons.filter((a) => a.state === "available" || a.state === "pending"), onAddons = addons.filter((a) => a.state === "installed"), offAddons = addons.filter((a) => a.state === "disabled");
    if (!mine.length && !addons.length) return page.set([c.emptyState.default({ icon: "☷", title: "Nothing to switch", hint: "No installed module can be switched off." })]);
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
      return c.toggle.default({ label: name(m), hint: ps.allOff() ? "Off: all enhancements are disabled" : ps.isOff(m.id) ? "Off for this project" : undefined, checked: !ps.allOff() && !ps.isOff(m.id), disabled: ps.allOff(),
        onChange: (on) => ps.setOff(m.id, !on).then(draw) });
    };
    const isOff = (m) => control[m.id].quarantine || ps.isOff(m.id);
    const notes = ps.saveNotes();
    const master = c.toggle.default({ label: "Disable all enhancements", hint: "Every module above sits out, and so do shot camera's rewrites and continuity. Your choices are kept — switch it back off to restore them. Values typed into a node in Models & Pipeline (negative erase, int8, SLA…) are not touched. Handy for an A/B test of whether a feature does anything.",
      checked: ps.allOff(), onChange: (on) => ps.setAllOff(on).then(draw) });
    const section = (label, rows) => rows.length ? [c.label.section({ text: label }), ...rows] : [];
    const installed = mine.filter((m) => !isOff(m));
    page.set([notes.length ? c.banner.warn({ text: notes.join(" ") }) : null,
      ...section("Available modules", offered.map(addonRow)),
      ...section("Installed modules", [...onAddons.map(addonRow), ...CATEGORIES.flatMap(([key]) => installed.filter((m) => (m.category || "") === key).map(row))]),
      ...section("Disabled modules", [...offAddons.map(addonRow), ...mine.filter(isOff).map(row)]),
      c.hint.default({ text: "A module you turn off here is off for this project only, and disappears from Engine settings. One that failed while generating is off for every project until its code is repaired or you turn it back on." }),
      master].filter(Boolean));
  }
  draw();
  loadAddons();
  ps.ensureLoaded().then(() => ps.refreshControl()).then(draw);
  const off = ps.subscribe(draw);
  return { node: page.node, destroy: () => { off(); page.node.remove(); } };
};
