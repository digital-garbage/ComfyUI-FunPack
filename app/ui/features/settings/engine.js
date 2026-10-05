// Engine: the settings every installed module has volunteered, by category.
import { composer as c } from "../../composer/composer.js";
import { settingRow, whenSatisfied } from "./field.js";

const CATEGORIES = [["continuity", "Continuity"], ["guidance", "Guidance"], ["conditioning", "Conditioning"], ["sampling", "Sampling"], ["post", "Post"], ["system", "System"], ["", "Other"]];

export const engine = (app) => function mount() {
  const ps = app.pipeline;
  const page = c.region.stack({ gap: "md" });
  let category = null;

  function draw() {
    if (ps.loading()) return page.set([c.hint.default({ text: "Loading…" })]);
    if (ps.loadError()) return page.set([c.banner.warn({ text: `Could not load Engine settings: ${ps.loadError()}` })]);
    if (ps.allOff()) return page.set([c.banner.info({ text: "All enhancements are disabled for this project, so there is nothing to set. Turn them back on in Settings ▸ Modules." })]);
    // A module whose settings only its own node reads does nothing in a pipeline without that node: hidden.
    const inPipeline = (m) => !(m.nodes || []).length || !ps.slots() || m.nodes.some((n) => ps.slots().some((s) => s.node === n));
    const withSettings = ps.activeModules().filter((m) => Object.keys(m.settings || {}).length && inPipeline(m));
    const cats = CATEGORIES.filter(([k]) => withSettings.some((m) => (m.category || "") === k));
    if (!cats.length) return page.set([c.emptyState.default({ icon: "⚡", title: "Nothing to set", hint: "No installed module exposes a setting for this pipeline." })]);
    if (!cats.some(([k]) => k === category)) category = cats[0][0];
    const values = ps.currentValues(), notes = ps.saveNotes();
    const change = async (id, name, value) => { await ps.setModuleValue(id, name, value); draw(); };
    page.set([
      c.segmented.sm({ label: "Category", value: category, options: cats.map(([value, label]) => ({ value, label })), onChange: (v) => { category = v; draw(); } }),
      notes.length ? c.banner.info({ text: notes.join(" ") }) : null,
      ...withSettings.filter((m) => (m.category || "") === category).flatMap((m) => [c.label.section({ text: m.title || m.id }),
        ...Object.entries(m.settings).filter(([, spec]) => whenSatisfied(values, m.id, spec.when)).map(([name, spec]) => settingRow(m.id, name, spec, values, change))]),
    ].filter(Boolean));
  }
  draw();
  ps.ensureLoaded().then(draw);
  const typing = () => page.node.contains(document.activeElement) && /^(input|textarea)$/i.test(document.activeElement.tagName);
  page.node.addEventListener("focusout", () => setTimeout(() => { if (!typing()) draw(); }));
  const off = ps.subscribe(() => { if (!typing()) draw(); });
  return { node: page.node, destroy: () => { off(); page.node.remove(); } };
};
