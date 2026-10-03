// Models & Pipeline: every node in the live pipeline, grouped as the pipeline groups them, with its values editable.
// Changing the shape of the pipeline (swap, add, remove, wire) is not here yet.
import { composer as c } from "../../composer/composer.js";

const widgetControl = (w, current, set) => {
  const label = w.name;
  if (w.type === "COMBO") {
    const choices = [...(w.choices || [])];
    if (current != null && !choices.includes(current)) choices.unshift(current);       // a file gone from disk still shows, so saving something else cannot swap it out
    return choices.length ? c.select.md({ label, value: current ?? choices[0], options: choices.map((v) => ({ value: v, label: String(v) })), onChange: set })
      : c.hint.default({ text: "No choices available — nothing found in the models folder." });
  }
  if (w.type === "INT" || w.type === "FLOAT") {
    return c.number.md({ label, value: current ?? w.default ?? 0, min: w.min, max: w.max, step: w.step ?? (w.type === "INT" ? 1 : undefined),
      precision: w.type === "INT" ? 0 : undefined, onChange: set });
  }
  return c.input.md({ label, value: current ?? "", onCommit: set });
};

export const models = (app) => function mount() {
  const ps = app.pipeline;
  const page = c.region.stack({ gap: "md" });
  let specs = {}, note = "", off = null;
  const slotsNow = () => ps.slots() || [];

  async function draw() {
    const slots = slotsNow();
    const missing = [...new Set(slots.map((s) => s.node))].filter((n) => !(n in specs));
    if (missing.length) {
      try { Object.assign(specs, (await app.api.describeNodes(missing)).nodes || {}); } catch (err) { note = err.message; }
      missing.forEach((n) => { if (!(n in specs)) specs[n] = null; });
    }
    const set = (slot, name) => async (value) => { await ps.save({ inputs: { [slot.id]: { [name]: value } } }); note = (ps.refused() || []).join(" "); draw(); };
    const groups = [];
    slots.forEach((s) => { const g = s.group || "Other"; (groups.find((x) => x[0] === g) || (groups.push([g, []]), groups[groups.length - 1]))[1].push(s); });
    page.set([
      note ? c.banner.warn({ text: note }) : null,
      !slots.length ? c.emptyState.default({ icon: "⬡", title: "No pipeline loaded", hint: ps.loadError() || "Is ComfyUI reachable?" }) : null,
      ...groups.flatMap(([group, list]) => [c.label.section({ text: group }), ...list.flatMap((slot) => {
        const spec = specs[slot.node];
        if (!spec) return [c.hint.default({ text: `${slot.node} is not installed — this slot can't be edited or run.` })];
        const rows = (spec.widgets || []).filter((w) => !Array.isArray(slot.inputs && slot.inputs[w.name]))        // fed by another node: edited there
          .map((w) => w.type === "BOOLEAN" ? c.toggle.default({ label: w.name, hint: w.tooltip || "", checked: Boolean(slot.inputs && slot.inputs[w.name] !== undefined ? slot.inputs[w.name] : w.default), onChange: set(slot, w.name) }) : c.settingsRow.default({ label: w.name, hint: w.tooltip || "", control: widgetControl(w, slot.inputs && slot.inputs[w.name] !== undefined ? slot.inputs[w.name] : w.default, set(slot, w.name)) }));
        return [c.header.sm({ text: spec.title || slot.node }), ...rows];
      })]),
    ].filter(Boolean));
  }
  ps.ensureLoaded().then(draw);
  off = ps.subscribe(() => draw());
  const handle = { node: page.node, destroy: () => { off && off(); page.node.remove(); } };
  return handle;
};
