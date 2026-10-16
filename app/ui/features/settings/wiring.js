// What a pipeline slot can be fed by, and what to say it is called. Pure.
export const labelOf = (slot, specs, slots) => {
  const title = (specs[slot.node] && specs[slot.node].title) || slot.node || slot.id;
  const twin = slots.some((o) => o.id !== slot.id && ((specs[o.node] && specs[o.node].title) || o.node) === title);
  return twin ? `${title} (${slot.id})` : title;
};

/** Outputs of the other slots that could feed an input of `type`: [{ value: "slot\0index", label }]. */
export function sourcesFor(slot, type, slots, specs) {
  const out = [];
  for (const other of slots) {
    const spec = specs[other.node];
    if (other.id === slot.id || !spec || !spec.outputs) continue;
    spec.outputs.forEach((kind, i) => {
      if (kind === type || kind === "*" || type === "*") out.push({ value: `${other.id}\0${i}`, label: `${labelOf(other, specs, slots)} · ${(spec.output_names && spec.output_names[i]) || kind}` });
    });
  }
  return out;
}

// The dropdowns' own values as the form would fill them, so a new node's revealed fields follow its default choice.
const shownParents = (slot, spec) => Object.fromEntries(((spec && spec.widgets) || []).filter((w) => !(slot.inputs && slot.inputs[w.name] !== undefined))
  .map((w) => [w.name, w.default !== undefined ? w.default : w.type === "BOOLEAN" ? false : (w.choices || [])[0]]));

/** A node that is new, or whose input was just unwired, has nothing SET on it while its form shows defaults:
 *  these are the values to send so that what runs is what is seen. */
/** Whether a field is drawn: every input its `shows` names ([{input, when}]: a dynamic dropdown's choice, or a
 *  node's own rule) holds one of the listed values -- as set, else as the form shows it by default. */
export const showing = (w, slot, spec) => (w.shows || []).every(({ input, when }) => {
  const parent = ((spec && spec.widgets) || []).find((x) => x.name === input);
  const picked = slot.inputs && slot.inputs[input] !== undefined ? slot.inputs[input] : parent && (parent.default !== undefined ? parent.default : parent.type === "BOOLEAN" ? false : (parent.choices || [])[0]);
  return when.includes(picked);
});

export function shownValues(slot, spec, onlyInput) {
  const out = {};
  for (const w of ((spec && spec.widgets) || []).filter((x) => showing(x, { inputs: { ...shownParents(slot, spec), ...(slot.inputs || {}) } }, spec))) {
    if ((onlyInput && w.name !== onlyInput) || (slot.inputs && slot.inputs[w.name] !== undefined)) continue;
    const v = w.default !== undefined ? w.default : w.type === "COMBO" ? (w.choices || [])[0] : w.type === "BOOLEAN" ? false : undefined;
    if (v !== undefined) out[w.name] = v;
  }
  return out;
}
