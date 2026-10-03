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

/** A node that is new, or whose input was just unwired, has nothing SET on it while its form shows defaults:
 *  these are the values to send so that what runs is what is seen. */
export function shownValues(slot, spec, onlyInput) {
  const out = {};
  for (const w of (spec && spec.widgets) || []) {
    if ((onlyInput && w.name !== onlyInput) || (slot.inputs && slot.inputs[w.name] !== undefined)) continue;
    const v = w.default !== undefined ? w.default : w.type === "COMBO" ? (w.choices || [])[0] : w.type === "BOOLEAN" ? false : undefined;
    if (v !== undefined) out[w.name] = v;
  }
  return out;
}
