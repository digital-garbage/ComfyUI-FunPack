// The frame counts the current model can make, read from its own length input (min + step): MiniMax H3's node says
// min 5 step 17 (22, 39, 56 …), LTX's min 1 step 8 (9, 17, 25 …). No family is named here: the node is the truth.
const specs = {};
const frameInputs = (slots) => (slots || []).flatMap((s) => (s.roles || []).filter((r) => r.drives === "frames" && r.input).map((r) => ({ node: s.node, input: r.input })));

/** {step, base} of the strictest frames input, or null when none steps by more than 1 (or its node is not described yet). */
export function gridOf(slots) {
  let best = null;
  for (const { node, input } of frameInputs(slots)) {
    const w = ((specs[node] || {}).widgets || []).find((x) => x.name === input);
    const step = w && Number(w.step), min = w ? Number(w.min) || 0 : 0;
    if (step > 1 && (!best || step > best.step)) best = { step, base: ((min % step) + step) % step, fps: null };
  }
  return best;
}

/** Asks for the length inputs' node descriptions not known yet. -> true when something new was learned. */
export async function learnGrid(api, slots) {
  const missing = [...new Set(frameInputs(slots).map((f) => f.node))].filter((n) => n && !(n in specs));
  if (!missing.length || !api || !api.describeNodes) return false;
  try { Object.assign(specs, (await api.describeNodes(missing)).nodes || {}); } catch { missing.forEach((n) => { specs[n] = null; }); }
  return true;
}
