// What one run sends: the scene's prompt (expanded with anchor/postfix/$variables), the project's video
// settings, the scene's picked media -- each placed by the ROLE a pipeline slot declares, never by slot id.
import { wireReferences } from "../../shell/reference_wiring.js";
import { DRIVEN } from "../../shell/scenes.js";

export const rolesAt = (slots, at) => (slots || []).flatMap((s) => (s.roles || []).filter((r) => r.at === at).map((role) => ({ slot: s, role })));

/** `frames`: how long this unit is (its clips together); a fresh `seed` per call so a re-roll is a different result (`seed: null` keeps the node's own). */
export async function buildInputs({ project, scene, slots, expand, frames, hooks = [], prefix = [], seed = () => Math.floor(Math.random() * 2 ** 31) }) {
  const raw = {};
  const put = (slot, name, value) => { raw[slot.id] = { ...raw[slot.id], [name]: value }; };

  for (const { slot, role } of (slots || []).flatMap((s) => (s.roles || []).filter((r) => (r.at || "").startsWith("project.") && r.input).map((role) => ({ slot: s, role })))) {
    const v = role.at === "project.negative" ? project.negative : DRIVEN[role.drives] ? (role.drives === "frames" && frames ? frames : project[DRIVEN[role.drives]]) : (project.video || {})[role.input];
    if (v !== undefined && v !== null && v !== "") put(slot, role.input, v);
  }
  if (seed) for (const { slot, role } of rolesAt(slots, "generation.seed")) put(slot, role.input, seed());      // null: the seed node's own number runs
  const source = rolesAt(slots, "assets.source_image")[0];
  if (source) put(source.slot, "media_id", project.generation_mode === "t2v" ? "" : scene.source_image || "");    // "from a prompt" starts every shot without a picture
  const { overrides, unwired } = wireReferences(scene.references || [], slots, source ? [source.slot.id] : []);
  for (const [id, fields] of Object.entries(overrides)) raw[id] = { ...raw[id], ...fields };

  const prompt = rolesAt(slots, "generation.prompt")[0];
  if (prompt) {
    const text = scene.text || "";
    const body = await expand({ text, anchor: [...prefix, project.anchor].filter((x) => x && String(x).trim()).join(" "), postfix: project.postfix, postfix_enabled: project.postfix_enabled, variables: project.variables })
      .catch(() => null);         // a failing prompt-craft feature must not block the run: send what was typed
    put(prompt.slot, prompt.role.input, body && typeof body.text === "string" ? body.text : text);
  }
  // Features that add inputs of their own (a Chat comment for the enhancer, say) say so through a hook: { inputs: {slot: {name: value}}, notes: [text] }.
  const notes = [];
  for (const hook of hooks) {
    let out;
    try { out = (await hook({ project, scene, slots })) || {}; } catch (err) { notes.push(`An add-on's input could not be added to this run: ${err.message}`); continue; }
    for (const [id, fields] of Object.entries(out.inputs || {})) raw[id] = { ...raw[id], ...fields };
    notes.push(...(out.notes || []));
  }
  // Not refused (a picture alone may be the point), but a shot with nothing to go on is said, not silently made.
  // Judged on what is SENT (a carried frame counts; a picture t2v blanks, or one this pipeline has no input for, does not).
  const picture = (source && (raw[source.slot.id] || {}).media_id) || Object.keys(overrides).length;
  if (prompt && !String((raw[prompt.slot.id] || {})[prompt.role.input] ?? "").trim() && !picture) notes.push("This shot has no text and no start picture, so the model makes whatever it likes.");
  return { inputs: raw, unwired, noPrompt: !prompt, notes };
}
