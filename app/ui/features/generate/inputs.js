// What one run sends: the scene's prompt (expanded with anchor/postfix/$variables), the project's video
// settings, the scene's picked media -- each placed by the ROLE a pipeline slot declares, never by slot id.
import { sceneReferences, wireReferences } from "../../shell/reference_wiring.js";
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
  for (const slot of (slots || []).filter((s) => s.node === "FunPackShotCamera")) put(slot, "project", project.id || "");      // what the node keeps per project (the words focus infusion draws on), also for pipelines saved before it was asked for
  const source = rolesAt(slots, "assets.source_image")[0];
  if (source) put(source.slot, "media_id", project.generation_mode === "t2v" ? "" : scene.source_image || "");    // "from a prompt" starts every shot without a picture
  const like = source && source.role.blank_like;
  if (like && !raw[source.slot.id].media_id) {          // text-only: a plain picture of the size the pipeline asks for stands in
    const target = (slots || []).find((s) => s.id === like.slot) || { inputs: {} };
    const size = (name) => (raw[target.id] || {})[name] ?? (target.inputs || {})[name];
    put(source.slot, "blank_width", Number(size(like.width)) || 0);
    put(source.slot, "blank_height", Number(size(like.height)) || 0);
    const off = like.unsend && (slots || []).find((s) => s.id === like.unsend.slot);      // a stand-in is not a picture to show anyone
    if (off) put(off, like.unsend.input, false);
  }
  const plan = sceneReferences(scene, project.fixed_references);
  const { overrides, unwired } = wireReferences(plan.ids, slots, source ? [source.slot.id] : [], plan.pins);
  for (const [id, fields] of Object.entries(overrides)) raw[id] = { ...raw[id], ...fields };

  const prompt = rolesAt(slots, "generation.prompt")[0];
  let failedExpand = false;
  if (prompt) {
    const text = scene.text || "";
    const body = await expand({ text, anchor: [...prefix, project.anchor].filter((x) => x && String(x).trim()).join(" "), postfix: project.postfix, postfix_enabled: project.postfix_enabled, variables: project.variables })
      .catch(() => null);         // a failing prompt-craft feature must not block the run: send what was typed
    put(prompt.slot, prompt.role.input, body && typeof body.text === "string" ? body.text : text);
    if (!(body && typeof body.text === "string")) failedExpand = true;
  }
  // Features that add inputs of their own (a Chat comment for the enhancer, say) say so through a hook: { inputs: {slot: {name: value}}, notes: [text] }.
  const notes = failedExpand ? ["The prompt could not be expanded (shortcuts, $variables, anchor, postfix): the text was sent as typed."] : [];
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
