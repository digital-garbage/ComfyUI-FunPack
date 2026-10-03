// What one run sends: the scene's prompt (expanded with anchor/postfix/$variables), the project's video
// settings, the scene's picked media -- each placed by the ROLE a pipeline slot declares, never by slot id.
import { wireReferences } from "../../shell/reference_wiring.js";
import { DRIVEN } from "../../shell/scenes.js";

const rolesAt = (slots, at) => (slots || []).flatMap((s) => (s.roles || []).filter((r) => r.at === at).map((role) => ({ slot: s, role })));

/** `frames`: how long this unit is (its clips together); a fresh `seed` per call so a re-roll is a different result. */
export async function buildInputs({ project, scene, slots, expand, frames, seed = () => Math.floor(Math.random() * 2 ** 31) }) {
  const raw = {};
  const put = (slot, name, value) => { raw[slot.id] = { ...raw[slot.id], [name]: value }; };

  for (const { slot, role } of (slots || []).flatMap((s) => (s.roles || []).filter((r) => (r.at || "").startsWith("project.") && r.input).map((role) => ({ slot: s, role })))) {
    const v = role.at === "project.negative" ? project.negative : DRIVEN[role.drives] ? (role.drives === "frames" && frames ? frames : project[DRIVEN[role.drives]]) : (project.video || {})[role.input];
    if (v !== undefined && v !== null && v !== "") put(slot, role.input, v);
  }
  for (const { slot, role } of rolesAt(slots, "generation.seed")) put(slot, role.input, seed());
  const source = rolesAt(slots, "assets.source_image")[0];
  if (source) put(source.slot, "media_id", project.generation_mode === "t2v" ? "" : scene.source_image || "");    // "from a prompt" starts every shot without a picture
  const { overrides, unwired } = wireReferences(scene.references || [], slots, source ? [source.slot.id] : []);
  for (const [id, fields] of Object.entries(overrides)) raw[id] = { ...raw[id], ...fields };

  const prompt = rolesAt(slots, "generation.prompt")[0];
  if (prompt) {
    const text = scene.text || "";
    const body = await expand({ text, anchor: project.anchor, postfix: project.postfix, postfix_enabled: project.postfix_enabled, variables: project.variables })
      .catch(() => null);         // a failing prompt-craft feature must not block the run: send what was typed
    put(prompt.slot, prompt.role.input, body && typeof body.text === "string" ? body.text : text);
  }
  return { inputs: raw, unwired, noPrompt: !prompt };
}
