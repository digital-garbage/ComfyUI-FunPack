// Continuity: a scene set to "From generated frame" starts from the last picture of the clip before it. This puts that picture into the
// run's start-picture input (a generate input hook); the choices live in Settings ▸ Engine ▸ Continuity.
import { previousClip } from "../../shell/continuity.js";
import { wireReferences } from "../../shell/reference_wiring.js";
import { names } from "../../shell/bin.js";

const startSlot = (slots) => (slots || []).find((s) => (s.roles || []).some((r) => r.at === "assets.source_image"));

export default {
  id: "continuity",
  mount: "menubar.menus",
  needs: ["project", "api", "pipeline", "inputHooks"],
  setup({ app }) {
    const pin = (p) => (p.continuity_settings || {}).identity_pin_ref || "";
    const setPin = (ref) => app.project.edit((pr) => { pr.continuity_settings = { ...(pr.continuity_settings || {}), identity_pin_ref: ref || null }; return true; });
    const menu = (it, item) => (item && item.kind === "image" && app.project.project ? [pin(app.project.project) === it.id
      ? { id: "unpin", label: "Unpin identity picture", run: () => setPin("") }
      : { id: "pin", label: "Use as the identity picture for every scene", run: () => setPin(it.id) }] : []);
    app.mediaMenu.push(menu);

    // The identity picture goes in after the scene's own references, through the pipeline's own reference inputs.
    const withPin = ({ project, scene, slots }) => {
      const ref = pin(project);
      if (!ref || (scene.references || []).includes(ref)) return {};
      if (names.size && !names.has(ref)) return { notes: ["The identity picture is no longer in the media bin, so it was not used."] };
      const reserved = startSlot(slots) ? [startSlot(slots).id] : [], own = scene.references || [];
      const withIt = wireReferences([...own, ref], slots, reserved), without = wireReferences(own, slots, reserved);
      return { inputs: withIt.overrides, notes: withIt.unwired > without.unwired ? ["The identity picture was not used: this pipeline has no free reference input for it."] : [] };
    };
    const carryHook = async ({ project, scene, slots }) => {
      // Read from the pipeline this run was frozen with, not the live one an edit may have changed since the click.
      if (app.pipeline.allOff?.(slots)) return {};        // Disable all enhancements means a plain run
      const mine = (app.pipeline.currentValues(slots) || {}).continuity;
      if (!mine) return {};       // the module that holds the choices is not here: this does nothing at all
      const carry = mine.carry !== false, guard = mine.dark_guard !== false;
      const slot = startSlot(slots);
      if (!slot || !carry || project.generation_mode === "t2v" || ((scene.source || {}).type || "carry") !== "carry" || scene.source_image) return {};
      const prev = previousClip(project, scene);
      if (!prev.sceneId) return prev.quiet ? {} : { notes: [`Starts without a picture: ${prev.why}.`] };
      let got;
      try { got = await app.api.lastFrame(project.id, { scene_id: prev.sceneId, render: prev.render, dur: prev.dur, src_in: prev.srcIn, reverse: prev.reverse }); }
      catch (err) { return { notes: [`Starts without a picture: the last frame of the clip before it could not be taken (${err.message}).`] }; }
      if (guard && got.dark) return { notes: ["Starts without a picture: the clip before it ends in a fade to black (turn off “Don't continue from a fade to black” in Engine settings to use it anyway)."] };
      return { inputs: { [slot.id]: { media_id: got.media_id } } };
    };
    const hook = async (ctx) => {
      const a = withPin(ctx), b = await carryHook(ctx);
      return { inputs: { ...a.inputs, ...b.inputs }, notes: [...(a.notes || []), ...(b.notes || [])] };
    };
    app.inputHooks.push(hook);
    return () => { for (const [list, x] of [[app.inputHooks, hook], [app.mediaMenu, menu]]) { const i = list.indexOf(x); if (i >= 0) list.splice(i, 1); } };
  },
};
