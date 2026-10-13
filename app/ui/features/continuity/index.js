// Continuity: a scene set to "From generated frame" starts from the last picture of the clip before it. This puts that picture into the
// run's start-picture input (a generate input hook); the choices live in Settings ▸ Engine ▸ Continuity.
import { previousClip } from "../../shell/continuity.js";
import { MAX_REFERENCES } from "../../shell/reference_wiring.js";

const KINDS = ["image", "video", "audio"];

const startSlot = (slots) => (slots || []).find((s) => (s.roles || []).some((r) => r.at === "assets.source_image"));

export default {
  id: "continuity",
  mount: "menubar.menus",
  needs: ["project", "api", "pipeline", "inputHooks"],
  setup({ app }) {
    // A reference pinned to a slot number goes into every scene at that number, until it is unpinned here.
    const setFixed = (id, n) => app.project.edit((pr) => {
      const next = { ...(pr.fixed_references || {}) };
      if (n) next[id] = n; else delete next[id];
      pr.fixed_references = next;
      return true;
    });
    const menu = (it, item) => {
      const pr = app.project.project, fx = (pr && pr.fixed_references) || {};
      if (!pr || !item || !KINDS.includes(item.kind)) return [];
      if (fx[it.id] != null) return [{ id: "unpin", label: `Unpin reference from slot R${fx[it.id]}`, run: () => setFixed(it.id, null) }];
      const taken = new Set(Object.values(fx)), n = [...Array(MAX_REFERENCES)].map((_, k) => k + 1).find((x) => !taken.has(x));
      return n ? [{ id: "pin", label: `Pin reference to slot R${n}`, run: () => setFixed(it.id, n) }] : [];
    };
    app.mediaMenu.push(menu);

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
    const hook = async (ctx) => { const b = await carryHook(ctx); return { inputs: b.inputs || {}, notes: b.notes || [] }; };
    app.inputHooks.push(hook);
    return () => { for (const [list, x] of [[app.inputHooks, hook], [app.mediaMenu, menu]]) { const i = list.indexOf(x); if (i >= 0) list.splice(i, 1); } };
  },
};
