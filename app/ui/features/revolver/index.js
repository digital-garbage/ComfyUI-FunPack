// Shortcut revolver: a shortcut with several replacements cycles through them without repeating until all have fired.
// The mode lives on the server (previews and generation share one cycle); the open project remembers it for a fresh machine.
import { composer as c } from "../../composer/composer.js";

export default {
  id: "revolver",
  mount: "menubar.menus",
  needs: ["project", "api"],
  setup({ app }) {
    const p = app.project, api = app.api;
    app.editorSettings.push(() => {
      const page = c.region.stack({ gap: "md" });
      const draw = (s, busy = false) => page.set([
        c.label.section({ text: "Shortcut revolver" }),
        c.hint.default({ text: "Off: a seeded random pick that may repeat. On: no replacement is reused until the whole set has fired. A server setting for every browser and for generation; the open project remembers it." }),
        c.toggle.default({ label: "Shortcut revolver", checked: s.enabled, disabled: busy, onChange: (v) => save({ enabled: v, random: s.random }) }),
        s.enabled ? c.toggle.default({ label: "Random", hint: "Off: replacements fire in order, then the cycle restarts. On: any order, still no repeats. Changing either restarts all cycles.", checked: s.random, disabled: busy, onChange: (v) => save({ enabled: s.enabled, random: v }) }) : null,
      ].filter(Boolean));
      const save = async (next) => {
        draw(next, true);
        try { const now = await api.setRevolver(next.enabled, next.random); draw(now); p.setPref("revolver", { enabled: Boolean(now.enabled), random: Boolean(now.random) }); }
        catch (err) { c.toast.warn({ text: `Revolver not changed: ${err.message}` }); api.revolver().then((s) => draw(s)).catch(() => {}); }
      };
      page.set([c.hint.default({ text: "Loading…" })]);
      api.revolver().then((s) => draw(s)).catch(() => page.set([c.banner.warn({ text: "Could not load revolver settings." })]));
      return page;
    });

    // A project opened on a machine whose server has not heard of its choice puts it back.
    let last = null;
    return app.on((what) => {
      const open = p.project, want = open && p.pref("revolver", null);
      if (what !== "open" || !want || last === open.id) return;
      last = open.id;
      api.revolver().then((s) => (s.enabled !== want.enabled || s.random !== want.random ? api.setRevolver(want.enabled, want.random) : null)).catch(() => {});
    });
  },
};
