// "Scene N · Rate…": say how a render turned out. FunPack's taste learns from it on the next Generate.
import { composer as c } from "../../composer/composer.js";
import { inPlace } from "../../shell/place.js";
import { genUnitId, isGenerative, unitRoot } from "../../shell/scenes.js";
import { CHOICES, FORGET, nameOf, tasteOf } from "./choices.js";

/** What to say when a rating taught nothing, or "" when it did. The run captured nothing, a learner is on and no key is
 *  set: the missing key is why. Any other reason is the server's own (the key on screen now may not be the run's). */
export function notTaught(r, ps) {
  if (!(r && r.why)) return "";
  const taste = ps.modulesById && ps.modulesById().taste;
  const keyless = r.reason === "nothing" && taste && ps.useful(taste) && !String((ps.currentValues().taste || {}).key || "").trim();
  return keyless ? "Taste not taught: no Taste key is set, so nothing learns from ratings. Name one in Settings ▸ Engine ▸ System ▸ Taste key." : `Taste not taught: ${r.why}.`;
}

export default {
  id: "rating",
  mount: "timeline.actions",
  needs: ["project", "api", "selection", "pipeline"],
  setup({ host, app }) {
    const p = app.project, sel = app.selection;
    const put = inPlace(host);
    const target = () => {                                  // the picked clip's generation unit, when it has a render to talk about
      const open = p.project, sc = open && open.scenes.find((s) => s.id === sel.focus);
      if (!sc || !isGenerative(sc)) return null;
      const root = unitRoot(open, genUnitId(sc));
      const render = root && (open.scene_renders || {})[root.id];
      return render && render.media ? { root, render, no: open.scenes.indexOf(sc) + 1 } : null;
    };
    const rate = async (value) => {
      const t = target();
      if (!t) return;
      p.setScene(t.root.id, "rating", value);
      const taste = tasteOf(value);
      if (!t.render.promptId || taste === null) return;      // no run to pair it with, or a word that teaches nothing: it stays a label
      const body = taste === "clear" ? { rating: null, axis: null } : taste;
      try {
        const r = await app.api.rateTaste(t.render.promptId, body.rating, body.axis);
        const text = notTaught(r, app.pipeline);
        if (text) c.toast.warn({ text });
      } catch { /* no taste module here: the rating stays a label */ }
    };
    const draw = () => {
      const t = target();
      if (!t) return put();
      const now = nameOf(t.root.rating);
      const button = c.button.sm({ label: now ? `★ ${now}` : "Rate scene…", tone: now ? "neutral" : "ghost", title: "Rate this scene's render: features that learn from ratings use it on the next generation",
        onClick: async () => {
          const pick = await c.modal.choice({ title: "How did it turn out?", items: [...CHOICES.map((o) => ({ id: o.value, label: o.label, hint: o.hint })), { id: FORGET, label: "Forget my rating", hint: "Clears it: nothing is learned from this clip." }] }).result;
          if (pick) rate(pick);
        } });
      put(c.text.sm({ text: `Scene ${t.no}` }).node, button.node);
    };
    draw();
    return app.on(draw);
  },
};
