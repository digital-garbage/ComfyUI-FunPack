// Takes: "Takes ×N" makes the picked scene N times with only the seed changing; the Scene tab steps through them.
// Rate each as it is on the clip: taste learns from the comparison, the best stays.
import { composer as c } from "../../composer/composer.js";
import { genUnitId, isGenerative } from "../../shell/scenes.js";
import { pickTake, stepTake, takesOf } from "../../shell/takes.js";

const COUNTS = [2, 3, 4, 6, 8];

export default {
  id: "takes",
  mount: "timeline.actions",
  needs: ["project", "selection", "runner"],
  setup({ host, app }) {
    const p = app.project;
    const many = c.button.sm({ label: "Takes ×N…", tone: "ghost", title: "Make the picked scene several times — only the seed changes — then compare and rate them",
      onClick: async () => {
        const sc = p.selected;
        if (!sc || !isGenerative(sc)) return c.toast.warn({ text: "Pick a generated scene first." });
        const n = await c.modal.choice({ title: "How many takes?", items: COUNTS.map((k) => ({ id: String(k), label: `${k} takes`, hint: k > 4 ? "Takes a while." : "" })) }).result;
        if (n) app.runner.units([genUnitId(sc)], Number(n));
      } });
    host.append(many.node);
    const section = {
      key: (sc, pr) => { const t = takesOf(pr, sc); return `${t.list.length}|${t.at}|${t.head.rating}`; },
      rows: (sc, pr) => {
        const { list, at } = takesOf(pr, sc);
        if (list.length < 2 || at < 0) return [];
        const go = (d) => () => { p.edit((x) => stepTake(x, x.scenes.find((s) => s.id === sc.id), d)); };
        const mark = (r) => (/^(10|[6-9])$/.test(String(r)) ? "★" : r ? "✕" : "");
        const still = (t) => `/funpack/api/m/render/poster?filename=${encodeURIComponent(t.media.filename)}&subfolder=${encodeURIComponent(t.media.subfolder || "")}&type=${encodeURIComponent(t.media.type || "output")}`;
        const items = list.map((t, i) => ({ id: String(i), label: `Take ${i + 1}`, thumb: still(t), icon: "▶", badge: mark(i === at ? takesOf(pr, sc).head.rating : t.rating) || undefined }));
        return [c.label.section({ text: "Takes" }),
          c.gallery.adaptive({ id: "takes", items, selection: [String(at)], cols: 3, empty: "", onActivate: (it) => p.edit((x) => pickTake(x, x.scenes.find((s) => s.id === sc.id), Number(it.id))) }),
          c.field.default({ label: `Take ${at + 1} of ${list.length}`, hint: "Step through the renders of this scene; the one showing is the one on the clip. Rate each, the best stays.",
            control: c.toolbar.default({ items: [c.button.sm({ label: "‹ Earlier", disabled: at === 0, onClick: go(-1) }), c.button.sm({ label: "Later ›", disabled: at === list.length - 1, onClick: go(1) })] }) })];
      },
    };
    app.sceneSections.push(section);
    return () => { const i = app.sceneSections.indexOf(section); if (i >= 0) app.sceneSections.splice(i, 1); many.node.remove(); };
  },
};
