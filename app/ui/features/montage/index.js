// Auto Montage: the button in the timeline header and its window.
import { offer } from "../../shell/actions.js";
import { composer as c } from "../../composer/composer.js";
import { playable, build } from "./build.js";

const snippet = (sc) => { const t = (sc.text || "").trim().replace(/\s+/g, " "); return t ? (t.length > 36 ? `${t.slice(0, 36)}…` : t) : "(no prompt)"; };

export default {
  id: "montage",
  mount: "timeline.actions",
  needs: ["project", "selection"],
  setup({ host, app }) {
    const p = app.project, sel = app.selection;
    const button = c.button.sm({ label: "⚡ Auto Montage", tone: "ghost", disabled: true, onClick: ask });
    host.append(button.node);
    const clips = () => (p.project ? p.scenes.filter((s) => !s.excluded && playable(s, p.project)) : []);

    function ask() {
      const list = clips();
      if (list.length < 2) return c.toast.warn({ text: "Auto Montage needs at least two rendered clips on the timeline: one lead and at least one cutaway." });
      const picked = list.filter((s) => sel.ids.includes(s.id));
      let lead = (picked[0] || list[0]).id;
      const pool = new Set(picked.slice(1).map((s) => s.id));
      let seg = 100, decay = 0.85;
      const label = (s) => `${snippet(s)} — ${Math.round(playable(s, p.project).total * playable(s, p.project).fps)}f`;
      const body = c.region.stack({ gap: "md" });
      const draw = () => body.set([
        c.field.default({ label: "Lead clip (plays in order)", control: c.select.md({ label: "Lead clip", value: lead, options: list.map((s) => ({ value: s.id, label: label(s) })), onChange: (v) => { lead = v; pool.delete(v); draw(); } }) }),
        c.field.default({ label: "Cutaway pool (random draw, in order within each clip)", control: c.checklist.default({ label: "Cutaway pool", values: [...pool],
          items: list.filter((s) => s.id !== lead).map((s) => ({ value: s.id, label: label(s) })), onChange: (v) => { pool.clear(); v.forEach((x) => pool.add(x)); } }) }),
        c.field.default({ label: "Segment length (frames)", control: c.number.md({ label: "Segment length", value: seg, min: 9, step: 1, precision: 0, onChange: (v) => { seg = v; } }) }),
        c.field.default({ label: "Acceleration", hint: "1.0 = constant, lower = faster cuts toward the end", control: c.slider.readout({ label: "Acceleration", value: decay, min: 0.5, max: 1, step: 0.05, onCommit: (v) => { decay = v; } }) }),
      ]);
      draw();
      const win = c.modal.generic({ title: "Auto Montage", subtitle: "Cuts the lead clip into shrinking segments and randomly inserts cutaways between them.", size: "md", body });
      win.setFooter({ actions: [c.button.sm({ label: "Build montage", tone: "primary", onClick: () => {
        if (!pool.size) return c.toast.warn({ text: "Pick at least one cutaway clip." });
        const added = p.edit((pr) => build(pr, { leadId: lead, poolIds: [...pool], segmentFrames: seg, decay }) || false);
        win.close("done");
        c.toast[added ? "good" : "warn"]({ text: added ? `Added ${added} clips to the end of the timeline.` : "Could not build the montage: the clips need more rendered length." });
      } })] });
    }
    const draw = () => button.setDisabled(!p.project);
    draw();
    const offered = offer(app, { id: "auto-montage", label: "Auto Montage", icon: "⚡", run: () => { if (!button.node.disabled) ask(); } });
    const off = app.on(draw);
    return () => { off(); offered(); };
  },
};
