// Project Setup Wizard: look, a name and a model's starting point, then the tour offer. Opens from File ▸ Project Setup Wizard.
import { composer as c } from "../../composer/composer.js";

export default {
  id: "wizard",
  mount: "menubar.menus",
  needs: ["project", "api", "pipeline", "theme"],
  setup({ app }) {
    let win = null;
    const ctx = { name: "Untitled montage", preset: "", tour: true };
    let presets = [], step = 0;

    const STEPS = [
      { title: "Choose your look", sub: "You can change this any time in Settings ▸ Appearance.", body: () => c.segmented.md({ label: "Colour scheme", value: app.theme.get(), onChange: (v) => app.theme.apply(v),
        options: [{ value: "dark", label: "Dark" }, { value: "light", label: "Light" }, { value: "auto", label: "Auto" }] }) },
      { title: "Name it, and pick a model", sub: "The model decides which nodes and which files this project needs.", body: () => c.region.stack({ gap: "sm", children: [
        c.field.default({ label: "Project name", control: c.input.md({ label: "Project name", value: ctx.name, onInput: (v) => { ctx.name = v; } }) }),
        c.field.default({ label: "Start from", hint: "A model's own pipeline, ready to fill with files. “Keep the current pipeline” leaves Models & Pipeline as it is.",
          control: c.select.md({ label: "Start from", value: ctx.preset, onChange: (v) => { ctx.preset = v; },
            options: [{ value: "", label: "Keep the current pipeline" }, ...presets.map((p) => ({ value: p.id, label: p.title }))] }) })] }) },
      { title: "Want the guided tour?", sub: "A walk through the screen, one part at a time. It changes nothing; Help ▸ Restart tour runs it again.", body: () => c.toggle.default({ label: "Show me around when I finish", checked: ctx.tour, onChange: (v) => { ctx.tour = v; } }) },
    ];

    let busy = false;
    async function finish() {
      if (busy) return;
      busy = true;
      const pick = presets.find((p) => p.id === ctx.preset);
      try {
        await app.project.newProject((ctx.name || "").trim() || "Untitled montage");
        if (pick) { const r = await app.pipeline.restore({ slots: pick.slots, removed: [], unwired: {} }); if (r.refused.length) c.toast.warn({ text: `The model's pipeline was not loaded: ${r.refused[0]}` }); }
      } catch (err) { busy = false; return c.toast.warn({ text: `Could not make the project: ${err.message}` }); }
      busy = false;
      close();
      if (ctx.tour) setTimeout(() => app.say("tour.start"), 300);
    }
    const close = () => { if (win) { const w = win; win = null; w.close("done"); } };
    function draw() {
      const s = STEPS[step], last = step === STEPS.length - 1;
      body(s);
      win.setFooter({ actions: [step ? c.button.sm({ label: "Back", tone: "ghost", onClick: () => { step -= 1; draw(); } }) : null,
        c.button.sm({ label: last ? "Create project" : "Continue", tone: "primary", onClick: () => (last ? finish() : (step += 1, draw())) })].filter(Boolean) });
    }
    const slot = c.region.stack({ gap: "md" });
    const body = (s) => (slot.set([c.label.section({ text: `${step + 1} / ${STEPS.length} · ${s.title}` }), c.hint.default({ text: s.sub }), s.body()]), slot);
    async function open() {
      if (win) return;
      try { presets = (await app.api.pipelinePresets()).presets || []; } catch { presets = []; }
      step = 0; ctx.name = "Untitled montage"; ctx.preset = ""; ctx.tour = true;
      win = c.modal.generic({ title: "Project Setup", size: "md", closeOnOutside: false, body: body(STEPS[0]), onClose: () => { win = null; } });
      draw();
    }
    app.has.add("wizard");
    const off = app.on((what) => { if (what === "wizard.open") open(); });
    return () => { off(); app.has.delete("wizard"); };
  },
};
