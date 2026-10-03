// Upscale a finished render with a model from ComfyUI's upscale_models; the result replaces the render. Never / button beside the rating / always.
import { composer as c } from "../../composer/composer.js";
import { inPlace } from "../../shell/place.js";
import { genUnitId, isGenerative, unitRoot } from "../../shell/scenes.js";
import { UPSCALED, keyOf, swap, upscale } from "./job.js";

export default {
  id: "upscale",
  mount: "timeline.actions",
  needs: ["project", "api", "selection"],
  setup({ host, app }) {
    const p = app.project, sel = app.selection;
    const put = inPlace(host), busy = new Set();
    const render = (id) => ((p.project || {}).scene_renders || {})[id];
    const target = () => {                               // the picked clip's generation unit root, when it has a render
      const open = p.project, sc = open && open.scenes.find((s) => s.id === sel.focus);
      const root = sc && isGenerative(sc) ? unitRoot(open, genUnitId(sc)) : null;
      const r = root && render(root.id);
      return r && r.media ? r.media : null;
    };

    async function run(media) {
      const model = p.pref("upscaleModel", "");
      if (!media || media.subfolder === UPSCALED) return;                    // never upscale twice
      if (!model) return c.toast.warn({ text: "Upscale: pick a model in Settings ▸ Editor ▸ Upscale finished renders." });
      const key = keyOf(media), pid = p.project.id;
      if (busy.has(key)) return;
      busy.add(key); draw(); c.toast.info({ text: `Upscaling with ${model}…` });
      try {
        const out = await upscale(app.api, media, model);
        const swapped = await p.editFor(pid, (pr) => swap(pr, media, out) > 0);
        if (!swapped) c.toast.warn({ text: "Upscaled, but that render was replaced meanwhile, so nothing was swapped." });
        else c.toast.good({ text: "Upscaled: the render now plays the larger video." });
      } catch (err) { c.toast.warn({ text: `Upscale failed: ${err.message}` }); }
      busy.delete(key); draw();
    }

    function draw() {
      const media = p.pref("upscaleMode", "never") === "button" ? target() : null;
      if (!media) return put();
      const working = busy.has(keyOf(media)) || media.subfolder === UPSCALED;
      put(c.button.sm({ label: busy.has(keyOf(media)) ? "Upscaling…" : media.subfolder === UPSCALED ? "Upscaled" : "Upscale", tone: "ghost", disabled: working,
        title: "Upscale this video with the model picked in Settings ▸ Editor; it replaces the render", onClick: () => run(media) }).node);
    }

    // "Always": a render that appears after the project was opened is upscaled. What was there on open is left alone.
    let seen = null, obj = null;
    const watch = () => {
      const open = p.project;
      if (!open) { seen = null; obj = null; return; }
      const now = new Map(Object.entries(open.scene_renders || {}).filter(([, r]) => r && r.media).map(([id, r]) => [id, keyOf(r.media)]));
      if (open !== obj || !seen) { obj = open; seen = now; return; }       // another project, or undo/redo put an older copy back: not a new render
      if (p.pref("upscaleMode", "never") === "always") {
        const fresh = new Map([...now].filter(([id, k]) => seen.get(id) !== k));
        for (const media of new Map([...fresh.keys()].map((id) => [fresh.get(id), open.scene_renders[id].media])).values()) run(media);
      }
      seen = now;
    };

    app.editorSettings.push(() => {
      const page = c.region.stack({ gap: "md" });
      const row = (models) => page.set([
        c.label.section({ text: "Upscale finished renders" }),
        c.hint.default({ text: "Runs an upscale model over a finished video and replaces the render with the result." }),
        c.settingsRow.default({ label: "Upscale", hint: "Never, with a button beside the rating, or on every new render.", control: c.select.md({ label: "Upscale", value: p.pref("upscaleMode", "never"),
          options: [{ value: "never", label: "Never" }, { value: "button", label: "When I press Upscale" }, { value: "always", label: "Always, when a render finishes" }], onChange: (v) => p.setPref("upscaleMode", v) }) }),
        c.settingsRow.default({ label: "Model", hint: "From ComfyUI/models/upscale_models.", control: models === null ? c.text.sm({ text: "Loading…" })
          : c.select.md({ label: "Model", value: p.pref("upscaleModel", ""), onChange: (v) => p.setPref("upscaleModel", v),
            options: [{ value: "", label: models.length ? "— pick a model —" : "Nothing in models/upscale_models" }, ...models.map((m) => ({ value: m, label: m })),
              ...(p.pref("upscaleModel", "") && !models.includes(p.pref("upscaleModel", "")) ? [{ value: p.pref("upscaleModel", ""), label: `${p.pref("upscaleModel", "")} (missing)` }] : [])] }) }),
      ]);
      row(null);
      app.api.upscaleModels().then((r) => row(r.models || [])).catch(() => page.set([c.banner.warn({ text: "Could not list upscale models." })]));
      return page;
    });

    draw(); watch();
    return app.on(() => { watch(); draw(); });
  },
};
