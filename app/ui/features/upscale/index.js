// Upscale a finished render with a model from ComfyUI's upscale_models, or with NVIDIA RTX Super Resolution; the result replaces the render. Never / button beside the rating / always.
import { composer as c } from "../../composer/composer.js";
import { inPlace } from "../../shell/place.js";
import { genUnitId, isGenerative, unitRoot } from "../../shell/scenes.js";
import { RTX, RTX_DEFAULTS, UPSCALED, keepsSize, keyOf, qualityLabel, swap, upscale } from "./job.js";

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

    const rtxOn = () => p.pref("upscaleEngine", "model") === "rtx";
    const rtxOpts = () => ({ ...RTX_DEFAULTS, ...(p.pref("upscaleRtx", {}) || {}) });
    // Pressed with nothing to upscale with: say what would fix it, and take the person there.
    async function suggest() {
      let all = []; try { all = await app.api.addons(); } catch { /* no add-on list: the model way alone */ }
      const vsr = all.find((a) => a.id === "nvidia_vsr");
      const items = [
        ...(vsr ? [vsr.state === "installed" ? { id: "editor", label: "Use NVIDIA RTX Super Resolution", hint: "It is installed: pick it as the Engine in Settings ▸ Editor." }
          : vsr.state === "pending" ? { id: "modules", label: "NVIDIA RTX Super Resolution is switched on", hint: "Restart ComfyUI to install it." }
          : { id: "modules", label: "Install NVIDIA RTX Super Resolution", hint: "Native upscaling, no model file needed. Switch it on in Settings ▸ Modules." }] : []),
        { id: "editor", label: "Download an upscale model", hint: "Put an upscale model (.safetensors) in ComfyUI/models/upscale_models, then pick it in Settings ▸ Editor." }];
      const pick = await c.modal.choice({ title: "Nothing to upscale with yet", items }).result;
      if (pick) app.openSettings(pick);
    }

    async function run(media, pressed = false) {
      const model = rtxOn() ? RTX : p.pref("upscaleModel", "");
      if (!media || media.subfolder === UPSCALED) return;                    // never upscale twice
      if (!model) return pressed ? suggest() : c.toast.warn({ text: "Upscale: pick a model or NVIDIA RTX Super Resolution in Settings ▸ Editor." });
      const key = keyOf(media), pid = p.project.id;
      if (busy.has(key)) return;
      busy.add(key); draw(); c.toast.info({ text: `Upscaling with ${model}…` });
      try {
        const out = await upscale(app.api, media, model, undefined, rtxOn() ? rtxOpts() : undefined);
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
        title: "Upscale this video with the engine picked in Settings ▸ Editor; it replaces the render", onClick: () => run(media, true) }).node);
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
      const set = (k, v) => { p.setPref("upscaleRtx", { ...rtxOpts(), [k]: v }); draw(); paint(info); };
      const gone = (v) => ({ value: v, label: `${v} (missing)` });
      const rtxRows = (r) => {
        const o = rtxOpts(), same = keepsSize(o.quality);
        return [
          c.settingsRow.default({ label: "Quality", hint: "Upscale is for normal videos, High bitrate for clean ones. Denoise and Deblur fix a video at its own size.", control: c.select.md({ label: "Quality", value: o.quality, onChange: (v) => set("quality", v),
            options: (r.rtx.qualities || []).map((q) => ({ value: q, label: qualityLabel(q) })) }) }),
          ...(same ? [c.hint.default({ text: "This quality keeps the video's size, so there is no scale." })] : [
            c.settingsRow.default({ label: "Size by", hint: "A multiplier of the video's size, or one fixed size.", control: c.select.md({ label: "Size by", value: o.resize, onChange: (v) => set("resize", v),
              options: [{ value: "scale", label: "Scale by multiplier" }, { value: "size", label: "Target dimensions" }] }) }),
            ...(o.resize === "scale"
              ? [c.settingsRow.default({ label: "Scale", hint: "1 to 4 times.", control: c.slider.readout({ label: "Scale", value: o.scale, min: 1, max: 4, step: 0.05, precision: 2, onCommit: (v) => set("scale", v) }) })]
              : [c.settingsRow.default({ label: "Width", hint: "In pixels, 64 to 8192.", control: c.number.md({ label: "Width", value: o.width, min: 64, max: 8192, step: 8, precision: 0, onChange: (v) => set("width", v) }) }),
                 c.settingsRow.default({ label: "Height", hint: "In pixels, 64 to 8192.", control: c.number.md({ label: "Height", value: o.height, min: 64, max: 8192, step: 8, precision: 0, onChange: (v) => set("height", v) }) })])]),
          c.settingsRow.default({ label: "Strength", hint: "How much of the effect to apply.", control: c.slider.readout({ label: "Strength", value: o.strength, min: 0, max: 1, step: 0.05, precision: 2, onCommit: (v) => set("strength", v) }) }),
          c.toggle.default({ label: "10-bit processing", hint: "More precise colour inside the upscale. The saved video stays 8-bit.", checked: o.tenbit, onChange: (v) => set("tenbit", v) }),
        ];
      };
      let info = null;
      const paint = (r) => {
        info = r;
        const chosen = p.pref("upscaleEngine", "model"), models = r === null ? null : r.models || [];
        page.set([
          c.label.section({ text: "Upscale finished renders" }),
          c.hint.default({ text: "Runs an upscale over a finished video and replaces the render with the result." }),
          c.settingsRow.default({ label: "Upscale", hint: "Never, with a button beside the rating, or on every new render.", control: c.select.md({ label: "Upscale", value: p.pref("upscaleMode", "never"),
            options: [{ value: "never", label: "Never" }, { value: "button", label: "When I press Upscale" }, { value: "always", label: "Always, when a render finishes" }], onChange: (v) => p.setPref("upscaleMode", v) }) }),
          c.settingsRow.default({ label: "Engine", hint: r && r.rtx ? "An upscale model from a file, or the RTX card's own upscaler." : "An upscale model from a file. NVIDIA RTX Super Resolution can be switched on in Settings ▸ Modules.",
            control: c.select.md({ label: "Engine", value: chosen, onChange: (v) => { p.setPref("upscaleEngine", v); paint(info); },
              options: [{ value: "model", label: "Upscale model" }, ...(r && r.rtx ? [{ value: "rtx", label: RTX }] : chosen === "rtx" ? [{ value: "rtx", label: `${RTX} (not installed)` }] : [])] }) }),
          ...(chosen === "rtx" ? (r && r.rtx ? rtxRows(r) : [c.banner.warn({ text: `${RTX} is not installed here: switch it on in Settings ▸ Modules and restart ComfyUI.` })])
            : [c.settingsRow.default({ label: "Model", hint: "From ComfyUI/models/upscale_models.", control: models === null ? c.text.sm({ text: "Loading…" })
              : c.select.md({ label: "Model", value: p.pref("upscaleModel", ""), onChange: (v) => p.setPref("upscaleModel", v),
                options: [{ value: "", label: models.length ? "— pick a model —" : "Nothing in models/upscale_models" }, ...models.map((m) => ({ value: m, label: m })),
                  ...(p.pref("upscaleModel", "") && !models.includes(p.pref("upscaleModel", "")) ? [gone(p.pref("upscaleModel", ""))] : [])] }) })]),
        ]);
      };
      paint(null);
      app.api.upscaleModels().then(paint).catch(() => page.set([c.banner.warn({ text: "Could not list upscale models." })]));
      return page;
    });

    draw(); watch();
    return app.on(() => { watch(); draw(); });
  },
};
