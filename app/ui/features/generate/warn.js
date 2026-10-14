// The chips beside Generate: the pipeline has nowhere to put the scene text; a learning feature is on with no Taste key;
// this torch build runs int8 models slowly.
import { composer as c } from "../../composer/composer.js";

export default {
  id: "generate-warn",
  mount: "timeline.actions",
  needs: ["pipeline"],
  setup({ host, app }) {
    const chip = c.button.sm({ label: "⚠ Nothing encodes your prompt", tone: "ghost", disabled: true,
      title: "This pipeline has no prompt input: the scene text is not sent." }).node;
    chip.hidden = true;
    const plain = c.button.sm({ label: "Enhancements off", tone: "ghost", title: "Disable all enhancements is on: runs use the plain pipeline. Switch it off in Settings ▸ Modules.",
      onClick: () => app.openSettings?.("modules") }).node;
    plain.hidden = true;
    const keyless = c.button.sm({ label: "⚠ No Taste key", tone: "ghost", title: "A feature that learns from ratings is on, but no Taste key is set: ratings teach nothing. Name one in Settings ▸ Engine ▸ System.",
      onClick: () => app.openSettings?.("engine") }).node;
    keyless.hidden = true;
    // Two sparse attentions at once: the pipeline's own SLA and a native Model Sparse Attention node. Both run on
    // every block; the native one's dense fall-throughs print their own log lines, and nothing else says it is there.
    const twice = c.button.sm({ label: "⚠ Two sparse attentions", tone: "ghost", disabled: true,
      title: "SLA and ComfyUI's Model Sparse Attention node are both in this pipeline. Both sparsify attention; remove one." }).node;
    twice.hidden = true;
    // A torch build that halves int8 speed on this GPU (cu12x on Blackwell). Asked once: the build cannot change while ComfyUI runs.
    const slow = c.button.sm({ label: "⚠ Slow torch for int8", tone: "ghost", title: "",
      onClick: () => app.maintenance?.cuda13(slow.title) }).node;
    slow.hidden = true;
    // Shot camera runs without spaCy, but aims its moves with plain word rules.
    const rough = c.button.sm({ label: "Camera targets rougher", tone: "ghost", title: "Shot camera is on, but spaCy is not: camera moves aim with simple word rules and pick worse targets. Switch on Automatic camera and focus in Settings ▸ Modules.",
      onClick: () => app.openSettings?.("modules") }).node;
    rough.hidden = true;
    host.append(chip, plain, keyless, twice, slow, rough);
    Promise.resolve().then(() => app.api?.torchBuild?.()).then((r) => {
      if (r && r.slow_int8) { slow.title = r.slow_int8; slow.hidden = false; }
    }).catch(() => {});      // unknown build: the loader still says it in the log when an int8 model loads
    const draw = (slots) => { chip.hidden = !(slots && !slots.some((s) => (s.roles || []).some((r) => r.at === "generation.prompt"))); };
    const drawTwice = (slots) => {
      const native = (slots || []).some((s) => /SparseAttention/.test(String(s.node || "")));
      const sla = (slots || []).some((s) => (s.inputs || {}).sla === true);
      twice.hidden = !(native && sla);
    };
    const drawPlain = () => {
      const ps = app.pipeline, taste = ps.modulesById?.().taste;
      plain.hidden = !ps.allOff?.();
      keyless.hidden = !(taste && ps.activeModules().includes(taste) && ps.useful(taste) && !String((ps.currentValues().taste || {}).key || "").trim());
    };
    let asking = 0;
    const drawRough = () => {
      const ps = app.pipeline, cam = ps.modulesById?.().conditioning_shot_camera, ask = ++asking;
      if (!cam || !ps.activeModules().includes(cam) || ps.allOff?.()) { rough.hidden = true; return; }
      Promise.resolve().then(() => app.api?.addons?.()).then((all) => {
        if (ask === asking) rough.hidden = !all || all.some((a) => a.id === "camera" && a.state === "installed");
      }).catch(() => {});                                   // no add-on list: say nothing
    };
    drawPlain();
    drawRough();
    draw(app.pipeline.slots());
    drawTwice(app.pipeline.slots());
    return app.pipeline.subscribe ? app.pipeline.subscribe(() => { draw(app.pipeline.slots()); drawTwice(app.pipeline.slots()); drawPlain(); drawRough(); }) : undefined;
  },
};
