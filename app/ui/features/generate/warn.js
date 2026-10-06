// The chips beside Generate: the pipeline has nowhere to put the scene text; a learning feature is on with no Taste key.
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
    host.append(chip, plain, keyless);
    const draw = (slots) => { chip.hidden = !(slots && !slots.some((s) => (s.roles || []).some((r) => r.at === "generation.prompt"))); };
    const drawPlain = () => {
      const ps = app.pipeline, taste = ps.modulesById?.().taste;
      plain.hidden = !ps.allOff?.();
      keyless.hidden = !(taste && ps.activeModules().includes(taste) && ps.useful(taste) && !String((ps.currentValues().taste || {}).key || "").trim());
    };
    drawPlain();
    draw(app.pipeline.slots());
    return app.pipeline.subscribe ? app.pipeline.subscribe(() => { draw(app.pipeline.slots()); drawPlain(); }) : undefined;
  },
};
