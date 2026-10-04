// The chip beside Generate when the pipeline has nowhere to put the scene text.
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
    host.append(chip, plain);
    const draw = (slots) => { chip.hidden = !(slots && !slots.some((s) => (s.roles || []).some((r) => r.at === "generation.prompt"))); };
    const drawPlain = () => { plain.hidden = !app.pipeline.allOff?.(); };
    drawPlain();
    draw(app.pipeline.slots());
    return app.pipeline.subscribe ? app.pipeline.subscribe(() => { draw(app.pipeline.slots()); drawPlain(); }) : undefined;
  },
};
