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
    host.append(chip);
    const draw = (slots) => { chip.hidden = !(slots && !slots.some((s) => (s.roles || []).some((r) => r.at === "generation.prompt"))); };
    draw(app.pipeline.slots());
    return app.pipeline.subscribe ? app.pipeline.subscribe(draw) : undefined;
  },
};
