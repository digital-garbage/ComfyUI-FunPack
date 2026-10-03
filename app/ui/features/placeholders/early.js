// Stand-ins that sit BEFORE a real feature in the same row (see index.js).
import { composer as c } from "../../composer/composer.js";

export default [{ id: "placeholder:sampler", mount: "timeline.status",
  setup: ({ host }) => host.append(c.button.sm({ label: "⏱ Sampler", tone: "neutral", disabled: true }).node) }];
