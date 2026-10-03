// Entry point: build the frame, load the features. Nothing here names a feature.
import { build } from "./shell/frame.js";
import { loadFeatures } from "./shell/features.js";
import paths from "./modules.js";

const frame = build(document.getElementById("app"));
const result = await loadFeatures(paths);
window.FunPackUI = { frame, ...result };
