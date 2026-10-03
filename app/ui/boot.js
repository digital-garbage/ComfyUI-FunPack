// Entry point: build the frame, load the features. Nothing here names a feature.
import { build } from "./shell/frame.js";
import { loadFeatures } from "./shell/features.js";
import { createSettings } from "./shell/settings.js";
import { hostFor } from "./shell/mounts.js";
import paths from "./modules.js";

const frame = build(document.getElementById("app"));
const settings = createSettings({ menubar: hostFor("menubar.menus") });
const result = await loadFeatures(paths);
settings.addButton();
window.FunPackUI = { frame, settings, ...result };
