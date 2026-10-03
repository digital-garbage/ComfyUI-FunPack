// Entry point: build the frame, start the core, load the features. Nothing here names a feature.
import { build } from "./shell/frame.js";
import { loadFeatures } from "./shell/features.js";
import { createSettings } from "./shell/settings.js";
import { createProject } from "./shell/project.js";
import { hostFor } from "./shell/mounts.js";
import paths from "./modules.js";

// What the core offers a feature, by name. `on(fn)` hears every change; fn gets "open" when a
// different project was opened, "change" for an edit.
const heard = new Set();
const say = (what) => heard.forEach((fn) => { try { fn(what); } catch (err) { console.error(err); } });
const app = {
  project: createProject({ onChange: () => say("change"), onOpen: () => say("open"), onError: (err) => console.warn(err) }),
  on: (fn) => { heard.add(fn); return () => heard.delete(fn); },
};

const frame = build(document.getElementById("app"));
const settings = createSettings({ menubar: hostFor("menubar.menus") });
const result = await loadFeatures(paths, { app });
settings.addButton();
app.project.start().catch((err) => console.warn("could not open a project:", err.message));
window.FunPackUI = { frame, settings, app, ...result };
