// Entry point: build the frame, start the core, load the features. Nothing here names a feature.
import { build } from "./shell/frame.js";
import { loadFeatures } from "./shell/features.js";
import { createSettings } from "./shell/settings.js";
import { createProject } from "./shell/project.js";
import { createPipelineState } from "./shell/pipeline_state.js";
import { createGenerate } from "./shell/generate.js";
import { linkPipeline } from "./shell/pipeline_link.js";
import { api } from "./shell/api.js";
import { createSelection } from "./shell/selection.js";
import { createPlayhead } from "./shell/playhead.js";
import { createMaintenance } from "./shell/maintenance.js";
import { syncSeparated } from "./shell/audio.js";
import { createKeys } from "./shell/keys.js";
import { createDom } from "./shell/dom.js";
import { createPeek } from "./shell/peek.js";
import { createDock } from "./shell/dock.js";
import { hostFor } from "./shell/mounts.js";
import paths from "./modules.js";
import { composer as c } from "./composer/composer.js";
import { startLoading } from "./shell/loading.js";

// First thing on screen, before the frame exists: the buttons arrive one by one and this says why.
const loading = startLoading(c);

// What the core offers a feature, by name. `on(fn)` hears every change; fn gets "open" when a
// different project was opened, "change" for an edit.
const heard = new Set();
const say = (what) => heard.forEach((fn) => { try { fn(what); } catch (err) { console.error(err); } });
const on = (fn) => { heard.add(fn); return () => heard.delete(fn); };
const project = createProject({ keepConsistent: syncSeparated, onChange: () => say("change"), onOpen: () => say("open"), onError: (err) => console.warn(err), beforeSwitch: () => pipeline.settled() });
const pipeline = createPipelineState(api);
const generate = createGenerate({ pipeline });
const link = linkPipeline({ project, pipeline, onOpen: (fn) => on((what) => { if (what === "open") fn(); }), say: (text) => c.toast.warn({ text }) });
const frame = build(document.getElementById("app"));
const selection = createSelection({ project });
selection.on(() => say("select"));
const app = { editorSettings: [], timelineLanes: [], timelineView: {}, monitor: {}, mediaMenu: [], mediaPeek: {}, sceneSections: [], clipTags: [], has: new Set(), actions: [], inputHooks: [], beforeRun: [], promptPrefix: [], lastRun: { typed: undefined, n: 0 }, runner: {}, project, pipeline, pipelineOwned: link.owns, pipelineWhy: link.why, generate, api, on, say, title: frame.setZoneTitle, selection, playhead: createPlayhead(), theme: window.ComposerTheme, dock: createDock(), maintenance: createMaintenance({ api, flush: async () => { await project.flush(); if (project.unsaved) throw new Error("Your latest edits could not be saved, so nothing was changed. Try again in a moment."); } }), keys: createKeys(), dom: createDom(), peek: createPeek(), resetLayout: () => { frame.resetLayout(); app.dock.reset(); } };
// Which backend modules the server has (loaded, even if this model cannot use them). Unknown (server down) = assume all, so nothing is hidden by a failed fetch.
const installed = await Promise.race([api.modules(), new Promise((_, no) => setTimeout(() => no(new Error("slow")), 4000))])      // a stalled server must not leave the page blank
  .then((m) => (Array.isArray(m.modules) ? new Set([...m.modules, ...(m.incompatible || [])].map((x) => x.id)) : null)).catch(() => null);
app.hasModule = (id) => !installed || installed.has(id);
const settings = createSettings({ menubar: hostFor("menubar.menus") });
app.openSettings = (id) => settings.open(id);
const result = await loadFeatures(paths, { app, onProgress: loading.progress }).finally(loading.done);
settings.addButton();
app.project.start().catch((err) => console.warn("could not open a project:", err.message));
window.FunPackUI = { frame, settings, app, ...result };
