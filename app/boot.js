// Entry point.
//
// The whole of what the app does at startup: build the regions, ask the server
// what exists, and mount whatever announced itself. Nothing here names a module.

import { build } from "./shell/layout.js";
import { fetchManifest } from "./shell/manifest.js";
import { mountAll } from "./shell/panels.js";
import { settle } from "./shell/mounts.js";
import { all as allValues } from "./shell/values.js";
import { composer } from "./composer/composer.js";
import { createRun, viewUrl, DONE, FAILED, CANCELLED } from "./shell/run.js";
import { clientId, connect, queuedFor, finishedFor } from "./shell/client.js";
import { wire, waitForTerminal } from "./shell/session.js";
import { check, load, describe, search } from "./shell/pipeline.js";
import { open as openPipeline } from "./shell/pipeline_window.js";
import { createPrompts } from "./shell/prompt.js";
import { createProject } from "./shell/projects.js";
import { createTimeline } from "./shell/timeline.js";
import { createInspector } from "./shell/inspector.js";
import { createWheel } from "./shell/wheel.js";
import { offerAction } from "./shell/actions.js";
import { open as openWizard } from "./shell/wizard.js";
import { open as openUpdates } from "./shell/updates.js";
import { open as openPacks } from "./shell/packs.js";
import { open as openLog } from "./shell/logwindow.js";
import { open as openTemp } from "./shell/tempfiles.js";

const root = document.querySelector("#app");

async function start() {
  const id = clientId();
  const run = createRun({ clientId: id, connect });

  // The pipeline the user has edited, if they have. Held here and passed in
  // rather than kept inside a module: the app has one store of live values
  // already, and a second one hiding in a transport file is a second place to
  // look when a run turns out to have used something other than what is on
  // screen. Null until the window says otherwise, which means "the server's
  // defaults" -- so a fresh page generates without the window ever opening.
  let slots = null;

  // The inputs the pipeline puts on the main window -- the prompt, today. Null
  // until the pipeline has been read, because until then nobody knows which
  // inputs those are.
  let prompts = null;

  // The project, and the strip that draws it. Both exist before the manifest is
  // read so a failure to reach the server leaves an empty timeline rather than
  // an undefined one.
  let timeline = null;
  let inspector = null;
  const project = createProject({
    onChange: () => {
      if (timeline) timeline.draw();
      if (inspector) inspector.draw();
    },
    // A different project is open. Everything that follows the project rather
    // than one of its fields is put back here -- by every route in, including
    // the ones that are not the File menu.
    onOpen: () => { showScene(project.selected); syncVideo(); },
    onError: (err) => page.transport.say(`The project could not be saved: ${err.message}`),
  });

  // A group with nothing in it is a heading over nothing. The pipeline decides
  // which inputs belong on the app's surface, so whether this group exists at
  // all is only known once its controls have been built -- and again whenever
  // the pipeline changes underneath.
  const showGroups = () => {
    const empty = !prompts || !prompts.controlsAt("project.video").length;
    page.constructor.video.node.toggleAttribute("hidden", empty);
  };

  /**
   * Put the open project's settings into the controls that produce them.
   *
   * Every time EITHER side moves: a different project is opened, or the pipeline
   * changed and the controls were rebuilt from its defaults. Seeding once at
   * startup left the boxes -- and so `overrides()`, and so the run -- holding the
   * numbers of whatever project was open first. The store's own getter reported
   * the new project correctly, so the only place the stale value showed was a
   * window nobody had a reason to reopen.
   */
  const syncVideo = () => {
    // A project that never touched an input is not "leave it alone" -- it is
    // "this input has no opinion here", which means the pipeline's own
    // default. Without the fallback, switching FROM a project that set width
    // TO one that never did left the box (and so overrides(), and so the run)
    // still showing the first project's number.
    for (const { input, control } of (prompts ? prompts.controlsAt("project.video") : [])) {
      control.setValue(project.video[input] !== undefined ? project.video[input] : control.default);
    }
  };

  // A scene's text and the prompt box are the same value seen twice. The box is
  // the editor; the scene is where it lives.
  const showScene = (scene) => {
    const box = prompts && prompts.at("generation.prompt");
    if (box) box.setValue(scene ? scene.text : "");
    // The properties column is about ONE scene, and says which.
    const at = scene ? project.scenes.findIndex((s) => s.id === scene.id) + 1 : 0;
    if (page && page.properties) page.properties.setTitle(at ? `Scene · ${at}` : "Scene");
  };

  // Which scene a run was started for, read at the moment it starts. A run
  // takes minutes and the user goes on clicking, so "the selected scene" at the
  // end is not the one that was generated. The project id travels with it --
  // nothing stops the user opening a DIFFERENT project while this one is in
  // flight, and the result still belongs to the project it was started in.
  let ranFor = null;
  let ranForProject = null;

  // Before the page, because the properties column is composed with it in.
  inspector = createInspector({ project, onRename: (name) => project.rename(name) });

  // "Load Project File..." trigger. One input, kept and reused rather than
  // built fresh per click -- a fresh one would need its `change` listener
  // rewired every time, which is a listener leaked on every click but the last.
  const loadProjectFile = document.createElement("input");
  loadProjectFile.type = "file";
  loadProjectFile.accept = "application/json,.json";
  loadProjectFile.hidden = true;
  document.body.appendChild(loadProjectFile);
  loadProjectFile.addEventListener("change", async () => {
    const file = loadProjectFile.files[0];
    loadProjectFile.value = "";                   // the same file picked twice must still change
    if (!file) return;
    try {
      await project.importProject(JSON.parse(await file.text()));
    } catch (err) {
      page.transport.say(`Could not load that project file: ${err.message}`);
    }
  });

  // What Generate does for whichever scene is selected right now -- shared
  // with "Generate All" below, so the two never disagree about how a run gets
  // attributed to its scene.
  const generateCurrentScene = () => {
    ranFor = project.selectedId;
    ranForProject = project.project ? project.project.id : null;
    return session.generate();
  };

  // Every scene, one at a time, waiting for each to actually finish before
  // starting the next. Not a montage -- v5 has no render/stitch stage -- this
  // is N separate generations, each landing on its own scene, the same as
  // pressing Generate N times without having to sit at the keyboard for each.
  let batchCancelled = false;
  async function generateAll() {
    // Excluded scenes are dropped from the count as well as the loop -- "3 of
    // 5" while two of the five were never going to run reads as a wrong
    // number, not as a batch that respected the exclude.
    const scenes = project.scenes.filter((s) => !s.excluded);
    if (!scenes.length) return;
    generateAllBtn.setDisabled(true);
    batchCancelled = false;
    page.transport.hold(`Generating scene 1 of ${scenes.length}…`);
    const failed = [];
    try {
      for (let i = 0; i < scenes.length; i++) {
        if (batchCancelled) break;
        const scene = scenes[i];
        // A scene removed -- or excluded -- mid-batch by whoever is not at
        // the keyboard right now is not a scene left to generate.
        const live = project.scenes.find((s) => s.id === scene.id);
        if (!live || live.excluded) continue;
        project.select(scene.id);
        showScene(project.selected);
        page.transport.say(`Generating scene ${i + 1} of ${scenes.length}…`);
        const waiting = waitForTerminal(run);
        const queued = await generateCurrentScene();
        if (!queued) {
          // Refused before it ever reached run.start() -- nothing will ever
          // transition the run, so nothing will ever resolve `waiting` either.
          waiting.cancel();
          failed.push(scene);
          continue;
        }
        const phase = await waiting;
        if (phase === CANCELLED) { batchCancelled = true; break; }
        if (phase === FAILED) failed.push(scene);
      }
    } finally {
      generateAllBtn.setDisabled(false);
      page.transport.release(run.state);
    }
    if (batchCancelled) page.transport.say("Generate All was cancelled.");
    else if (failed.length) page.transport.say(`${failed.length} of ${scenes.length} scenes did not generate.`);
  }
  const generateAllBtn = composer.button.sm({
    label: "Generate All", tone: "ghost",
    onClick: () => generateAll(),
  });

  const page = build(root, {
    inspector,
    onGenerate: generateCurrentScene,
    generateAll: generateAllBtn,
    onCancel: () => run.cancel(),
    onConstructor: () => page.constructor.open(),
    // The window asks whether a run is in flight, because the restart that
    // follows an update would take it with it.
    onUpdates: () => openUpdates({ running: () => ["queued", "running"].includes(run.state.phase) }),
    onPacks: () => openPacks(),
    onLog: () => openLog(),
    // Opening one puts it in the Preview, which is where somebody hunting for a
    // file wants it -- the same place a result from the bin goes.
    onTemp: () => openTemp({ onOpen: (item) => page.viewer.setSource(
      item.url, item.kind, item.file ? { ...item.file, type: "temp" } : null) }),
    // What the Edit menu offers, and what the keyboard reaches. One list, so a
    // menu item and its shortcut cannot drift apart.
    edits: {
      canUndo: () => project.canUndo,
      canRedo: () => project.canRedo,
      excluded: () => Boolean(project.selected && project.selected.excluded),
      run: (id) => {
        if (id === "undo") project.undo();
        else if (id === "redo") project.redo();
        else if (id === "scene") { project.addScene(); showScene(project.selected); }
        else if (id === "remove" && project.scenes.length > 1) {
          project.removeScene(project.selectedId);
          showScene(project.selected);
        } else if (id === "exclude" && project.selected) {
          project.setScene(project.selectedId, "excluded", !project.selected.excluded);
        } else if (id === "earlier") project.move(project.selectedId, -1);
        else if (id === "later") project.move(project.selectedId, 1);
      },
    },
    projects: () => project.recent,
    currentProject: () => (project.project ? project.project.id : null),
    onProject: async (id) => {
      if (id === "new") {
        const asked = composer.modal.prompt({
          title: "New project", label: "Name", value: "Untitled", confirmLabel: "Create",
        });
        const name = await asked.result;
        if (name === null) return;              // cancelled
        await project.newProject(name);
      } else if (id === "save-file") {
        const url = project.downloadUrl();
        if (!url) return;
        // Navigation, not a fetch: the server's own Content-Disposition is
        // what makes this a save-as instead of a page full of JSON.
        const a = document.createElement("a");
        a.href = url;
        a.download = "";
        document.body.appendChild(a);
        a.click();
        a.remove();
      } else if (id === "load-file") {
        loadProjectFile.click();
      } else if (id !== (project.project || {}).id) {
        await project.open(id);
      } else {
        return;                                 // already open
      }
      // Nothing to put back here: the store says when a project was opened, and
      // whatever follows it is wired to that.
    },
    onPipeline: () => openPipeline({
      load, describe, check, search,
      onApply: (next) => {
        slots = next;
        // The boxes on the main window are for inputs of THESE slots. A slot
        // that was removed takes its box with it, and a value saved in the
        // window is what its box now shows -- otherwise the two windows hold
        // different text for one input and the run uses whichever was sent.
        if (prompts) {
          // sync() rebuilds every control from the PIPELINE's values, which is
          // right for the prompt and wrong for the project's own settings: they
          // outrank the pipeline's defaults and have to be put back.
          Promise.resolve(prompts.sync(next)).then(() => { showGroups(); syncVideo(); });
        }
      },
    }),
  });
  const session = wire({ run, page, check, id, queuedFor, finishedFor,
                         slots: () => slots, values: allValues,
                         inputs: () => (prompts ? prompts.overrides() : {}),
                         // Read by run.start() at the moment IT queues -- not
                         // at Generate-click time, which is what onGenerate
                         // sets ranFor/ranForProject from.
                         extra: () => (ranForProject && ranFor
                           ? { funpack_scene_id: ranFor, funpack_project_id: ranForProject } : null),
                         // The other way a run reaches this page: reattaching
                         // to one after a reload, where the click that started
                         // it happened on a page that is gone. Called BEFORE
                         // the run is adopted -- an already-finished run can go
                         // straight to DONE inside adopt() itself, and setting
                         // these from session.ready instead would run one tick
                         // too late, after that DONE already found nothing to
                         // attach the result to.
                         onAdopt: (sceneId, projectId) => { ranFor = sceneId; ranForProject = projectId; } });

  // What a run produced, on the scene it was started from -- which is what puts
  // it on the timeline and what makes it still there after a reload.
  run.subscribe((state) => {
    if (state.phase !== DONE || !ranFor || !state.images.length) return;
    if (ranForProject) project.setResultFor(ranForProject, ranFor, viewUrl(state.images[state.images.length - 1]));
    else project.setResult(ranFor, viewUrl(state.images[state.images.length - 1]));
    ranFor = null;
    ranForProject = null;
  });

  // Undo from the keyboard, which is where anyone will reach for it first.
  //
  // Not inside a field: a text box has its own undo stack and taking that over
  // would mean one keystroke un-typing a whole paragraph instead of a word.
  window.addEventListener("keydown", (event) => {
    if (!(event.metaKey || event.ctrlKey) || event.key.toLowerCase() !== "z") return;
    const on = event.target;
    const typing = on && (on.tagName === "INPUT" || on.tagName === "TEXTAREA" || on.isContentEditable);
    if (typing) return;
    event.preventDefault();
    if (event.shiftKey) project.redo();
    else project.undo();
  });

  // Saved on the way out. A debounce that has not fired yet is work the user
  // did and cannot see anywhere, and a reload is exactly when it is lost.
  window.addEventListener("pagehide", () => {
    // Blur first: a control commits on blur, so text still being typed is not
    // in the project yet -- and this is exactly the moment it would be lost.
    if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
    project.flush();
  });

  let manifest;
  try {
    manifest = await fetchManifest();
  } catch (err) {
    // The one failure the user must see: with no manifest there is no app, so
    // silence here would be an empty window with no explanation.
    root.replaceChildren(composer.emptyState.default({
      icon: "▲",
      title: "Could not reach FunPack",
      hint: `${err.message}. Is ComfyUI running?`,
    }).node);
    return;
  }

  const { mounted, hidden } = await mountAll(manifest);

  // After the modules, because a region has to exist before anything can be put
  // in it, and a module may be sharing the region a role names.
  try {
    prompts = await createPrompts((await load()).slots, {
      // Controls come from the node's own description: a combo gets the node's
      // real choices, a number its real bounds.
      describe,
      // Typing in the box writes to the scene it belongs to. Without this the
      // prompt is a value the run uses and the project never hears about, so a
      // reload shows a timeline whose scenes are all empty.
      onChange: (field) => {
        // Two places take a value now, and which one is decided by where the
        // pipeline asked for the control -- not by what it is called.
        if (field && field.at === "project.video") {
          project.setVideo(field.input, field.control.value);
          return;
        }
        const box = prompts && prompts.at("generation.prompt");
        if (box && project.selectedId) project.setText(project.selectedId, box.value);
      },
    });
  } catch (err) {
    // Not fatal and not silent: the app still runs on the server's own
    // defaults, and an empty prompt panel with no explanation is the failure
    // this project keeps finding.
    console.warn(`[FunPack] the pipeline could not be read, so nothing it puts on the main window is here: ${err.message}`);
  }

  // Everything that mounts has now had its turn, so a region still holding its
  // stand-in is a region nothing wanted.
  settle();
  showGroups();

  // What the app can be asked to do, offered by the part that owns each one.
  // The wheel shows whatever is here; nothing has to tell it about a new one.
  offerAction({ id: "generate", icon: "▶", label: "Generate", run: () => session.generate() });
  offerAction({ id: "cancel", icon: "■", label: "Cancel", run: () => run.cancel() });
  offerAction({ id: "constructor", icon: "✎", label: "Constructor",
                run: () => page.constructor.open() });
  offerAction({ id: "scene", icon: "＋", label: "Add scene",
                run: () => { project.addScene(); showScene(project.selected); } });
  offerAction({ id: "pipeline", icon: "⚙", label: "Models",
                run: () => page.menubar.settings.node.click() });
  offerAction({ id: "assets", icon: "▤", label: "Assets",
                run: () => page.workspace.toggle("left") });
  const wheel = createWheel();

  // After the prompt exists, so the first scene's text has somewhere to go.
  try {
    const fresh = await project.start();
    // Nothing to come back to: offer the way in. Dismissing is a real answer --
    // the project is already made, and it carries on with one empty scene.
    if (!(fresh.scenes || []).length) {
      openWizard({ onPick: (choice) => {
        const wanted = choice === "scenes" ? 3 : 1;
        for (let i = 0; i < wanted; i += 1) project.addScene();
        showScene(project.selected);
      } });
      if (!project.scenes.length) project.addScene();
    }

    timeline = createTimeline({ project, onSelect: showScene });
    page.timelineBody.set([page.transport.warning, timeline]);

    showScene(project.selected);
  } catch (err) {
    console.warn(`[FunPack] no project: ${err.message}`);
    page.transport.say(`The project could not be opened: ${err.message}`);
  }

  const failed = manifest.failed || [];

  console.info(`[FunPack] ${mounted.length} module(s) mounted`,
    hidden.length ? `· ${hidden.length} hidden` : "",
    failed.length ? `· ${failed.length} failed to load` : "");
  for (const { id, why } of hidden) console.warn(`[FunPack] ${id} is hidden: ${why}`);
  // These never reached the manifest, so no panel could be missing "in a way
  // the user notices" -- which is exactly why they have to be said. A module
  // that failed to import looks identical to one nobody installed.
  for (const { where, why } of failed) console.warn(`[FunPack] ${where} did not load: ${why}`);

  window.FunPack = {
    manifest, values: allValues, failed, hidden, run, bin: page.bin, project, wheel,
    viewer: page.viewer,
    prompts: () => (prompts ? prompts.overrides() : {}),
    mounted: mounted.map((m) => m.id),
  };
}

start();
