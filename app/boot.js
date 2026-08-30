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
import { createRun } from "./shell/run.js";
import { clientId, connect, queuedFor, finishedFor } from "./shell/client.js";
import { wire } from "./shell/session.js";
import { check, load, describe, search } from "./shell/pipeline.js";
import { open as openPipeline } from "./shell/pipeline_window.js";
import { createPrompts } from "./shell/prompt.js";

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

  const page = build(root, {
    onGenerate: () => session.generate(),
    onCancel: () => run.cancel(),
    onPipeline: () => openPipeline({
      load, describe, check, search,
      onApply: (next) => {
        slots = next;
        // The boxes on the main window are for inputs of THESE slots. A slot
        // that was removed takes its box with it, and a value saved in the
        // window is what its box now shows -- otherwise the two windows hold
        // different text for one input and the run uses whichever was sent.
        if (prompts) prompts.sync(next);
      },
    }),
  });
  const session = wire({ run, page, check, id, queuedFor, finishedFor,
                         slots: () => slots, values: allValues,
                         inputs: () => (prompts ? prompts.overrides() : {}) });

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
    prompts = createPrompts((await load()).slots);
  } catch (err) {
    // Not fatal and not silent: the app still runs on the server's own
    // defaults, and an empty prompt panel with no explanation is the failure
    // this project keeps finding.
    console.warn(`[FunPack] the pipeline could not be read, so nothing it puts on the main window is here: ${err.message}`);
  }

  // Everything that mounts has now had its turn, so a region still holding its
  // stand-in is a region nothing wanted.
  settle();
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
    manifest, values: allValues, failed, hidden, run, bin: page.bin,
    prompts: () => (prompts ? prompts.overrides() : {}),
    mounted: mounted.map((m) => m.id),
  };
}

start();
