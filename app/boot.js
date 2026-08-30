// Entry point.
//
// The whole of what the app does at startup: build the regions, ask the server
// what exists, and mount whatever announced itself. Nothing here names a module.

import { build } from "./shell/layout.js";
import { fetchManifest } from "./shell/manifest.js";
import { mountAll } from "./shell/panels.js";
import { all as allValues } from "./shell/values.js";
import { composer } from "./composer/composer.js";
import { createRun } from "./shell/run.js";
import { clientId, connect, queuedFor, finishedFor } from "./shell/client.js";
import { wire } from "./shell/session.js";
import { check, load, describe, search } from "./shell/pipeline.js";
import { open as openPipeline } from "./shell/pipeline_window.js";

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

  const page = build(root, {
    onGenerate: () => session.generate(),
    onCancel: () => run.cancel(),
    onPipeline: () => openPipeline({
      load, describe, check, search,
      onApply: (next) => { slots = next; },
    }),
  });
  const session = wire({ run, page, check, id, queuedFor, finishedFor,
                         slots: () => slots });

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
    mounted: mounted.map((m) => m.id),
  };
}

start();
