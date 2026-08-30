// Entry point.
//
// The whole of what the app does at startup: build the regions, ask the server
// what exists, and mount whatever announced itself. Nothing here names a module.

import { build } from "./shell/layout.js";
import { fetchManifest } from "./shell/manifest.js";
import { mountAll } from "./shell/panels.js";
import { all as allValues } from "./shell/values.js";
import { composer } from "./composer/composer.js";
import { createRun, viewUrl } from "./shell/run.js";
import { clientId, connect, runningFor, finishedFor } from "./shell/client.js";
import { wire } from "./shell/session.js";
import { check } from "./shell/pipeline.js";

const root = document.querySelector("#app");

async function start() {
  const id = clientId();
  const run = createRun({ clientId: id, connect });

  const page = build(root, {
    onGenerate: () => session.generate(),
    onCancel: () => run.cancel(),
  });
  const session = wire({ run, page, check, id, runningFor, finishedFor });

  run.subscribe((state) => {
    page.transport.draw(state);
    const last = state.images[state.images.length - 1];
    if (last) page.viewer.setSource(viewUrl(last), kindOf(last));
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
    manifest, values: allValues, failed, hidden, run,
    mounted: mounted.map((m) => m.id),
  };
}

/** Video and image results arrive the same way and cannot be shown the same way. */
function kindOf(image) {
  return /\.(mp4|webm|mov|mkv)$/i.test(image.filename || "") ? "video" : "image";
}

start();
