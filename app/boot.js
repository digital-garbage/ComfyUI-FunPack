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
import { clientId, connect } from "./shell/client.js";
import { check } from "./shell/pipeline.js";

const root = document.querySelector("#app");

async function start() {
  const run = createRun({ clientId: clientId(), connect });

  const page = build(root, {
    async onGenerate() {
      // Asked for, not assembled: core builds the graph from the pipeline's
      // slots, and a refusal comes back as a reason rather than as a failure
      // several stages later.
      let plan;
      try {
        plan = await check({});
      } catch (err) {
        page.transport.status.setText(`The pipeline could not be read: ${err.message}`);
        return;
      }
      const stopping = [...plan.refused, ...plan.incomplete];
      if (!plan.queueable || !plan.prompt) {
        page.transport.status.setText(stopping[0] || "This pipeline is not ready to run.");
        return;
      }
      try {
        await run.start(plan.prompt);
      } catch { /* the run's own state carries the reason, and draw() shows it */ }
    },
    onCancel: () => run.cancel(),
  });

  // One subscription draws everything the run affects. Two would be two places
  // that can disagree about what is happening.
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
