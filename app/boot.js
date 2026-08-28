// Entry point.
//
// The whole of what the app does at startup: build the regions, ask the server
// what exists, and mount whatever announced itself. Nothing here names a module.

import { build } from "./shell/layout.js";
import { fetchManifest } from "./shell/manifest.js";
import { mountAll } from "./shell/panels.js";
import { all as allValues } from "./shell/values.js";
import { composer } from "./composer/composer.js";

const root = document.querySelector("#app");

async function start() {
  build(root);

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
  console.info(`[FunPack] ${mounted.length} module(s) mounted`,
    hidden.length ? `· ${hidden.length} hidden` : "");
  for (const { id, why } of hidden) console.warn(`[FunPack] ${id} is hidden: ${why}`);

  // Handy while there is no generate button to send them anywhere.
  window.FunPack = { manifest, values: allValues, mounted: mounted.map((m) => m.id), hidden };
}

start();
