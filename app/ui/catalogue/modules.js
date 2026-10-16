// What is installed, what announced itself, and what did not make it.
//
// The plan called this a required tool rather than a nicety, and it was right:
// a module that fails to import is ABSENT, which is the correct behaviour and
// also indistinguishable from one nobody installed. Somewhere has to say the
// difference out loud, and it must not be the main app, where an absent feature
// is supposed to be simply absent.

import { composer } from "../composer/composer.js";
import { fetchManifest } from "../shell/manifest.js";
import { build } from "../shell/layout.js";
import { offered, _reset } from "../shell/mounts.js";

const root = document.querySelector("#app");

function rows(items) {
  return composer.list.rows({ items, reorder: false });
}

async function start() {
  let manifest;
  try {
    manifest = await fetchManifest();
  } catch (err) {
    root.replaceChildren(composer.emptyState.default({
      icon: "▲", title: "Could not reach FunPack", hint: err.message,
    }).node);
    return;
  }

  // Build the real shell off-screen purely to learn which mount points exist.
  // Asking the layout is the only honest answer: nothing else knows.
  _reset();
  build(document.createElement("div"));
  const available = new Set(offered());

  const modules = manifest.modules || [];
  const failed = manifest.failed || [];
  const unclaimed = modules.filter(
    (m) => Object.keys(m.settings || {}).length && m.mount && !available.has(m.mount));

  const sections = [];

  sections.push(composal("Loaded", `${modules.length} module(s) announced themselves.`,
    modules.map((m) => ({
      id: m.id,
      label: m.id,
      hint: `${m.stage} · ${m.status}` +
        (m.mount ? ` · ${m.mount}` : " · no panel") +
        (Object.keys(m.settings || {}).length ? ` · ${Object.keys(m.settings).length} setting(s)` : ""),
    }))));

  if (unclaimed.length) {
    sections.push(composal(
      "Asking for a mount point nothing offers",
      "These loaded correctly and their panels will not appear anywhere. " +
      `Offered: ${[...available].join(", ")}.`,
      unclaimed.map((m) => ({ id: m.id, label: m.id, hint: `wants "${m.mount}"` }))));
  }

  if (failed.length) {
    sections.push(composal("Did not load", "Absent from the app, and from the graph.",
      failed.map((f, i) => ({ id: `f${i}`, label: f.where, hint: f.why }))));
  }

  if (manifest.incompatible?.length) {
    sections.push(composal("Not for this model", "Filtered out by what the model is.",
      manifest.incompatible.map((m) => ({
        id: m.id, label: m.id, hint: `needs ${m.requires.join(", ")}` }))));
  }

  root.replaceChildren(...sections.map((s) => s.node));
}

function composal(title, hint, items) {
  return composer.panel.default({
    title,
    body: composer.group.default({
      hint,
      rows: items.length ? [rows(items)] : [composer.hint.default({ text: "None." })],
    }),
  });
}

start();
