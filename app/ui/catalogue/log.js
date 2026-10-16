// The log, where a person can read it.
//
// The terminal is where this ends up by default, buried among ComfyUI's own
// output. The whole reason for a severity and a source is that a line can be
// found later, so there has to be somewhere to look.

import { composer } from "../composer/composer.js";

const root = document.querySelector("#app");

// Severity -> the chip tone that already exists in the kit. Info is deliberately
// quiet: most lines are Info, and a wall of coloured chips says nothing.
const TONE = { Info: "neutral", Alert: "info", Warning: "warn", Critical: "danger" };

let filter = null;

async function load() {
  const url = "/funpack/api/log/funpack" + (filter ? `?level=${encodeURIComponent(filter)}` : "");
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} answered ${response.status}`);
  return response.json();
}

function render(data) {
  const counts = {};
  for (const level of data.levels) counts[level] = 0;
  for (const record of data.records) counts[record.level] = (counts[record.level] || 0) + 1;

  const filters = composer.segmented.md({
    label: "Show",
    value: filter || "all",
    options: [{ value: "all", label: "Everything" },
      ...data.levels.map((l) => ({ value: l, label: `${l} (${counts[l] || 0})` }))],
    onChange: (value) => { filter = value === "all" ? null : value; start(); },
  });

  // Newest last, like a terminal: the eye expects the end to be the present.
  const items = data.records.map((record, i) => ({
    id: `r${i}`,
    label: record.message,
    hint: `${record.level} · ${record.source || "FunPack"}`,
  }));

  root.replaceChildren(composer.panel.default({
    title: "Log",
    body: composer.group.default({
      rows: [
        filters,
        items.length
          ? composer.list.rows({ items, reorder: false })
          : composer.emptyState.default({
              icon: "○", title: "Nothing logged yet",
              hint: filter ? `No ${filter} messages.` : "Run a generation and come back.",
            }),
      ],
    }),
  }).node);
}

async function start() {
  try {
    render(await load());
  } catch (err) {
    root.replaceChildren(composer.emptyState.default({
      icon: "▲", title: "Could not read the log", hint: err.message,
    }).node);
  }
}

start();
