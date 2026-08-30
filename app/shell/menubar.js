// The menu bar: who this is, what is not a zone, and whether the server is
// there.
//
// v4's arrangement, and the reason an app with only zones reads as unfinished:
// a zone head holds what acts on THAT zone, so anything that acts on the app --
// a window, a preference, the state of the connection -- has nowhere to live
// unless there is a bar that belongs to the app itself.
//
// Only menus with something real in them exist. An empty File menu is a promise
// the app does not keep, and this project's rule for an inapplicable feature is
// that it is absent rather than greyed.

import { composer } from "../composer/composer.js";
import { LIVE, LOST } from "./run.js";
import { services } from "./services.js";

const THEMES = [
  { id: "dark", label: "Dark" },
  { id: "light", label: "Light" },
  { id: "auto", label: "Auto", hint: "Follows the system" },
];

/** A dot and a word for whether ComfyUI is still on the other end. */
export function createConnection() {
  const dot = composer.dot.default({ tone: "neutral", label: "ComfyUI" });
  const text = composer.text.xs({ text: "Connecting" });

  const SAID = {
    [LIVE]: { tone: "good", words: "ComfyUI live" },
    [LOST]: { tone: "danger", words: "Reconnecting" },
  };

  return {
    /** Two handles, not a wrapper: the bar it goes in already lays out a row. */
    items: [dot, text],
    dot, text,
    /** Reads the same run state everything else does. */
    draw(state) {
      const said = SAID[state && state.connection] || { tone: "neutral", words: "Connecting" };
      dot.node.className = `cx-dot cx-dot-${said.tone}`;
      dot.node.setAttribute("aria-label", said.words);
      text.setText(said.words);
    },
  };
}

/**
 * menu(label, items, onPick) -> a button that opens them.
 *
 * The menu is built on each press rather than kept: what is in it depends on
 * what is true when it is opened -- which panel is showing, which theme is on
 * -- and a menu built once is a menu that goes stale the first time anything
 * changes underneath it.
 */
function menu(label, itemsOf, onPick) {
  let open = null;
  const button = composer.button.sm({
    label, tone: "ghost",
    onClick: () => {
      if (open) { open.close("toggle"); open = null; return; }
      open = composer.menu.dropdown({
        anchor: button, side: "bottom", align: "start",
        items: itemsOf(),
        onPick: (id) => { open = null; onPick(id); },
        onClose: () => { open = null; },
      });
    },
  });
  return button;
}

export function createMenubar({ workspace, onPipeline, theme = services.theme } = {}) {
  const connection = createConnection();

  const view = menu("View", () => [
    { id: "left", label: `${workspace && workspace.isOpen("left") ? "Hide" : "Show"} Assets` },
    { id: "right", label: `${workspace && workspace.isOpen("right") ? "Hide" : "Show"} Properties` },
    { separator: true },
    ...THEMES.map((t) => ({
      ...t,
      // A tick rather than a disabled row: which one is on is a fact about the
      // app, and a menu that hides it makes the user press one to find out.
      icon: theme.get() === t.id ? "✓" : " ",
    })),
  ], (id) => {
    if (id === "left" || id === "right") workspace.toggle(id);
    else theme.set(id);
  });

  const settings = menu("Settings", () => [
    { id: "pipeline", label: "Models and pipeline…" },
  ], (id) => { if (id === "pipeline" && onPipeline) onPipeline(); });

  const bar = composer.toolbar.default({
    label: "FunPack",
    items: [composer.brand.default({ name: "FunPack" }), view, settings],
    trailing: connection.items,
  });

  return { node: bar.node, bar, connection, view, settings };
}
