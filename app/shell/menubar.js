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

export function createMenubar({ workspace, onPipeline, onProject, projects = () => [],
                                current = () => null, edits = null, onUpdates, onPacks, onLog, onTemp,
                                theme = services.theme } = {}) {
  const connection = createConnection();

  const file = menu("File", () => [
    { id: "new", label: "New project…" },
    ...(projects().length ? [{ separator: true }] : []),
    // A tick rather than leaving the current one out: which project is open is
    // a fact about the app, and a menu that hides it makes you press one to find out.
    ...projects().map((p) => ({ id: p.id, label: p.name, icon: p.id === current() ? "✓" : " " })),
  ], (id) => { if (onProject) onProject(id); });

  // Built on each press, so "Undo" is offered only when there is something to
  // undo -- a menu that always offers it teaches people it does nothing.
  const edit = edits && menu("Edit", () => [
    { id: "undo", label: "Undo", hint: "⌘Z", disabled: !edits.canUndo() },
    { id: "redo", label: "Redo", hint: "⇧⌘Z", disabled: !edits.canRedo() },
    { separator: true },
    { id: "scene", label: "Add scene" },
    { id: "remove", label: "Delete scene" },
    { separator: true },
    { id: "earlier", label: "Move clip left" },
    { id: "later", label: "Move clip right" },
  ], (id) => edits.run(id));

  const view = menu("View", () => [
    { id: "left", label: `${workspace && workspace.isOpen("left") ? "Hide" : "Show"} Assets` },
    { id: "right", label: `${workspace && workspace.isOpen("right") ? "Hide" : "Show"} Properties` },
    { separator: true },
    { id: "reset", label: "Reset layout" },
    { separator: true },
    ...THEMES.map((t) => ({
      ...t,
      // A tick rather than a disabled row: which one is on is a fact about the
      // app, and a menu that hides it makes the user press one to find out.
      icon: theme.get() === t.id ? "✓" : " ",
    })),
  ], (id) => {
    if (id === "left" || id === "right") workspace.toggle(id);
    // Cheap insurance: a person who has closed both panels and forgotten which
    // control brings them back has no way home otherwise.
    else if (id === "reset") { workspace.open("left"); workspace.open("right"); }
    else theme.set(id);
  });

  const settings = menu("Settings", () => [
    { id: "pipeline", label: "Models and pipeline…" },
    { separator: true },
    { id: "updates", label: "Updates…" },
    { id: "packs", label: "Node packs…" },
    { separator: true },
    { id: "log", label: "ComfyUI log…" },
    { id: "temp", label: "Temp files…" },
  ], (id) => {
    if (id === "pipeline" && onPipeline) onPipeline();
    else if (id === "updates" && onUpdates) onUpdates();
    else if (id === "packs" && onPacks) onPacks();
    else if (id === "log" && onLog) onLog();
    else if (id === "temp" && onTemp) onTemp();
  });

  const bar = composer.toolbar.default({
    label: "FunPack",
    items: [composer.brand.default({ name: "FunPack" }),
            ...[file, edit, view, settings].filter(Boolean)],
    trailing: connection.items,
  });

  return { node: bar.node, bar, connection, file, edit, view, settings };
}
