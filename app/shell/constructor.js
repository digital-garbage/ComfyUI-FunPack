// The Constructor: where a run is written, in a window of its own.
//
// v4 called this the Composer. The name is taken here -- `app/composer` is the
// element kit -- and two things called Composer in one codebase is one thing
// nobody can search for.
//
// The prompt is not a panel on the main window. It is the longest thing a
// person writes, it is edited in bursts and read rarely, and giving it a
// permanent third of the centre column cost the timeline the room it needs. So
// it opens, is written in, and closes.
//
// The HOST is built once, at startup, and outlives every opening. Two reasons,
// and both were faults before this file existed: a mount point has to exist
// when modules mount, which is long before anyone opens a window; and a window
// that rebuilt its contents on each open would throw away whatever was typed
// and not saved.

import { composer } from "../composer/composer.js";

export function createConstructor({ title = "Constructor", onChange } = {}) {
  const empty = composer.emptyState.default({
    icon: "✎",
    title: "Nothing to write yet",
    hint: "The pipeline decides which of its inputs are written here.",
  });

  // Two groups, by purpose: what this run SAYS, and what it is generated AT.
  // Grouping them is the whole reason they share a window -- the prompt and the
  // size of what it produces are settled in one sitting.
  const written = composer.region.stack({ gap: "sm", label: "Prompt", children: [empty] });
  // Labelled, because with five fields in a column the two purposes are not
  // legible from the field names alone. Hidden by whoever fills it when the
  // pipeline offers nothing to put here -- a heading over nothing is a lie.
  const videoLabel = composer.label.section({ text: "Video settings" });
  const video = composer.region.stack({ gap: "sm", label: "Video settings",
                                        children: [videoLabel] });

  // The host is a plain region: it is handed to the modal as a body and handed
  // back when the modal closes, which is why the modal never owns it.
  const host = composer.region.stack({ gap: "md", label: "Constructor", fill: true,
                                       children: [written, video] });

  let window_ = null;

  function open() {
    if (window_) return window_;                 // one window onto one prompt
    window_ = composer.modal.generic({
      title,
      subtitle: "What this run is made of.",
      size: "lg",
      body: host,
      actions: [composer.button.md({ label: "Done", tone: "primary",
                                     onClick: () => window_ && window_.close("done") })],
      onClose: () => { window_ = null; if (onChange) onChange(); },
    });
    if (onChange) onChange();
    return window_;
  }

  return {
    host, empty, open,
    /** The two mount points, offered by the shell under these names. */
    written, video,
    /** Open, and mounted. Tests and the shell both ask. */
    get isOpen() { return Boolean(window_); },
    close() { if (window_) window_.close("close"); },
  };
}
