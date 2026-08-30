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

export function createConstructor({ title = "Constructor" } = {}) {
  const empty = composer.emptyState.default({
    icon: "✎",
    title: "Nothing to write yet",
    hint: "The pipeline decides which of its inputs are written here.",
  });

  // The host is a plain region: it is handed to the modal as a body and handed
  // back when the modal closes, which is why the modal never owns it.
  const host = composer.region.stack({ gap: "sm", label: "Constructor", fill: true,
                                       children: [empty] });

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
      onClose: () => { window_ = null; },
    });
    return window_;
  }

  return {
    host, empty, open,
    /** Open, and mounted. Tests and the shell both ask. */
    get isOpen() { return Boolean(window_); },
    close() { if (window_) window_.close("close"); },
  };
}
