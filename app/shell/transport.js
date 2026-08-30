// Generate, and everything that happens after it.
//
// One row that owns the whole lifecycle, because the alternative is a button
// here and a progress bar there and a status somewhere else, each with its own
// idea of what is happening. This reads ONE state object and draws it -- so
// "queued" cannot appear next to a finished result, whatever order the messages
// arrive in.
//
// Warnings live here too, beside Generate, rather than in the panel that would
// have caused them: a setting that will do nothing has to be said where the run
// is started, not where it was switched on.

import { composer } from "../composer/composer.js";
import { IDLE, QUEUED, RUNNING, DONE, FAILED, CANCELLED } from "./run.js";

const WORKING = new Set([QUEUED, RUNNING]);

export function createTransport({ onGenerate, onCancel } = {}) {
  const generate = composer.button.lg({
    label: "Generate", tone: "primary",
    onClick: () => { if (onGenerate) onGenerate(); },
  });
  const cancel = composer.button.md({
    label: "Cancel", tone: "ghost",
    onClick: () => { if (onCancel) onCancel(); },
  });
  const progress = composer.progress.bar({ value: 0, max: 100, label: "Generating" });
  const status = composer.text.sm({ text: "Ready" });

  const bar = composer.actionBar.sticky({
    actions: [status, progress, cancel, generate],
  });

  // Something to say that is not about a run in progress: why the last attempt
  // did not become one. It has to survive the next draw(), because a refused
  // attempt leaves the run exactly where it was -- idle -- and drawing an idle
  // run says "Ready", which wiped the reason a moment after it appeared.
  let note = null;

  // Held for a reason that is not a run: the page is still working out whether
  // it already has one. Kept as a flag rather than left to whoever disabled the
  // button, because draw() runs on every state change and each of those would
  // otherwise hand the button back -- which is exactly what a subscription
  // delivering the current state did, one line after the button was disabled.
  let held = false;

  // Hidden by attribute rather than removed: the row's shape then does not jump
  // as a run starts and finishes, and there is one node to find in a test.
  const hide = (handle, yes) => {
    if (yes) handle.node.setAttribute("hidden", "");
    else handle.node.removeAttribute("hidden");
  };
  hide(progress, true);
  hide(cancel, true);

  function draw(state) {
    // Anything other than idle means a run answered for itself, and what it
    // says outranks a note about an attempt that never started.
    if (state.phase !== IDLE) note = null;
    const working = WORKING.has(state.phase);
    generate.setDisabled(working || held);
    generate.setBusy(working);
    hide(cancel, !working);

    const measured = state.progress && state.progress.max;
    hide(progress, !(working && measured));
    if (measured) {
      progress.node.setAttribute("aria-valuemax", String(state.progress.max));
      progress.setValue(state.progress.value);
    }

    status.setText(note || describe(state));
  }

  /** Say why nothing was started, or -- with null -- stop saying it. Cleared by
   *  the next run that says otherwise. */
  function say(text, state) {
    note = text || null;
    status.setText(note || describe(state || { phase: IDLE, images: [] }));
  }

  /** Stop offering Generate for a reason of the page's own, and say why. */
  function hold(reason) {
    held = Boolean(reason);
    if (reason) say(reason);
    generate.setDisabled(true);
  }

  /** Offer it again. The run's own state decides from here. */
  function release(state) {
    held = false;
    say(null, state);
    draw(state);
  }

  return { node: bar.node, draw, say, hold, release, generate, cancel, progress, status };
}

/** One line saying where the run is. The only place these words are decided. */
export function describe(state) {
  switch (state.phase) {
    case QUEUED:
      return "Queued";
    case RUNNING: {
      const where = state.node ? `node ${state.node}` : "working";
      if (state.progress && state.progress.max) {
        return `${where} — ${state.progress.value} of ${state.progress.max}`;
      }
      return where.charAt(0).toUpperCase() + where.slice(1);
    }
    case DONE:
      // The count, because a graph that saved nothing finished successfully and
      // produced no picture, and those must not read the same.
      return state.images.length
        ? `Done — ${state.images.length} result${state.images.length === 1 ? "" : "s"}`
        : "Done, but nothing was saved";
    case FAILED: {
      const where = state.error && state.error.node ? `${state.error.node}: ` : "";
      return `${where}${(state.error && state.error.message) || "the run failed"}`;
    }
    case CANCELLED:
      return "Cancelled";
    default:
      return "Ready";
  }
}
