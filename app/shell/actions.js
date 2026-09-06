// Everything the app can be asked to do, by name.
//
// Announced, not listed: whoever builds a control offers it here, and whoever
// shows a list of them -- the wheel today, a command palette or a keymap later
// -- reads what is there. A part of the app that failed to load offers nothing
// and is simply not among them, which is the same rule regions and modules
// live by.

const actions = new Map();

/**
 * offerAction({ id, label, icon, run }) -> a function that takes it away again.
 *
 * The id is the address. Offering the same one twice replaces it, because the
 * alternative is two entries that look identical and do different things.
 */
export function offerAction({ id, label, icon, run } = {}) {
  if (!id || typeof run !== "function") {
    throw new TypeError("An action needs an id and something to run.");
  }
  actions.set(id, { id, label: label || id, icon: icon || "•", run });
  return () => actions.delete(id);
}

/** In the order they were offered, which is the order they were built in. */
export const offered = () => [...actions.values()];

/** Run one by name. Unknown is not an error: whatever offered it may be gone. */
export function run(id) {
  const action = actions.get(id);
  if (!action) return false;
  try {
    action.run();
  } catch (err) {
    // One broken action must not take down the thing that listed it.
    console.warn(`[FunPack] the "${id}" action failed: ${err.message}`);
    return false;
  }
  return true;
}

export function _reset() { actions.clear(); }
