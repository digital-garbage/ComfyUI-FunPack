// The live value of every setting, keyed exactly as the declaration keys them.
//
// This is the only thing in the app that reads the UI. A module never does: it
// declares what it needs and is handed values, so the same module works with a
// panel on screen and with no browser at all.

const store = new Map();            // moduleId -> { key: value }
const listeners = new Set();

export function seed(moduleId, defaults) {
  store.set(moduleId, { ...defaults });
}

export function set(moduleId, key, value) {
  const current = store.get(moduleId) || {};
  current[key] = value;
  store.set(moduleId, current);
  for (const fn of listeners) {
    try { fn(moduleId, key, value); } catch { /* a listener must not break the edit */ }
  }
}

export const valuesOf = (moduleId) => ({ ...(store.get(moduleId) || {}) });

/** Everything, as it would be sent with a generation request. */
export const all = () => Object.fromEntries([...store].map(([id, v]) => [id, { ...v }]));

export function onChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function _reset() {
  store.clear();
  listeners.clear();
}
