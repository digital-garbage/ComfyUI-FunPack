// What the media bin's files are called, as last seen by whoever looked: one shared memory so a rename shows everywhere.
export const names = new Map();
export const bin = { version: 0 };

/** Remember a listing (replacing what was known). */
export function learn(list) {
  names.clear();
  list.forEach((m) => names.set(m.id, m.name));
  bin.version += 1;
}
