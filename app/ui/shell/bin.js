// What the media bin's files are called, as last seen by whoever looked: one shared memory so a rename shows everywhere.
export const names = new Map();
/** What each picture means when used as a reference ("<Subject 1> is the woman in <Picture 1>…"), as last seen. */
export const subjects = new Map();
export const bin = { version: 0 };

/** Remember a listing (replacing what was known). */
export function learn(list) {
  names.clear();
  subjects.clear();
  list.forEach((m) => { names.set(m.id, m.name); if (m.subject) subjects.set(m.id, m.subject); });
  bin.version += 1;
}
