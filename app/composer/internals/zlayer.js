// The only place a z-index is written.
//
// v4 ended up with 9000, 9999, 10001, 12000, 99999, 200000 and two cross-file
// collisions, because every overlay picked its own number in isolation. Here the
// ladder is ordered once and each rung owns a block of slots, so peers stack
// within a rung and can never cross into the next one.
//
// A rung is an ORDERED LIST and z is derived from position, rather than each
// claim owning a fixed offset. That is what makes raise() and release() always
// correct: with fixed offsets, a near-full rung has no free slot above the top,
// and "move to the front" has nowhere to go.

// Order is the contract; the numbers are derived. Autocomplete sits ABOVE modal
// on purpose: it opens from a field that is often inside one.
export const LADDER = [
  "local",
  "dropline",
  "sticky",
  "menubar",
  "popover",
  "backdrop",
  "wizard",
  "modal",
  "floatingWindow",
  "toast",
  "autocomplete",
  "tour",
];

export const SLOTS = 100;              // peers per rung

const stacks = new Map();              // rung -> entries, bottom to top

export function baseOf(rung) {
  const i = LADDER.indexOf(rung);
  if (i === -1) {
    throw new RangeError(`Unknown z rung "${rung}". Known: ${LADDER.join(", ")}.`);
  }
  return (i + 1) * SLOTS;
}

/**
 * Take a z-index in `rung`. `onChange(z)` fires whenever this claim's z moves --
 * which happens when a PEER raises or releases, not only when this one does.
 * Without it a window that was above becomes stale in the DOM the moment
 * another is brought to the front.
 */
export function claim(rung, onChange) {
  const start = baseOf(rung);
  if (!stacks.has(rung)) stacks.set(rung, []);
  const list = stacks.get(rung);

  if (list.length >= SLOTS) {
    // Running a rung dry means something opens overlays and never releases
    // them; quietly reusing a slot would hide the leak and stack two things at
    // one z.
    throw new RangeError(`z rung "${rung}" is full (${SLOTS} live claims). Something is not releasing.`);
  }

  const entry = { onChange };
  list.push(entry);
  let live = true;
  let lastZ = start + list.indexOf(entry);

  const handle = {
    get z() {
      if (!live) return lastZ;
      lastZ = start + list.indexOf(entry);
      return lastZ;
    },
    get rung() { return rung; },
    get live() { return live; },
    raise() {
      if (!live) return handle;
      const i = list.indexOf(entry);
      if (i === -1 || i === list.length - 1) return handle;
      list.splice(i, 1);
      list.push(entry);
      notify(list, start);
      return handle;
    },
    release() {
      if (!live) return;
      lastZ = handle.z;
      const i = list.indexOf(entry);
      if (i !== -1) list.splice(i, 1);
      live = false;
      notify(list, start);
    },
  };
  return handle;
}

function notify(list, start) {
  list.forEach((entry, i) => { if (entry.onChange) entry.onChange(start + i); });
}

/** Live claims in a rung, bottom to top. Diagnostics and tests. */
export const liveCount = (rung) => (stacks.get(rung) || []).length;

export function _resetLayers() {
  stacks.clear();
}
