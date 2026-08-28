// The only place a z-index is written.
//
// v4 ended up with 9000, 9999, 10001, 12000, 99999, 200000 and two cross-file
// collisions, because every overlay picked its own number in isolation. Here the
// ladder is ordered once and each rung owns a block of slots, so peers stack
// within a rung and can never cross into the next one.

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
const base = (rung) => (LADDER.indexOf(rung) + 1) * SLOTS;

const used = new Map();                // rung -> Set(offset)

export function baseOf(rung) {
  if (!LADDER.includes(rung)) {
    throw new RangeError(`Unknown z rung "${rung}". Known: ${LADDER.join(", ")}.`);
  }
  return base(rung);
}

/**
 * Take a z-index in `rung`. Returns { z, raise(), release() }.
 * `raise()` moves this claim above its current peers -- click-to-front, without
 * anyone outside knowing what a z-index is.
 */
export function claim(rung) {
  const start = baseOf(rung);
  if (!used.has(rung)) used.set(rung, new Set());
  const taken = used.get(rung);

  const nextOffset = () => {
    for (let i = 0; i < SLOTS; i += 1) if (!taken.has(i)) return i;
    // Running a rung dry means something is opening overlays and never
    // releasing them; silently reusing a slot would hide that.
    throw new RangeError(`z rung "${rung}" is full (${SLOTS} live claims). Something is not releasing.`);
  };

  let offset = nextOffset();
  taken.add(offset);
  let live = true;

  const handle = {
    get z() { return start + offset; },
    get rung() { return rung; },
    raise() {
      if (!live) return handle;
      const highest = Math.max(...taken);
      if (offset === highest) return handle;
      taken.delete(offset);
      offset = Math.min(highest + 1, SLOTS - 1);
      if (taken.has(offset)) {           // rung is full at the top; shuffle down
        offset = nextOffset();
      }
      taken.add(offset);
      return handle;
    },
    release() {
      if (!live) return;
      taken.delete(offset);
      live = false;
    },
    get live() { return live; },
  };
  return handle;
}

export function _resetLayers() {
  used.clear();
}
