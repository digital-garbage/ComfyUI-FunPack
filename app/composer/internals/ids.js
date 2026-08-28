// Unique ids, so a label can point at its control without the caller inventing
// a naming scheme (and colliding when the same panel is mounted twice).

let n = 0;

export function uid(prefix = "cx") {
  n += 1;
  return `${prefix}-${n}`;
}

// Tests only: ids are otherwise monotonic for the life of the page.
export function _resetIds() {
  n = 0;
}
