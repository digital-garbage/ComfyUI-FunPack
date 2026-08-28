// Mount points are ANNOUNCED by the regions that offer them.
//
// A fixed list in core would be a global list -- the one thing the architecture
// avoids -- and it would have to be edited every time the shell grows a region.
// Instead a region says what it can host, and a module naming something no
// region offers is simply absent. That is "hide, don't warn" one level up.

const regions = new Map();          // mount point -> host element

export function offer(mount, host) {
  if (regions.has(mount)) {
    throw new Error(`Mount point "${mount}" is already offered by another region.`);
  }
  regions.set(mount, host);
  return () => regions.delete(mount);
}

export const hostFor = (mount) => regions.get(mount) || null;
export const offered = () => [...regions.keys()].sort();
export const claimed = (mount) => regions.has(mount);

export function _reset() {
  regions.clear();
}
