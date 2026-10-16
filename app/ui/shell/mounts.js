// Mount points are ANNOUNCED by the regions that offer them.
//
// A fixed list in core would be a global list -- the one thing the architecture
// avoids -- and it would have to be edited every time the shell grows a region.
// Instead a region says what it can host, and a module naming something no
// region offers is simply absent. That is "hide, don't warn" one level up.

const regions = new Map();          // mount point -> { host, standIn }

/**
 * offer(mount, host, standIn?)
 *
 * `standIn` is what the region shows while nothing has mounted into it -- an
 * empty state, not a label. It is taken down by settle() once something is
 * there, because a region that says "modules appear here" ABOVE the modules
 * that appeared is a region explaining itself to nobody.
 */
export function offer(mount, host, standIn = null) {
  if (regions.has(mount)) {
    throw new Error(`Mount point "${mount}" is already offered by another region.`);
  }
  regions.set(mount, { host, standIn });
  return () => regions.delete(mount);
}

export const hostFor = (mount) => {
  const region = regions.get(mount);
  return region ? region.host : null;
};
export const offered = () => [...regions.keys()].sort();
export const claimed = (mount) => regions.has(mount);

/**
 * Take down the stand-in of every region something mounted into.
 *
 * Called once, after everything that mounts has had its turn. Doing it as each
 * one mounts would work too and would be wrong the moment two regions share a
 * host: the one that mounted first would take down a stand-in belonging to
 * both.
 */
export function settle() {
  for (const { host, standIn } of regions.values()) {
    if (!standIn || !standIn.parentNode) continue;
    const others = [...host.children].filter((child) => child !== standIn);
    if (others.length) standIn.remove();
  }
}

export function _reset() {
  regions.clear();
}
