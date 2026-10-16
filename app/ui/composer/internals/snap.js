// Magnetic edges: a dragged edge within `reach` seconds of an anchor lands on it.

/** The shift (seconds) that puts the nearest of `edges` on the nearest anchor, or `delta` unchanged when none is close. */
export function snapDelta(delta, edges, anchors, reach) {
  let best = null;
  for (const edge of edges) for (const a of anchors) {
    if (Math.abs(a - edge) < 1e-6) continue;          // where the edge already is: a butting neighbour must not pin it
    const off = a - (edge + delta);
    if (Math.abs(off) < reach && (best === null || Math.abs(off) < Math.abs(best))) best = off;
  }
  return best === null ? delta : delta + best;
}
