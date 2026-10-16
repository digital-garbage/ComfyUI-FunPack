// Measure, flip, clamp -- once.
//
// v4 had seven hand-rolled copies of this (autocomplete, suggestions, rating
// picker, role menu, choice picker, timeline dropdown, composer add-menu), each
// with its own idea of what to do near an edge. The geometry here is pure and
// takes plain rectangles, so it can be tested exhaustively without a browser;
// `anchorTo` is the thin part that touches the DOM.

const OPPOSITE = { top: "bottom", bottom: "top", left: "right", right: "left" };

/**
 * place({ anchor, float, viewport, side, align, gap, flip, clamp, padding })
 *
 * anchor   {x, y, width, height}  the thing being pointed at, in viewport coords
 * float    {width, height}        the thing being positioned
 * viewport {width, height}
 *
 * Returns { x, y, side, align, flipped, clamped }. `side` is where the float
 * ENDED UP, which is what a caller needs to point an arrow the right way.
 */
export function place({
  anchor,
  float,
  viewport,
  side = "bottom",
  align = "start",
  gap = 6,
  flip = true,
  clamp = true,
  padding = 8,
} = {}) {
  if (!OPPOSITE[side]) throw new RangeError(`Unknown side "${side}".`);

  let chosen = side;
  let flipped = false;

  if (flip && !fits(chosen, anchor, float, viewport, gap, padding)) {
    const other = OPPOSITE[chosen];
    // Only flip if the other side is genuinely better -- flipping into an even
    // tighter gap just moves the clipping somewhere less expected.
    if (room(other, anchor, viewport) > room(chosen, anchor, viewport)) {
      chosen = other;
      flipped = true;
    }
  }

  let { x, y } = position(chosen, align, anchor, float, gap);
  let clamped = false;

  if (clamp) {
    const cx = Math.min(Math.max(x, padding), Math.max(padding, viewport.width - float.width - padding));
    const cy = Math.min(Math.max(y, padding), Math.max(padding, viewport.height - float.height - padding));
    clamped = cx !== x || cy !== y;
    x = cx;
    y = cy;
  }

  return { x: Math.round(x), y: Math.round(y), side: chosen, align, flipped, clamped };
}

function room(side, anchor, viewport) {
  switch (side) {
    case "top": return anchor.y;
    case "bottom": return viewport.height - (anchor.y + anchor.height);
    case "left": return anchor.x;
    case "right": return viewport.width - (anchor.x + anchor.width);
    default: return 0;
  }
}

function fits(side, anchor, float, viewport, gap, padding) {
  const needed = (side === "top" || side === "bottom" ? float.height : float.width) + gap + padding;
  return room(side, anchor, viewport) >= needed;
}

function position(side, align, anchor, float, gap) {
  const vertical = side === "top" || side === "bottom";
  const main = vertical
    ? (side === "bottom" ? anchor.y + anchor.height + gap : anchor.y - float.height - gap)
    : (side === "right" ? anchor.x + anchor.width + gap : anchor.x - float.width - gap);

  const anchorStart = vertical ? anchor.x : anchor.y;
  const anchorSize = vertical ? anchor.width : anchor.height;
  const floatSize = vertical ? float.width : float.height;

  let cross;
  if (align === "center") cross = anchorStart + anchorSize / 2 - floatSize / 2;
  else if (align === "end") cross = anchorStart + anchorSize - floatSize;
  else cross = anchorStart;

  return vertical ? { x: cross, y: main } : { x: main, y: cross };
}

/**
 * Position `floatNode` against `anchorNode`. The float must already be in the
 * document and visible to measure -- a hidden element measures as 0x0 and lands
 * in the corner, which is how "my popover appears top-left" bugs happen.
 */
export function anchorTo(floatNode, anchorNode, opts = {}) {
  const a = anchorNode.getBoundingClientRect();
  const f = floatNode.getBoundingClientRect();
  const result = place({
    anchor: { x: a.x, y: a.y, width: a.width, height: a.height },
    float: { width: f.width, height: f.height },
    viewport: { width: window.innerWidth, height: window.innerHeight },
    ...opts,
  });
  floatNode.style.position = "fixed";
  floatNode.style.left = `${result.x}px`;
  floatNode.style.top = `${result.y}px`;
  floatNode.dataset.side = result.side;
  return result;
}
