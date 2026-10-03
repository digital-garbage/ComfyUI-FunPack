// Adaptive sizing from the CONTAINER, not the viewport.
//
// v4's media bin got this right and nothing else used it: a panel that is 280px
// wide should lay out like a 280px panel whether the window is 1200px or 3000px.
// The column ratios there were hand-tuned (92/44/29/21 cqw) and drifted from the
// gaps they were meant to account for; here they are derived.

export const AUTO = 0;
export const MAX_COLS = 4;

/**
 * Cell width for `cols` columns inside a container, as a cqw percentage,
 * accounting for the gaps between them. cols = 0 means "as many as fit".
 */
export function cellWidth(cols, gapPercent = 2.5) {
  if (!Number.isInteger(cols) || cols < 1) return null;      // auto
  const gaps = (cols - 1) * gapPercent;
  return Math.max(1, Number(((100 - gaps) / cols).toFixed(2)));
}

/** The CSS value for `--cell` at a given column count. */
export function cellExpression(cols) {
  const pct = cellWidth(cols);
  return pct === null
    ? "clamp(var(--cell-thumb), 46cqw, var(--cell-panel))"
    : `${pct}cqw`;
}

export function normaliseCols(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return AUTO;
  return Math.min(MAX_COLS, Math.max(AUTO, Math.trunc(n)));
}

/** Apply a column choice to a grid host. */
export function applyDensity(host, cols) {
  const n = normaliseCols(cols);
  host.dataset.cols = String(n);
  host.style.setProperty("--cell", cellExpression(n === AUTO ? null : n));
  return n;
}

// Per-viewer convenience, not state anything depends on: a browser that refuses
// storage should still get a working grid.
export function rememberCols(id, cols) {
  try { localStorage.setItem(`cx.density.${id}`, String(normaliseCols(cols))); } catch { /* private mode */ }
}

export function recallCols(id, fallback = AUTO) {
  try {
    const raw = localStorage.getItem(`cx.density.${id}`);
    return raw === null ? fallback : normaliseCols(raw);
  } catch {
    return fallback;
  }
}
