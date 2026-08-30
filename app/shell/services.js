// What the app offers modules beyond the kit.
//
// Deliberately tiny and named one by one. The alternative -- letting a module
// reach for a global -- means the surface is "whatever happens to be on window",
// which nobody can review and nothing can keep stable.

import { applyDensity, normaliseCols, rememberCols, recallCols } from "../composer/internals/density.js";

export const services = {
  theme: {
    set(choice) { window.ComposerTheme.apply(choice); },
    get: () => window.ComposerTheme.get(),
    resolved: () => window.ComposerTheme.resolved(),
  },
  density: {
    /** What the grids are set to, which the browser remembers between visits. */
    get: () => recallCols("app"),
    /** Column count for every adaptive grid on the page. */
    set(cols) {
      const n = normaliseCols(cols);
      for (const grid of document.querySelectorAll(".cx-gallery")) applyDensity(grid, n);
      rememberCols("app", n);
      return n;
    },
  },
};
