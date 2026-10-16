// Turning announcements into panels.
//
// Everything a module can reach is decided here, and it is deliberately little:
// { values, on, shell }. No DOM node, no document, no element factory.
//
// `composer` used to be in that list, and it was the hole in the guarantee: a
// factory returns a handle whose `.node` is a real element, so a module could
// take one, style it, and reach the whole document through `node.ownerDocument`
// -- no lint to evade, just the intended API. No module ever used it. Panels are
// rendered from the DECLARATION, and a module that wants to draw something is a
// module that should be declaring it instead.
//
// What is true, stated exactly: the kit hands a module nothing that touches the
// DOM. What is NOT true, and must not be claimed: that a module CANNOT reach the
// DOM. A ui.js is an ES module in the page realm and `document` is a global.
// tools/kit_lint raises the cost of doing that accidentally; it is a discipline
// check and not a boundary, and only real isolation (a worker, another realm)
// would make it one. First-party modules in this repo are reviewed; a
// third-party module would need that isolation before it could be trusted.

import { composer } from "../composer/composer.js";
import { renderPanel, defaultsOf } from "../composer/panel.js";
import { hostFor, offered } from "./mounts.js";
import * as values from "./values.js";
import { services } from "./services.js";

/**
 * mountAll(manifest) -> { mounted, hidden }
 *
 * `hidden` is diagnostic only. Nothing about a hidden module reaches the screen:
 * no placeholder, no greyed row, no warning chip. A module that failed is
 * indistinguishable from one that was never installed, which is the point --
 * the alternative is a UI full of things that cannot be used.
 */
export async function mountAll(manifest, { load = (path) => import(path) } = {}) {
  const mounted = [];
  const hidden = [];

  for (const spec of manifest.modules || []) {
    // Nothing to render. A module may exist purely to contribute ComfyUI nodes
    // (a loader, a model's compatibility module), and those have no panel and
    // no mount. Skipping is not hiding: there is no failure to report.
    const renders = Object.keys(spec.settings || {}).length > 0 || spec.ui;
    if (!renders) continue;

    const host = hostFor(spec.mount);
    if (!host) {
      hidden.push({ id: spec.id, why: `no region offers "${spec.mount}" (offered: ${offered().join(", ") || "none"})` });
      continue;
    }

    try {
      // Values first: a module's ui.js may read them during setup, and a
      // headless run has them without any of this having happened.
      values.seed(spec.id, defaultsOf(spec));

      let ui = null;
      if (spec.ui) ui = await load(spec.ui);
      let teardown = null;

      if (ui && typeof ui.setup === "function") {
        // Kept, not discarded: setup() returns the unsubscribe from on(), and
        // dropping it means every re-mount leaves another live listener behind.
        teardown = ui.setup({
          values: {
            get: () => values.valuesOf(spec.id),
            set: (key, value) => values.set(spec.id, key, value),
          },
          // Scoped to this module: a module hearing about everyone else's
          // edits would have to filter, and filtering needs an id it should
          // not need to know.
          on: (fn) => values.onChange((id, key, value) => {
            if (id === spec.id) fn(key, value);
          }),
          // The app services a module may use, named explicitly. A short,
          // reviewable list beats reaching for a global, which is how a module
          // ends up depending on something nobody meant to expose.
          shell: services,
        });
      }

      // Built AFTER setup, so a control shows what setup settled on. The panel
      // is drawn from a snapshot of the values, so building it first meant a
      // module that adopted a live value in setup -- the theme, which the page
      // applies before any module exists -- had a control showing something the
      // app was not set to.
      const panel = renderPanel(spec, {
        values: values.valuesOf(spec.id),
        onChange: (key, value) => values.set(spec.id, key, value),
      });

      // Only now, with a complete panel AND a setup() that returned, does
      // anything reach the DOM. Appending before setup ran left a module whose
      // setup threw reported as hidden with its panel still on screen -- which
      // is precisely the guarantee this file states it keeps.
      host.appendChild(panel.node);

      mounted.push({
        id: spec.id,
        panel,
        destroy() {
          if (typeof teardown === "function") teardown();
          panel.destroy();
        },
      });
    } catch (err) {
      hidden.push({ id: spec.id, why: `${err.name}: ${err.message}` });
      console.warn(`[FunPack] ${spec.id} did not mount: ${err.message}`);
    }
  }

  return { mounted, hidden };
}
