// Turning announcements into panels.
//
// Everything a module can reach is decided here, and it is deliberately little:
// { composer, values, on }. No DOM node, no document, no kit internals. A module
// cannot style anything because it is never handed anything that could.

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

      const panel = renderPanel(spec, {
        values: values.valuesOf(spec.id),
        onChange: (key, value) => values.set(spec.id, key, value),
      });

      // Only now, with a complete panel in hand, does anything reach the DOM.
      host.appendChild(panel.node);

      if (ui && typeof ui.setup === "function") {
        // Kept, not discarded: setup() returns the unsubscribe from on(), and
        // dropping it means every re-mount leaves another live listener behind.
        teardown = ui.setup({
          composer,
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
