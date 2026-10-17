// Turning announcements into panels.
//
// A module draws nothing. It declares its settings and this file has the kit
// render them: no DOM node, no document, no element factory ever reaches a module.
// A module with a broken declaration is hidden whole, never half-mounted.

import { renderPanel, defaultsOf } from "../composer/panel.js";
import { hostFor, offered } from "./mounts.js";
import * as values from "./values.js";

/**
 * mountAll(manifest) -> { mounted, hidden }
 *
 * `hidden` is diagnostic only. Nothing about a hidden module reaches the screen:
 * no placeholder, no greyed row, no warning chip. A module that failed is
 * indistinguishable from one that was never installed, which is the point --
 * the alternative is a UI full of things that cannot be used.
 */
export async function mountAll(manifest) {
  const mounted = [];
  const hidden = [];

  for (const spec of manifest.modules || []) {
    // Nothing to render. A module may exist purely to contribute ComfyUI nodes
    // (a loader, a model's compatibility module), and those have no panel and
    // no mount. Skipping is not hiding: there is no failure to report.
    if (!Object.keys(spec.settings || {}).length) continue;

    const host = hostFor(spec.mount);
    if (!host) {
      hidden.push({ id: spec.id, why: `no region offers "${spec.mount}" (offered: ${offered().join(", ") || "none"})` });
      continue;
    }

    try {
      values.seed(spec.id, defaultsOf(spec));

      const panel = renderPanel(spec, {
        values: values.valuesOf(spec.id),
        onChange: (key, value) => values.set(spec.id, key, value),
      });

      // Only a complete panel reaches the DOM.
      host.appendChild(panel.node);

      mounted.push({
        id: spec.id,
        panel,
        destroy() { panel.destroy(); },
      });
    } catch (err) {
      hidden.push({ id: spec.id, why: `${err.name}: ${err.message}` });
      console.warn(`[FunPack] ${spec.id} did not mount: ${err.message}`);
    }
  }

  return { mounted, hidden };
}
