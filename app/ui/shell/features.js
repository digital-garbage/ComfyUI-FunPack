// Loads UI features. A feature is a folder with an index.js hub:
//   export default { id, mount, needs?, setup({ host }) -> teardown? }
// `mount` names a region (see mounts.js). A feature that cannot load, names a region nobody offers,
// needs a service the app lacks, or throws in setup is hidden -- nothing of it reaches the screen.
import { hostFor, offered } from "./mounts.js";

export async function loadFeatures(paths, { load = (path) => import(path), has = () => true } = {}) {
  const mounted = [], hidden = [];
  for (const path of paths) {
    let hub;
    try {
      hub = (await load(path)).default;
      const host = hostFor(hub.mount);
      if (!host) throw new Error(`no region offers "${hub.mount}" (offered: ${offered().join(", ")})`);
      const lacking = (hub.needs || []).filter((name) => !has(name));
      if (lacking.length) throw new Error(`needs ${lacking.join(", ")}`);
      const before = [...host.childNodes];
      try {
        mounted.push({ id: hub.id, teardown: hub.setup({ host }) });
      } catch (err) {                       // whatever it put on the page goes with it
        [...host.childNodes].filter((n) => !before.includes(n)).forEach((n) => n.remove());
        throw err;
      }
    } catch (err) {
      hidden.push({ id: hub?.id || path, why: err.message });
      console.warn(`[FunPack] ${hub?.id || path} hidden: ${err.message}`);
    }
  }
  return { mounted, hidden };
}
