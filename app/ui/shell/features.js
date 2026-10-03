// Loads UI features. A feature file's default export is one hub, or several:
//   { id, mount, needs?, setup({ host, app }) -> teardown? }
// `mount` names a region (see mounts.js). A feature that cannot load, names a region nobody offers,
// needs a service the app lacks, or throws in setup is hidden -- nothing of it reaches the screen.
import { hostFor, offered } from "./mounts.js";

function mountOne(hub, has, app) {
  const host = hostFor(hub.mount);
  if (!host) throw new Error(`no region offers "${hub.mount}" (offered: ${offered().join(", ")})`);
  const lacking = (hub.needs || []).filter((name) => !has(name));
  if (lacking.length) throw new Error(`needs ${lacking.join(", ")}`);
  const before = [...host.childNodes];
  try {
    return { id: hub.id, teardown: hub.setup({ host, app }) };
  } catch (err) {                           // whatever it put on the page goes with it
    [...host.childNodes].filter((n) => !before.includes(n)).forEach((n) => n.remove());
    throw err;
  }
}

// `app` is what the core offers features (project, ...). A feature names what it needs; a missing one hides it.
export async function loadFeatures(paths, { load = (path) => import(path), app = {}, has = (name) => name in app } = {}) {
  const mounted = [], hidden = [];
  const hide = (id, err) => { hidden.push({ id, why: err.message }); console.warn(`[FunPack] ${id} hidden: ${err.message}`); };
  for (const path of paths) {
    let hubs;
    try { hubs = [].concat((await load(path)).default); } catch (err) { hide(path, err); continue; }
    for (const hub of hubs) {
      try { mounted.push(mountOne(hub, has, app)); } catch (err) { hide(hub?.id || path, err); }
    }
  }
  return { mounted, hidden };
}
