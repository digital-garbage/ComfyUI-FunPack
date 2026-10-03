// The dock: which of Assets, Preview, Properties are showing. Kept in this browser, written onto the page as
// `fp-hide-<zone>` classes where the stylesheet reads them.
const KEY = "funpack_dock", ZONES = ["assets", "preview", "properties"];

export function createDock(root = document.documentElement) {
  const heard = new Set();
  let hidden = new Set();
  try { hidden = new Set(JSON.parse(localStorage.getItem(KEY)).filter((z) => ZONES.includes(z))); } catch { /* none remembered */ }
  const paint = () => ZONES.forEach((z) => root.classList.toggle(`fp-hide-${z}`, hidden.has(z)));
  const change = () => { paint(); try { localStorage.setItem(KEY, JSON.stringify([...hidden])); } catch { /* lasts until reload */ } heard.forEach((fn) => fn()); };
  paint();
  return {
    shown: (zone) => !hidden.has(zone),
    toggle(zone) { if (!ZONES.includes(zone)) return; if (!hidden.delete(zone)) hidden.add(zone); change(); },
    reset() { if (hidden.size) { hidden.clear(); change(); } },
    on(fn) { heard.add(fn); return () => heard.delete(fn); },
  };
}
