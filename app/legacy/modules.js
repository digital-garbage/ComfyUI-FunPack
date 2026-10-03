// The browser-side module contract. A feature file calls Modules.define(id, {needs}, mount): the app
// core runs it, and a feature that cannot run is rejected and its UI removed -- the baseline never sees it.
//   needs: window globals the feature stands on (a missing one rejects it with a reason, nothing thrown)
(function (root) {
  const list = [];

  function define(id, opts, mount) {
    const entry = { id, ok: false, why: "" };
    list.push(entry);
    const missing = ((opts && opts.needs) || []).filter((name) => !root[name]);
    if (missing.length) { entry.why = `needs ${missing.join(", ")}`; return entry; }
    // Whatever the feature puts on the page while mounting is taken back if it throws half-way.
    const seen = root.MutationObserver ? new root.MutationObserver(() => {}) : null;
    if (seen) seen.observe(root.document, { childList: true, subtree: true });
    try {
      mount();
      entry.ok = true;
    } catch (e) {
      entry.why = String((e && e.message) || e);
      (seen ? seen.takeRecords() : []).forEach((r) => r.addedNodes.forEach((n) => n.remove && n.remove()));
      console.warn(`[FunPack] module "${id}" rejected: ${entry.why}`);
    }
    if (seen) seen.disconnect();
    return entry;
  }

  root.Modules = { define, status: () => list.map((m) => ({ ...m })) };
})(typeof window !== "undefined" ? window : globalThis);
