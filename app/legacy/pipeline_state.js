// The pipeline's slots, held ONCE per page session and shared by every panel
// that edits them (Engine Settings' module-values, Models & Pipeline's
// node-input edits). Two independent copies of "the current slots", each
// re-fetched on its own panel's mount, is how one panel's save silently
// reverts the other's: GET /api/pipeline is stateless (core/routes.py builds
// the DEFAULT graph fresh on every call, remembering nothing), so a panel
// that fetches on reopen throws away whatever ANY panel already placed --
// its own edits included, per the bug this file was extracted to fix once
// for good rather than have each new panel rediscover it.
(function () {
  const API = window.MovieEditorAPI;

  let modulesById = {};
  let slots = null;           // null = never loaded this session; loaded exactly once
  let incomplete = [];
  let refused = [];
  let queueable = false;
  let loading = false;
  let loadError = null;
  let saving = false;
  let pending = false;
  let pendingBody = null;
  let saveNotes = [];

  // What a settings_sink already holds, read back out of the live slots --
  // place() (core/graph.py) writes a whole values blob as one opaque JSON
  // string into the sink's input, never merging. No route tells the client
  // WHICH node/input is the real sink (core does not name an implementation,
  // by design), so this scans every string input for one that decodes to a
  // plain object -- and only accepts a decoded key that names a module this
  // session already knows is installed, so an unrelated node's incidentally
  // JSON-shaped string can't inject a bogus entry that round-trips forever.
  function valuesAlreadyPlaced() {
    const known = new Set(Object.keys(modulesById));
    const merged = {};
    (slots || []).forEach((slot) => {
      Object.values(slot.inputs || {}).forEach((v) => {
        if (typeof v !== "string") return;
        let parsed;
        try { parsed = JSON.parse(v); } catch (_) { return; }
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return;
        Object.entries(parsed).forEach(([moduleId, own]) => {
          if (!known.has(moduleId)) return;
          if (own && typeof own === "object") merged[moduleId] = { ...(merged[moduleId] || {}), ...own };
        });
      });
    });
    return merged;
  }

  // Two panels can mount close enough together that both call this before
  // the first fetch settles (`slots` only becomes non-null at the end of a
  // load, so a bare `if (slots !== null) return` doesn't cover the window
  // while one is already in flight). Every caller awaits the SAME in-flight
  // promise instead, so exactly one fetch ever goes out per session, however
  // many panels ask.
  let loadPromise = null;

  async function ensureLoaded() {
    if (slots !== null) return; // session-wide: every consumer shares this one load
    if (loadPromise) return loadPromise;
    loading = true; loadError = null;
    loadPromise = (async () => {
      try {
        const [manifest, pipe] = await Promise.all([API.modules(), API.pipeline()]);
        modulesById = {};
        (manifest.modules || []).forEach((m) => { modulesById[m.id] = m; });
        slots = pipe.slots || [];
        incomplete = pipe.incomplete || [];
        refused = pipe.refused || [];
        queueable = !!pipe.queueable;
      } catch (e) {
        loadError = e && e.message ? e.message : String(e);
      }
      loading = false;
      loadPromise = null;
    })();
    return loadPromise;
  }

  // `save()`'s body is one level deeper than a shallow spread can merge
  // correctly: `inputs`/`values` are both {outerId: {innerKey: value}}, and
  // two queued saves that both touch `inputs` (two different slots, or even
  // two fields on the SAME slot from models.js's per-call-fresh objects) had
  // the second's `inputs` object silently REPLACE the first's rather than
  // combine with it -- the earlier edit vanished with no error. Merges one
  // level deeper than Object.assign/spread does, for exactly the two keys
  // that are ever shaped this way.
  function mergeBodies(a, b) {
    const out = { ...(a || {}) };
    Object.entries(b || {}).forEach(([key, val]) => {
      const prior = out[key];
      const bothPlainObjects = val && typeof val === "object" && !Array.isArray(val)
        && prior && typeof prior === "object" && !Array.isArray(prior);
      if (!bothPlainObjects) { out[key] = val; return; }
      const merged = { ...prior };
      Object.entries(val).forEach(([innerId, inner]) => {
        const priorInner = merged[innerId];
        merged[innerId] = (inner && typeof inner === "object" && !Array.isArray(inner)
          && priorInner && typeof priorInner === "object" && !Array.isArray(priorInner))
          ? { ...priorInner, ...inner }
          : inner;
      });
      out[key] = merged;
    });
    return out;
  }

  // One shared save queue for every caller. `body` is layered onto the
  // CURRENT `slots` (the one shared array every panel mutates through its
  // own edits) each time a request actually goes out -- so a Models &
  // Pipeline node-input edit and an Engine Settings module-value edit, made
  // moments apart, both survive: neither's save can send a snapshot that
  // predates the other's. See engine_settings.js's original fix for why the
  // retry-after-in-flight shape exists (a same-panel double-edit dropped
  // one edit before this).
  async function save(body) {
    pendingBody = mergeBodies(pendingBody, body);
    if (saving) { pending = true; return; }
    saving = true;
    do {
      pending = false;
      const toSend = { slots, ...pendingBody };
      pendingBody = null;
      try {
        const res = await API.editPipeline(toSend);
        if (res && res.slots) slots = res.slots;
        incomplete = (res && res.incomplete) || [];
        refused = (res && res.refused) || [];
        queueable = !!(res && res.queueable);
        saveNotes = (res && res.notes) || [];
      } catch (e) {
        saveNotes = [`Could not save: ${e && e.message ? e.message : e}`];
      }
    } while (pending);
    saving = false;
  }

  // Edits already sent toward the server but not yet confirmed by a landed
  // `slots` response. valuesAlreadyPlaced() only reflects an edit AFTER its
  // save's response arrives -- without this overlay, a second setModuleValue
  // call fired before the first one's round-trip resolves would compute its
  // own "whole tree" snapshot from data that does not include the first
  // edit yet, and silently revert it when its own write lands. Cleared per
  // key once valuesAlreadyPlaced() actually agrees with it (see
  // _reconcilePending), not wholesale on every resolve -- clearing the
  // whole thing on any one save's completion would erase a DIFFERENT edit
  // that is still in flight.
  let pendingValues = {};

  function _reconcilePending() {
    const already = valuesAlreadyPlaced();
    Object.keys(pendingValues).forEach((moduleId) => {
      const pend = pendingValues[moduleId];
      const real = already[moduleId] || {};
      Object.keys(pend).forEach((name) => { if (real[name] === pend[name]) delete pend[name]; });
      if (!Object.keys(pend).length) delete pendingValues[moduleId];
    });
  }

  // The full module-settings tree as it stands right now: every installed
  // module's own defaults, then whatever the graph already holds wins (same
  // recovery valuesAlreadyPlaced() exists for), then any not-yet-confirmed
  // edit wins over that. Always computed fresh rather than cached, so two
  // editors of the same tree (Engine Settings, the sampler quick-access bar)
  // can never hold a stale copy of each other's CONFIRMED state between
  // them -- the pending overlay above is the one deliberate exception,
  // needed so an in-flight edit from either editor isn't lost by the other.
  function currentValues() {
    const merged = {};
    Object.values(modulesById).forEach((m) => {
      const own = {};
      Object.entries(m.settings || {}).forEach(([name, spec]) => { own[name] = spec.default; });
      if (Object.keys(own).length) merged[m.id] = own;
    });
    const already = valuesAlreadyPlaced();
    Object.entries(already).forEach(([moduleId, own]) => {
      merged[moduleId] = { ...(merged[moduleId] || {}), ...own };
    });
    Object.entries(pendingValues).forEach(([moduleId, own]) => {
      merged[moduleId] = { ...(merged[moduleId] || {}), ...own };
    });
    return merged;
  }

  // Patch ONE field and save the WHOLE tree. place() (core/graph.py) writes
  // values as a single opaque JSON blob with no merge on the server side --
  // sending only the module being edited would silently erase every other
  // module's settings from that blob. Every write goes through here so that
  // rule lives in one place instead of being re-derived per caller.
  // pendingValues is updated SYNCHRONOUSLY, before the read below, so a
  // second setModuleValue call made while this one is still in flight sees
  // this edit too (see currentValues()) -- that is what actually closes the
  // race, not save()'s own body-merge queue, which only merges DELTAS and
  // has nothing to merge against once a caller is sending a full snapshot.
  //
  // The returned promise is NOT "this specific edit is server-confirmed" --
  // save()'s queue has exactly one caller actually await the network (the
  // one that finds `saving` false); every other concurrent caller's save()
  // call returns as soon as it queues, before its own write is sent, let
  // alone answered. Harmless today: both real callers (engine_settings.js,
  // sampler_quickbar.js) only do `.then(render)`, and render() reads
  // currentValues(), which already carries this edit optimistically via
  // pendingValues regardless of where the real network call is. A future
  // caller that needs "my write actually landed" (a per-edit success
  // toast, reading saveNotes()/incomplete() right after resolve) cannot
  // get that from this promise as written -- it would need save() itself
  // reworked to resolve each queued caller at ITS OWN write's landing, not
  // at whichever write happens to be in flight when the queue drains.
  function setModuleValue(moduleId, name, value) {
    pendingValues[moduleId] = { ...(pendingValues[moduleId] || {}), [name]: value };
    const values = currentValues();
    return save({ values }).then(_reconcilePending);
  }

  window.PipelineState = {
    ensureLoaded, save, valuesAlreadyPlaced, currentValues, setModuleValue,
    modulesById: () => modulesById,
    slots: () => slots,
    incomplete: () => incomplete,
    refused: () => refused,
    queueable: () => queueable,
    loading: () => loading,
    loadError: () => loadError,
    saving: () => saving,
    saveNotes: () => saveNotes,
  };
})();
