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

  window.PipelineState = {
    ensureLoaded, save, valuesAlreadyPlaced,
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
