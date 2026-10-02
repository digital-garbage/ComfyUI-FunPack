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

  // Loader nodes core knows how to point /api/probe at, and which of their
  // inputs names the file. Mirrors the same map api.js's probeFamily() is
  // meant to be fed from -- nothing else in this app currently resolves
  // "the file the pipeline's model loader points at" from raw slots.
  const MODEL_FILE_INPUT = {
    FunPackCheckpointLoader: "ckpt_name",
    FunPackDiffusionModelLoader: "model_name",
  };

  function currentModelFile(pipelineSlots) {
    const slot = (pipelineSlots || []).find((s) => MODEL_FILE_INPUT[s.node] && s.inputs);
    return slot ? slot.inputs[MODEL_FILE_INPUT[slot.node]] || null : null;
  }

  // Which traits the pipeline's chosen model has, or null when that is not
  // knowable (no loader slot, no file chosen, or the probe found nothing).
  // null must reach API.modules() as "do not filter" rather than as an
  // empty list -- /api/modules?traits= with an empty traits set hides every
  // module that requires ANY trait at all, for every model this cheap,
  // load-free probe has no opinion about yet (this cheap path only knows
  // MiniMax H3's own traits today -- see modules/models/minimax_h3).
  async function probeModelTraits(pipelineSlots) {
    const filename = currentModelFile(pipelineSlots);
    if (!filename) return null;
    try {
      const data = await API.probeFamily(filename);
      return data.detected ? (data.traits || []) : null;
    } catch (_) {
      return null;
    }
  }

  let modulesById = {};
  let offered = null;         // the server's default pipeline, as first fetched: what a project's saved copy is laid over
  let slots = null;           // null = never loaded this session; loaded exactly once
  // The model file `modulesById` was last filtered for. A save() whose slots
  // still name the same file has nothing new to learn -- re-probing on every
  // unrelated edit (a prompt, a slider) would mean one extra round trip per
  // keystroke-driven save for no reason.
  let lastProbedFile = null;

  async function refreshManifest() {
    const file = currentModelFile(slots);
    if (file === lastProbedFile) return;
    lastProbedFile = file;
    const traits = await probeModelTraits(slots);
    const manifest = await API.modules(traits ? traits.join(",") : undefined);
    modulesById = {};
    (manifest.modules || []).forEach((m) => { modulesById[m.id] = m; });
  }
  let incomplete = [];
  let refused = [];
  let queueable = false;
  let loading = false;
  let loadError = null;
  let deferred = null;        // a project's saved pipeline waiting for the first successful load
  let epoch = 0;              // bumped whenever a project's pipeline replaces the live one
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
        // Sequential, not Promise.all: which modules are even compatible
        // depends on which model the pipeline's loader slot names, so the
        // manifest fetch has to know that before it can ask -- refreshManifest()
        // reads `slots`, so it has to be set first.
        const pipe = await API.pipeline();
        slots = pipe.slots || [];
        offered = JSON.parse(JSON.stringify(slots));
        incomplete = pipe.incomplete || [];
        refused = pipe.refused || [];
        queueable = !!pipe.queueable;
        await refreshManifest();
      } catch (e) {
        loadError = e && e.message ? e.message : String(e);
      }
      loading = false;
      loadPromise = null;
      if (slots !== null && deferred) { const d = deferred; deferred = null; await adopt(d); }
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
      const mine = epoch;
      try {
        const res = await API.editPipeline(toSend);
        // A different pipeline was put in while this was in flight (another project opened):
        // this answer is about the old one and must not overwrite the new.
        if (mine !== epoch) continue;
        if (res && res.slots) slots = res.slots;
        incomplete = (res && res.incomplete) || [];
        refused = (res && res.refused) || [];
        queueable = !!(res && res.queueable);
        saveNotes = (res && res.notes) || [];
        _changed();
        // A no-op for the common case (same file as last time) -- see
        // refreshManifest()'s own guard. Only a Models & Pipeline edit that
        // actually changes the loader's file does a second round trip here.
        // Its own try/catch: a failure here is not "could not save" -- the
        // edit above already landed -- so it must not overwrite saveNotes
        // with a message about the wrong failure.
        try {
          await refreshManifest();
        } catch (e) {
          console.warn(`[FunPack] could not refresh modules for the new model: ${e && e.message ? e.message : e}`);
        }
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

  // Whoever keeps the pipeline with the project hears every landed edit.
  const listeners = new Set();
  function _changed() { listeners.forEach((fn) => { try { fn(slots); } catch (e) { console.error(e); } }); }

  // Put a project's saved pipeline over the one the server offers: the person's inputs (model
  // files, node values, the module-settings blob), group and bypass win on every slot the
  // default still has; slots they added are kept. The server's own defaults fill in whatever
  // the saved copy predates, so an update's new settings are not lost to an old project.
  // Not announced as a change: it IS the project's own copy.
  async function adopt(saved) {
    await ensureLoaded();
    // Not loaded (ComfyUI unreachable): remember what the project holds and put it in the moment a
    // load succeeds, so the default is never what the next edit is built on.
    if (slots === null) { if (Array.isArray(saved) && saved.length) deferred = saved; return; }
    if (!Array.isArray(saved) || !saved.length) return;
    // A saved slot is only what the server would accept: a project file outlives the code that
    // wrote it (v4 files name the node differently), and one bad slot refuses every later edit.
    const sound = (s) => s && typeof s.id === "string" && typeof s.node === "string"
      && s.inputs && typeof s.inputs === "object" && !Array.isArray(s.inputs);
    const byId = new Map(saved.filter(sound).map((s) => [s.id, s]));
    const have = new Set(offered.map((s) => s.id));
    const lay = (extras) => offered.map((def) => {
      const mine = byId.get(def.id);
      if (!mine || mine.node !== def.node) return JSON.parse(JSON.stringify(def));
      const out = { ...def, inputs: { ...(def.inputs || {}), ...mine.inputs } };
      if (typeof mine.group === "string" && mine.group) out.group = mine.group;
      if (typeof mine.bypassed === "boolean") out.bypassed = mine.bypassed;
      return out;
    }).concat(extras ? saved.filter((s) => sound(s) && !have.has(s.id)) : []);
    // Replaces whatever pipeline was live: nothing queued for the old one may land on the new.
    const mine = ++epoch;
    pendingBody = null; pending = false;
    for (let waited = 0; saving && waited < 10000; waited += 20) await new Promise((r) => setTimeout(r, 20));
    for (const extras of [true, false]) {
      try {
        const res = await API.editPipeline({ slots: lay(extras) });
        if (mine !== epoch) return;                     // a later project was opened meanwhile
        if (res && res.slots && !(res.refused || []).length) {
          slots = res.slots;
          incomplete = res.incomplete || []; refused = []; queueable = !!res.queueable;
          saveNotes = res.notes || [];
          // The modules offered depend on the model this pipeline names, which may differ from the
          // previous project's: refresh, or the next settings edit drops this project's modules.
          try { await refreshManifest(); } catch (_) { /* the next save retries it */ }
          return;
        }
      } catch (_) { /* this layering was refused; try the next */ }
    }
    slots = JSON.parse(JSON.stringify(offered));        // nothing saved was usable: the default, said
    saveNotes = ["This project's saved pipeline could not be loaded, so the default is in use."];
  }

  window.PipelineState = {
    ensureLoaded, save, adopt, subscribe: (fn) => { listeners.add(fn); return () => listeners.delete(fn); }, valuesAlreadyPlaced, currentValues, setModuleValue,
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
