// The pipeline's slots, held ONCE per page session and shared by every panel
// that edits them (Engine Settings' module-values, Models & Pipeline's
// node-input edits). Two independent copies of "the current slots", each
// re-fetched on its own panel's mount, is how one panel's save silently
// reverts the other's: GET /api/pipeline is stateless (core/routes.py builds
// the DEFAULT graph fresh on every call, remembering nothing), so a panel
// that fetches on reopen throws away whatever ANY panel already placed --
// its own edits included, per the bug this file was extracted to fix once
// for good rather than have each new panel rediscover it.
// Every request this state makes gives up after a while: one that never answers (a stalled tunnel, a Mac that slept)
// would otherwise hold the pipeline's lock, and every edit behind it, until a reload. A Proxy, so an API object
// changed after creation (tests do) is still the one called.
const ANSWER_MS = 30000;
const bounded = (api, ms) => new Proxy(api, {
  get(target, key) {
    const f = target[key];
    if (typeof f !== "function") return f;
    return (...args) => {
      let timer;
      const late = new Promise((_, no) => { timer = setTimeout(() => no(new Error("ComfyUI did not answer in time")), ms); });
      return Promise.race([Promise.resolve(f.apply(target, args)), late]).finally(() => clearTimeout(timer));
    };
  },
});

export function createPipelineState(rawApi, { answerMs = ANSWER_MS } = {}) {
  const API = bounded(rawApi, answerMs);

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

  // Which traits the pipeline's chosen model has; null when that is not knowable (no file, or no model
  // module recognised it, or the one that did names no traits): null means "do not filter", while a list,
  // even an empty one, hides every module needing a trait it lacks. undefined: the probe failed, try again.
  let modelNote = null;       // why nothing is hidden for the chosen file, when no model module recognised it
  async function probeModelTraits(pipelineSlots) {
    const filename = currentModelFile(pipelineSlots);
    modelNote = null;
    if (!filename) return null;
    try {
      const data = await API.probeFamily(filename);
      if (!data.detected && data.reason) modelNote = `${data.reason}, so every module is offered, including ones this model cannot use.`;
      return data.detected && Array.isArray(data.traits) ? data.traits : null;
    } catch (_) {
      return undefined;
    }
  }

  let modulesById = {};
  let offered = null;         // the server's default pipeline, as first fetched: what a project's saved copy is laid over
  let slots = null;           // null = never loaded this session; loaded exactly once
  // The model file `modulesById` was last filtered for. A save() whose slots
  // still name the same file has nothing new to learn -- re-probing on every
  // unrelated edit (a prompt, a slider) would mean one extra round trip per
  // keystroke-driven save for no reason.
  let lastProbedFile;         // undefined = never probed (a pipeline with no model file is null, and still needs its first probe)

  async function refreshManifest() {
    const file = currentModelFile(slots);
    if (file === lastProbedFile) return false;
    const traits = await probeModelTraits(slots);
    const manifest = await API.modules(traits ? traits.join(",") : undefined);
    modulesById = {};
    (manifest.modules || []).forEach((m) => { modulesById[m.id] = m; });
    controlState = manifest.control || controlState;
    if (Array.isArray(manifest.sinks)) sinks = manifest.sinks;
    if (traits !== undefined) lastProbedFile = file;    // only once it worked: a failed probe is tried again
    loadError = null;                                   // the pipeline is in and its modules are known: nothing is failing now
    return true;
  }
  let incomplete = [];
  let refused = [];
  let queueable = false;
  let loading = false;
  let loadError = null;
  let unwired = {};           // default links the person cut: {slotId: [input, ...]}
  let removed = new Set();    // default slots the person took out of this project's pipeline
  let adoptGate = null;       // a promise while a project's pipeline is being put in
  let whole = false;          // the pipeline replaced the default (a preset, an import, the wizard) rather than editing it
  let adoptedOk = true;       // false while the live pipeline is a stand-in for one that could not go in
  let refusalLast = null;     // the server's own reason for refusing the last project's pipeline
  let unreachableLast = false; // the last project's pipeline failed because a request did not get through, not because it was refused
  let adopting = 0;           // adopts under way (waiting for the load, or putting a pipeline in)
  let deferred = null;        // a project's saved pipeline ([slots, removed, unwired, whole]) waiting for the first successful load
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
  // Once the server has named its sinks, only those are read, and EVERY module's entry in them is kept: a module the
  // current model hides still has its values there, and writing the tree back must not drop them.
  let sinks = null;           // [{node, input}] from the manifest; null until known (then the scan below is the fallback)
  const isSink = (slot, key) => sinks.some((k) => k.node === slot.node && k.input === key);
  function valuesAlreadyPlaced(from = slots) {
    const known = new Set([...Object.keys(modulesById), "_off"]);    // _off: the modules this project turned off
    const merged = {};
    (from || []).forEach((slot) => {
      Object.entries(slot.inputs || {}).forEach(([key, v]) => {
        if (typeof v !== "string" || (sinks && !isSink(slot, key))) return;
        let parsed;
        try { parsed = JSON.parse(v); } catch (_) { return; }
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return;
        Object.entries(parsed).forEach(([moduleId, own]) => {
          if (!sinks && !known.has(moduleId)) return;
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
    if (slots !== null && !loadError) return; // session-wide: every consumer shares this one load
    if (slots !== null) {                       // the pipeline came, its modules did not: ask for those again
      if (!loadPromise) loadPromise = refreshManifest().then(() => { loadPromise = null; _changed("load"); }, () => { loadPromise = null; });
      return loadPromise;
    }
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
      // An adopt already under way carries the newest project and lays it itself; waiting on it from here would deadlock.
      if (slots !== null && deferred) { const d = deferred; deferred = null; if (!adopting) await adopt(...d); }
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
  // Set by whoever ties the pipeline to a project: a reason when an edit now would land on a stand-in it could not keep.
  let editGuard = () => null;
  const blocked = (kind = "edit") => { const why = editGuard(kind); if (why) { saveNotes = [why]; _changed("load"); } return why; };    // said, and views redraw to what is true

  async function save(body) {
    if (adoptGate) await adoptGate;                     // an edit made while a project's pipeline goes in lands on THAT pipeline
    if (blocked()) return;
    pendingBody = mergeBodies(pendingBody, body);
    if (saving) { pending = true; return; }
    saving = true;
    do {
      pending = false;
      applyGroups();
      const toSend = { slots, ...pendingBody };
      pendingBody = null;
      const mine = epoch;
      try {
        const res = await API.editPipeline(toSend);
        // A different pipeline was put in while this was in flight (another project opened):
        // this answer is about the old one and must not overwrite the new.
        if (mine !== epoch) continue;
        if (res && res.slots) slots = res.slots;
        if (Object.keys(pendingValues).length) mirrorValues(currentValues());    // an edit queued behind this one stays in the slots a run reads
        applyGroups();
        incomplete = (res && res.incomplete) || [];
        refused = (res && res.refused) || [];
        queueable = !!(res && res.queueable);
        saveNotes = (res && res.notes) || [];
        _changed("edit");                               // now: the project this edit was made in takes it, even if another opens during the probe
        // A no-op for the common case (same file as last time) -- see
        // refreshManifest()'s own guard. Only a Models & Pipeline edit that
        // actually changes the loader's file does a second round trip here.
        // Its own try/catch: a failure here is not "could not save" -- the
        // edit above already landed -- so it must not overwrite saveNotes
        // with a message about the wrong failure.
        let fresh = false;
        try {
          fresh = await refreshManifest();
        } catch (e) {
          console.warn(`[FunPack] could not refresh modules for the new model: ${e && e.message ? e.message : e}`);
        }
        if (fresh && mine === epoch) _changed("edit");        // again once a new model's modules are known: a listener sees those
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
    if (saving) return;                         // a save still on its way may answer with older values: judge only once the queue is empty
    const already = valuesAlreadyPlaced();
    Object.keys(pendingValues).forEach((moduleId) => {
      const pend = pendingValues[moduleId];
      const real = already[moduleId] || {};
      Object.keys(pend).forEach((name) => {
        if (real[name] === pend[name] || (typeof pend[name] === "object" && JSON.stringify(real[name]) === JSON.stringify(pend[name]))) delete pend[name];
      });
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
  // `from`: a pipeline other than the live one (Generate's, frozen at the click); edits not yet landed then do not count.
  function currentValues(from) {
    const merged = {};
    Object.values(modulesById).forEach((m) => {
      const own = {};
      Object.entries(m.settings || {}).forEach(([name, spec]) => { own[name] = spec.default; });
      if (Object.keys(own).length) merged[m.id] = own;
    });
    const already = valuesAlreadyPlaced(from || slots);
    Object.entries(already).forEach(([moduleId, own]) => {
      merged[moduleId] = { ...(merged[moduleId] || {}), ...own };
    });
    Object.entries(from ? {} : pendingValues).forEach(([moduleId, own]) => {
      merged[moduleId] = { ...(merged[moduleId] || {}), ...own };
    });
    return merged;
  }

  // Which modules this project turned off (a reserved entry of the same settings tree), and which
  // the server has quarantined after a fault. Neither shows in Engine settings or the quick bar.
  let controlState = {};
  function isOff(id) {
    const off = (currentValues()._off || {}).modules;
    return Array.isArray(off) && off.includes(id);
  }
  function setOff(id, off) {
    const cur = new Set(((currentValues()._off || {}).modules) || []);
    if (off) cur.add(id); else cur.delete(id);
    return setModuleValue("_off", "modules", [...cur]);
  }
  // "Disable all enhancements": one project switch that makes every run plain, for A/B tests. The per-module
  // choices are kept underneath, so switching it back restores exactly what was on.
  const allOff = (from) => Boolean((currentValues(from)._off || {}).all);
  const setAllOff = (on) => setModuleValue("_off", "all", Boolean(on));
  // The screen is the truth: an edit is written into the live slots the moment it is made (the same blob the
  // server's place() writes), so a run started now reads what is shown without waiting for any round trip.
  function mirrorValues(values) {
    const known = new Set([...Object.keys(modulesById), "_off"]);
    (slots || []).forEach((slot) => {
      // a settings node just added starts with no input at all
      if (sinks) sinks.forEach((k) => { if (k.node === slot.node && !((slot.inputs ||= {})[k.input] !== undefined)) slot.inputs[k.input] = "{}"; });
      Object.entries(slot.inputs || {}).forEach(([key, v]) => {
        if (sinks) { if (isSink(slot, key) && !Array.isArray(v)) slot.inputs[key] = JSON.stringify(values); return; }    // a fresh "{}" too
        if (typeof v !== "string") return;
        let parsed;
        try { parsed = JSON.parse(v); } catch (_) { return; }
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return;
        if (Object.keys(parsed).some((id) => known.has(id))) slot.inputs[key] = JSON.stringify(values);
      });
    });
  }
  // A settings node with no settings yet (just added, swapped in, or from a snapshot taken before it had any):
  // given the project's values, or the run would not be queueable. -> whether one was.
  function fillEmptySinks(values) {
    const empty = (sinks || []).some((k) => (slots || []).some((s) => s.node === k.node && [undefined, ""].includes((s.inputs || {})[k.input])));
    if (empty) mirrorValues(values);
    return empty;
  }
  async function refreshControl() {
    try { controlState = (await API.modules()).control || {}; } catch (_) { /* keeps the last answer */ }
    return controlState;
  }
  function isQuarantined(id) { return !!(controlState[id] && controlState[id].quarantine); }
  // Whether anything would act on a module's settings now: its own node is in the pipeline (when it has one),
  // and every capability it offers (the modifier loader, the taste store) reaches a user that is switched on for this
  // project, in a mode that asks for it (its uses_when), has every node it needs in the pipeline, and is itself acted on (a taste learner
  // the loader never installs is not). "Disable all enhancements" is not counted: it sits on top, and the Modules page
  // still lists what is kept underneath it. -> a predicate; the values are read once for a whole list.
  function usefulness() {
    const values = currentValues(), list = (values._off || {}).modules, off = new Set(Array.isArray(list) ? list : []);
    const users = Object.values(modulesById).filter((u) => (u.uses || []).length && !off.has(u.id) && !isQuarantined(u.id));
    const has = (n) => !slots || slots.some((s) => s.node === n);
    const filled = (at) => !slots || slots.some((s) => (s.roles || []).some((r) => r.at === at));
    const check = (m, seen) => {
      const working = (u) => {
        const v = values[u.id] || {};
        return !seen.has(u.id) && (!("enabled" in (u.settings || {})) || v.enabled) && Object.entries(u.uses_when || {}).every(([k, want]) => (Array.isArray(want) ? want : [want]).includes(v[k]))
          && (u.nodes || []).every(has) && check(u, new Set([...seen, u.id]));
      };
      return (!(m.nodes || []).length || m.nodes.some(has)) && (!(m.roles || []).length || m.roles.some(filled))
        && (m.serves || []).every((cap) => users.some((u) => u.uses.includes(cap) && working(u)));
    };
    return (m) => check(m, new Set([m.id]));
  }
  const useful = (m) => usefulness()(m);

  function activeModules() {
    if (allOff()) return [];
    const list = (currentValues()._off || {}).modules, off = new Set(Array.isArray(list) ? list : []);      // read once, not once per module
    return Object.values(modulesById).filter((m) => !off.has(m.id) && !isQuarantined(m.id));
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
  async function setModuleValue(moduleId, name, value) {
    if (adoptGate) await adoptGate;                     // the values tree must be the pipeline this lands on, not the one going out
    if (blocked()) return;
    pendingValues[moduleId] = { ...(pendingValues[moduleId] || {}), [name]: value };
    const values = currentValues();
    mirrorValues(values);
    return save({ values }).then(_reconcilePending);
  }

  // A change to the pipeline's SHAPE (add / replace / remove / wire / unwire), sent on its own: two
  // of these folded into one request would lose the first. Waits for any value edit in flight and
  // for a project's pipeline going in. -> {refused: [...]}; empty when it happened.
  async function edit(body) {
    await ensureLoaded();
    if (slots === null) return { refused: ["The pipeline has not loaded: is ComfyUI reachable?"] };
    if (adoptGate) await adoptGate;
    { const why = blocked("shape"); if (why) return { refused: [why] }; }
    for (let waited = 0; saving && waited < 10000; waited += 20) await new Promise((r) => setTimeout(r, 20));
    if (saving) return { refused: ["A save is still in progress: try again in a moment."] };
    const keep = currentValues();                       // a settings node added or swapped in starts empty: it gets these
    let refill = false;
    const mine = epoch;
    saving = true;
    let refusedNow = [];
    try {
      applyGroups();
      const res = await API.editPipeline({ slots, ...body });
      if (mine !== epoch) return { refused: ["The project was changed while this was being sent."] };
      refusedNow = (res && res.refused) || [];
      if (res && res.slots && !refusedNow.length) { slots = res.slots; applyGroups(); }
      incomplete = (res && res.incomplete) || [];
      refused = refusedNow;
      queueable = !!(res && res.queueable);
      saveNotes = (res && res.notes) || [];
      if (!refusedNow.length) {
        if (body.action === "remove" && offered.some((s) => s.id === body.slot)) removed.add(body.slot);   // the person's own additions just go
        if (body.action === "add") (res.slots || []).forEach((s) => removed.delete(s.id));
        if (body.action === "unwire" && offered.some((s) => s.id === body.slot)) {
          unwired[body.slot] = [...new Set([...(unwired[body.slot] || []), body.input])];
        }
        if (body.action === "wire" && unwired[body.slot]) {
          unwired[body.slot] = unwired[body.slot].filter((k) => k !== body.input);
          if (!unwired[body.slot].length) delete unwired[body.slot];
        }
        if (body.action === "remove") { delete unwired[body.slot]; delete groupEdits[body.slot]; }
        // A default slot that was swapped for the same node, or taken out and put back, starts empty:
        // none of the default's links may come back over it when the project is opened again.
        const fresh = body.action === "replace" ? [body.slot] : body.action === "add" ? (res.slots || []).map((s) => s.id) : [];
        fresh.forEach((id) => {
          const def = offered.find((s) => s.id === id);
          if (!def) return;
          const links = Object.keys(def.inputs || {}).filter((k) => Array.isArray(def.inputs[k]));
          if (links.length) unwired[id] = [...new Set([...(unwired[id] || []), ...links])];
        });
        try { await refreshManifest(); } catch (_) { /* the next save retries it */ }
        refill = fillEmptySinks(keep);
        _changed("shape");                              // a change of shape: may replace a refused saved pipeline
      }
    } catch (e) {
      refusedNow = [e && e.message ? e.message : String(e)];
    } finally {
      saving = false;
    }
    if (refill || pending) await save({});              // the filled settings, or a value edit queued behind this one
    return { refused: refusedNow };
  }

  // The pipeline as it stands, to come back to: Models & Pipeline's Cancel puts this back.
  function snapshot() {
    if (slots === null) return null;
    return { slots: JSON.parse(JSON.stringify(slots)), removed: [...removed], unwired: JSON.parse(JSON.stringify(unwired)), whole };
  }
  // -> {refused: [...]}; empty when the snapshot is back. Announced as a change, so the project follows.
  async function restore(snap) {
    if (!snap || !Array.isArray(snap.slots)) return { refused: ["Nothing to go back to."] };
    // Already what is live: nothing to do. Above all not a change of shape, which would let Revert replace a refused saved pipeline.
    if (JSON.stringify(snap.slots) === JSON.stringify(slots)) return { refused: [] };
    if (adoptGate) await adoptGate;
    { const why = blocked("shape"); if (why) return { refused: [why] }; }
    for (let waited = 0; saving && waited < 10000; waited += 20) await new Promise((r) => setTimeout(r, 20));
    if (saving) return { refused: ["A save is still in progress: try again in a moment."] };
    const keep = currentValues();
    let refill = false;
    const mine = epoch;
    saving = true;
    let refusedNow = [];
    try {
      const res = await API.editPipeline({ slots: JSON.parse(JSON.stringify(snap.slots)) });
      if (mine !== epoch) return { refused: ["The project was changed while this was being sent."] };
      refusedNow = (res && res.refused) || [];
      if (res && res.slots && !refusedNow.length) {
        slots = res.slots;
        // A default slot the new pipeline lacks is removed, whatever the caller said: otherwise the next
        // open lays the default back over it (a preset, an import or the wizard would regain the default's loaders).
        const kept = new Set(res.slots.map((s) => s.id));
        removed = new Set([...(snap.removed || []), ...(offered || []).map((s) => s.id).filter((id) => !kept.has(id))]);
        whole = snap.whole !== false;                   // only a snapshot of an edited default says otherwise
        unwired = JSON.parse(JSON.stringify(snap.unwired || {}));
        groupEdits = {}; pendingValues = {}; pendingBody = null; pending = false;
        incomplete = res.incomplete || []; refused = []; queueable = !!res.queueable;
        saveNotes = res.notes || [];
        try { await refreshManifest(); } catch (_) { /* the next save retries it */ }
        refill = fillEmptySinks(keep);
        _changed("shape");                              // a change of shape: may replace a refused saved pipeline
      }
    } catch (e) {
      refusedNow = [e && e.message ? e.message : String(e)];
    } finally {
      saving = false;
    }
    if (refill) await save({});
    return { refused: refusedNow };
  }

  // A slot's group is the person's own to name (any node in any group).
  // Kept until it has landed: a save already in flight answers with the slots as they were when
  // it left, and would otherwise put the old group back.
  let groupEdits = {};
  function applyGroups() {
    (slots || []).forEach((s) => {
      if (!(s.id in groupEdits)) return;
      if (groupEdits[s.id]) s.group = groupEdits[s.id]; else delete s.group;
    });
  }
  function setGroup(slotId, group) {
    if (!(slots || []).some((s) => s.id === slotId)) return Promise.resolve();
    groupEdits[slotId] = String(group || "").trim();
    applyGroups();
    const mine = groupEdits[slotId];
    return save({}).then(() => { if (groupEdits[slotId] === mine) delete groupEdits[slotId]; });
  }

  // Whoever keeps the pipeline with the project hears every landed edit.
  const listeners = new Set();
  // kind: "edit" (a change the person made landed) or "load" (a project's pipeline went in, or the module list came).
  function _changed(kind = "edit") { listeners.forEach((fn) => { try { fn(slots, kind); } catch (e) { console.error(e); } }); }

  // Put a project's saved pipeline over the one the server offers: the person's inputs (model
  // files, node values, the module-settings blob), group and bypass win on every slot the
  // default still has; slots they added are kept. The server's own defaults fill in whatever
  // the saved copy predates, so an update's new settings are not lost to an old project.
  // Announced once it is in, so views redraw; the project link writes nothing while a project is going in.
  // True when the project's pipeline is in (or it had none to put in); false when it could not be.
  // `inherit`: a project with nothing saved keeps the live pipeline (a new one); otherwise it gets the default.
  async function adopt(saved, removedIds, unwiredMap, wholeSaved, inherit = true) {
    let open;
    const gate = new Promise((r) => { open = r; }), before = adoptGate;
    adoptGate = gate;                                   // before anything awaits: an edit right behind this waits too
    adopting += 1;
    deferred = null;                                    // an earlier project's pipeline waiting for the load is not this one's
    try {
      if (before) await before;                         // one at a time: a project opened behind another is put in after it, never under it
      await ensureLoaded();
      return await _adopt(saved, removedIds, unwiredMap, wholeSaved, inherit);
    } finally { adopting -= 1; if (adoptGate === gate) adoptGate = null; open(); _changed("load"); }
  }

  /** Resolves true once no edit is on its way and no project's pipeline is going in; false if that took over
   *  three minutes (every request gives up after ANSWER_MS, so only a server failing again and again gets there). */
  async function settled() {
    for (let waited = 0; (saving || adoptGate) && waited < 180000; waited += 20) await new Promise((r) => setTimeout(r, 20));
    return !(saving || adoptGate);
  }

  async function _adopt(saved, removedIds, unwiredMap, wholeSaved, inherit) {
    // Not loaded (ComfyUI unreachable): remember what the project holds and put it in the moment a
    // load succeeds, so the default is never what the next edit is built on.
    const hadSomething = (Array.isArray(saved) && saved.length) || (Array.isArray(removedIds) && removedIds.length);
    if (slots === null) { if (hadSomething) deferred = [saved, removedIds, unwiredMap, wholeSaved]; return (adoptedOk = !hadSomething); }
    if (!hadSomething && inherit) { saveNotes = []; return (adoptedOk = true); }      // a new project: no note about the last one's load
    if (!Array.isArray(saved)) saved = [];
    // A saved slot is only what the server would accept: a project file outlives the code that
    // wrote it (v4 files name the node differently), and one bad slot refuses every later edit.
    const sound = (s) => s && typeof s.id === "string" && typeof s.node === "string"
      && s.inputs && typeof s.inputs === "object" && !Array.isArray(s.inputs)
      && (s.group === undefined || (typeof s.group === "string" && s.group.trim()))
      && (s.roles === undefined || Array.isArray(s.roles));
    const byId = new Map(saved.filter(sound).map((s) => [s.id, s]));
    const have = new Set(offered.map((s) => s.id));
    // Slots the person took out stay out; ones whose node they swapped keep their swap.
    const gone = new Set(Array.isArray(removedIds) ? removedIds.filter((x) => typeof x === "string") : []);
    // A whole pipeline comes back exactly as saved: laid over the default it would regain the default's slots
    // and roles. Files saved before the flag existed: a whole one has slots of its own and lacks default ones
    // nobody removed. ponytail: an edited default saved before an update added a default slot reads as whole
    // too, and simply goes without that slot.
    const isWhole = typeof wholeSaved === "boolean" ? wholeSaved
      : [...byId.keys()].some((id) => !have.has(id)) && offered.some((d) => !gone.has(d.id) && !byId.has(d.id));
    const lay = (extras) => isWhole ? JSON.parse(JSON.stringify(saved.filter(sound))) : offered.filter((def) => !gone.has(def.id)).map((def) => {
      const mine = byId.get(def.id);
      if (!mine) return JSON.parse(JSON.stringify(def));
      if (mine.node !== def.node) return JSON.parse(JSON.stringify(mine));
      // A default LINK the saved copy lacks was unwired by the person; a default VALUE it lacks is
      // one an update added, and fills in.
      const inputs = { ...(def.inputs || {}) };
      const cut = (unwiredMap || {})[def.id];
      Object.keys(inputs).forEach((k) => { if (Array.isArray(cut) && cut.includes(k) && !(k in mine.inputs)) delete inputs[k]; });
      const out = { ...def, inputs: { ...inputs, ...mine.inputs } };
      if (typeof mine.group === "string" && mine.group) out.group = mine.group;
      if (typeof mine.bypassed === "boolean") out.bypassed = mine.bypassed;
      return out;
    }).concat(extras ? saved.filter((s) => sound(s) && !have.has(s.id)) : []);
    // Replaces whatever pipeline was live: nothing queued for the old one may land on the new.
    const mine = ++epoch;
    pendingValues = {};                                 // the new project's values win, not an edit made to the old one
    removed = new Set(gone);
    whole = isWhole;
    unwired = {};
    Object.entries(unwiredMap && typeof unwiredMap === "object" ? unwiredMap : {}).forEach(([id, list]) => {
      if (Array.isArray(list)) unwired[id] = list.filter((x) => typeof x === "string");
    });
    groupEdits = {};
    pendingBody = null; pending = false;
    for (let waited = 0; saving && waited < 10000; waited += 20) await new Promise((r) => setTimeout(r, 20));
    let unreachable = false, reasons = [];
    for (const extras of [true, false]) {
      try {
        const res = await API.editPipeline({ slots: lay(extras) });
        if (mine !== epoch) return true;                // a later project was opened meanwhile: not this one's to report
        if (res && res.slots && !(res.refused || []).length) {
          slots = res.slots;
          incomplete = res.incomplete || []; refused = []; queueable = !!res.queueable;
          saveNotes = res.notes || [];
          // The modules offered depend on the model this pipeline names, which may differ from the
          // previous project's: refresh, or the next settings edit drops this project's modules.
          try { await refreshManifest(); } catch (_) { /* the next save retries it */ }
          return (adoptedOk = true);
        }
        reasons = (res && res.refused) || reasons;
      } catch (e) {
        // No answer, or a server error (a tunnel, a restarting ComfyUI): try again later. A 4xx is a refusal.
        if (!e || !e.status || e.status >= 500) unreachable = true;
        else reasons = [e.message];
      }
    }
    unreachableLast = unreachable;
    refusalLast = reasons.length ? String(reasons[0]) : null;
    // Nothing saved was accepted (a failed request, or a file the server refuses): the default
    // runs, said, and the caller must NOT save it over the project's own copy.
    slots = JSON.parse(JSON.stringify(offered));
    removed = new Set(); unwired = {}; whole = false;   // the stand-in is the plain default: no flags of the project it could not load
    incomplete = []; refused = []; queueable = false;   // nor the last project's readiness
    saveNotes = ["This project's saved pipeline could not be loaded, so the default is in use."];
    return (adoptedOk = false);
  }

  return {
    ensureLoaded, save, edit, restore, snapshot, setGroup, adopt, settled, subscribe: (fn) => { listeners.add(fn); return () => listeners.delete(fn); }, valuesAlreadyPlaced, currentValues, setModuleValue,
    modulesById: () => modulesById,
    setEditGuard: (fn) => { editGuard = fn; }, adoptedOk: () => adoptedOk, unreachable: () => slots === null || unreachableLast, refusal: () => refusalLast, activeModules, useful, usefulness, isOff, setOff, allOff, setAllOff, refreshControl, control: () => controlState,
    removedIds: () => [...removed],
    whole: () => whole,
    unwiredMap: () => JSON.parse(JSON.stringify(unwired)),
    slots: () => slots,
    incomplete: () => incomplete,
    refused: () => refused,
    queueable: () => queueable,
    loading: () => loading,
    loadError: () => loadError,
    saving: () => saving,
    saveNotes: () => (modelNote ? [...saveNotes, modelNote] : saveNotes),
  };
}
