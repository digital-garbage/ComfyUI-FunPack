// Engine settings: a schema-driven renderer over v5's real module manifest
// (GET /funpack/api/modules) instead of v4's ~84 hardcoded Studio/Chain
// Sampler `kind:` declarations, which describe nodes that no longer exist.
// Same outer shell as before (SettingsWindow section, "engine"/"Generation"/
// order 1) so every existing entry point -- the inspector's "Engine settings
// →" button, the wheel pin, the tour -- keeps working unchanged; only the
// CONTENT is new. A category with nothing to show is simply absent from the
// sidebar: this tab looking emptier than v4's screenshot is it reflecting
// real backend state, not a bug (see the port plan).
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const API = window.MovieEditorAPI;

  const CATEGORY_LABELS = {
    continuity: "Continuity", guidance: "Guidance", conditioning: "Conditioning",
    sampling: "Sampling", post: "Post", system: "System", "": "Other",
  };
  const CATEGORY_ORDER = ["continuity", "guidance", "conditioning", "sampling", "post", "system", ""];

  let _mounted = null;
  let unsub = null;
  let category = null;      // selected sidebar category, chosen once modules load
  let loading = true;
  let loadError = null;
  let saving = false;
  let saveNotes = [];
  let modulesById = {};      // id -> manifest module (settings, title, category)
  let slots = null;          // current in-memory pipeline slots, re-sent on every edit
  let values = {};           // moduleId -> {settingName: value}, the UI's own draft

  function field(labelText, control, hint) {
    const row = el("div", "sw-row eng-field");
    const main = el("div", "sw-row-main");
    main.append(el("div", "sw-row-title", labelText));
    if (hint) main.append(el("div", "sw-row-hint", hint));
    row.append(main, control);
    return row;
  }
  function toggleField(labelText, checkbox, hint) {
    checkbox.style.width = "auto";
    return field(labelText, checkbox, hint);
  }
  function group(parent, label) {
    if (label) parent.append(el("div", "sw-rows-label", label));
    const g = el("div", "sw-rows");
    parent.append(g);
    return g;
  }
  function hintEl(text) { return el("div", "sw-hint", text); }

  // A `when` clause names sibling settings on the SAME module that must hold
  // their given value for this row to show -- e.g. a strength slider only
  // matters once its own "enabled" checkbox is on.
  function whenSatisfied(moduleId, when) {
    if (!when) return true;
    const current = values[moduleId] || {};
    return Object.entries(when).every(([k, v]) => current[k] === v);
  }

  function currentValue(moduleId, name, spec) {
    const v = values[moduleId] && values[moduleId][name];
    return v !== undefined ? v : spec.default;
  }
  function setValue(moduleId, name, v) {
    values[moduleId] = { ...(values[moduleId] || {}), [name]: v };
  }

  function controlFor(moduleId, name, spec) {
    const label = spec.label || name;
    const hint = spec.hint ? spec.hint + (spec.unit ? ` (${spec.unit})` : "") : (spec.unit || "");
    const value = currentValue(moduleId, name, spec);

    if (spec.type === "bool") {
      const cb = el("input", "");
      cb.type = "checkbox";
      cb.checked = !!value;
      cb.onchange = () => { setValue(moduleId, name, cb.checked); commit(); render(); };
      return toggleField(label, cb, spec.hint);
    }
    if (spec.type === "int" || spec.type === "float") {
      const useSlider = (spec.ui || "slider") === "slider" && spec.min != null && spec.max != null;
      const input = el("input", "eng-num-input");
      input.type = useSlider ? "range" : "number";
      if (spec.min != null) input.min = spec.min;
      if (spec.max != null) input.max = spec.max;
      if (spec.step != null) input.step = spec.step;
      input.value = value;
      const commitFn = () => {
        const n = spec.type === "int" ? parseInt(input.value, 10) : parseFloat(input.value);
        if (!Number.isNaN(n)) { setValue(moduleId, name, n); commit(); }
      };
      input.onchange = () => { commitFn(); render(); };
      return field(label, input, hint);
    }
    if (spec.type === "enum") {
      const select = el("select", "eng-select");
      (spec.options || []).forEach((opt) => {
        const o = el("option", "", opt.label || opt.value);
        o.value = opt.value;
        if (opt.value === value) o.selected = true;
        select.append(o);
      });
      select.onchange = () => { setValue(moduleId, name, select.value); commit(); render(); };
      return field(label, select, hint);
    }
    if (spec.type === "color") {
      const input = el("input", "");
      input.type = "color";
      input.value = value || "#000000";
      input.onchange = () => { setValue(moduleId, name, input.value); commit(); };
      return field(label, input, hint);
    }
    // text / multiline / path: a plain box. Committed on blur, not on every
    // keystroke -- a POST per character would be a request storm.
    const input = el(spec.type === "multiline" ? "textarea" : "input", "eng-text-input");
    if (spec.type !== "multiline") input.type = "text";
    input.value = value || "";
    input.onblur = () => { setValue(moduleId, name, input.value); commit(); };
    return field(label, input, hint);
  }

  // What a settings_sink already holds, read back out of the live slots --
  // place() (core/graph.py) writes the whole values blob as one opaque JSON
  // string into the sink's input, and never merges. Re-seeding `values` from
  // schema defaults alone (as this used to do) meant every mount's first edit
  // sent {the one field just touched} ∪ {everything else reset to default},
  // and place() blind-overwrote the sink with that -- silently discarding
  // every previously-set value across every module, on every reopen. Reading
  // the same JSON contract back out is the fix.
  //
  // No route tells the client WHICH node/input is the real sink (that is
  // core/graph.py's private business, by design -- core does not name an
  // implementation), so this can't look up the one true location and instead
  // scans every string input for one that decodes to a plain object. To keep
  // that honest rather than a coincidence machine: a decoded key is only
  // accepted when it names a module THIS SESSION ALREADY KNOWS IS INSTALLED
  // (`knownModuleIds`) -- an unrelated node whose string input happens to
  // parse as `{"tags": {...}}` cannot inject a bogus "tags" entry that then
  // round-trips forever through every future save.
  function valuesAlreadyPlaced(currentSlots, knownModuleIds) {
    const merged = {};
    (currentSlots || []).forEach((slot) => {
      Object.values(slot.inputs || {}).forEach((v) => {
        if (typeof v !== "string") return;
        let parsed;
        try { parsed = JSON.parse(v); } catch (_) { return; }
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return;
        Object.entries(parsed).forEach(([moduleId, own]) => {
          if (!knownModuleIds.has(moduleId)) return;
          if (own && typeof own === "object") merged[moduleId] = { ...(merged[moduleId] || {}), ...own };
        });
      });
    });
    return merged;
  }

  async function ensureLoaded() {
    loading = true; loadError = null;
    try {
      const [manifest, pipe] = await Promise.all([API.modules(), API.pipeline()]);
      modulesById = {};
      (manifest.modules || []).forEach((m) => { modulesById[m.id] = m; });
      slots = pipe.slots || [];
      // Seed from each setting's own default, then let whatever is already
      // placed in the graph win -- a field never placed before (a module
      // just installed, say) still needs a value from somewhere.
      values = {};
      Object.values(modulesById).forEach((m) => {
        const own = {};
        Object.entries(m.settings || {}).forEach(([name, spec]) => { own[name] = spec.default; });
        if (Object.keys(own).length) values[m.id] = own;
      });
      const already = valuesAlreadyPlaced(slots, new Set(Object.keys(modulesById)));
      Object.entries(already).forEach(([moduleId, own]) => {
        values[moduleId] = { ...(values[moduleId] || {}), ...own };
      });
    } catch (e) {
      loadError = e && e.message ? e.message : String(e);
    }
    loading = false;
  }

  // Edits arrive faster than a round-trip: a checkbox flipped while a slider's
  // request is still in flight must not be dropped. `values` is one shared,
  // synchronously-mutated object, so a save started after the in-flight one
  // finishes already sees every edit made in between -- `pending` just says
  // "go again" rather than trusting a single request to have carried it.
  let pending = false;

  async function commit() {
    if (saving) { pending = true; return; }
    saving = true;
    do {
      pending = false;
      try {
        const res = await API.editPipeline({ slots, values });
        if (res && res.slots) slots = res.slots;
        saveNotes = (res && res.notes) || [];
      } catch (e) {
        saveNotes = [`Could not save: ${e && e.message ? e.message : e}`];
      }
    } while (pending);
    saving = false;
    render();
  }

  function categoriesWithContent() {
    const found = new Set();
    Object.values(modulesById).forEach((m) => {
      if (Object.keys(m.settings || {}).length) found.add(m.category || "");
    });
    return CATEGORY_ORDER.filter((c) => found.has(c));
  }

  function renderPane(pane) {
    if (loading) { pane.append(hintEl("Loading…")); return; }
    if (loadError) {
      pane.append(hintEl(`Could not load Engine settings: ${loadError}`));
      return;
    }
    const cats = categoriesWithContent();
    if (!cats.length) {
      pane.append(hintEl(
        "No installed module exposes a setting yet. This fills in as model "
        + "and effect modules are built -- see the port plan."));
      return;
    }
    if (!category || !cats.includes(category)) category = cats[0];

    const modulesInCategory = Object.values(modulesById)
      .filter((m) => (m.category || "") === category && Object.keys(m.settings || {}).length);

    if (saveNotes.length) pane.append(hintEl(saveNotes.join(" ")));

    modulesInCategory.forEach((m) => {
      const g = group(pane, m.title || m.id);
      Object.entries(m.settings || {}).forEach(([name, spec]) => {
        if (!whenSatisfied(m.id, spec.when)) return;
        g.append(controlFor(m.id, name, spec));
      });
    });
  }

  function renderContent(container) {
    if (loading || loadError || !categoriesWithContent().length) {
      const solo = el("div", "models-pane");
      renderPane(solo);
      container.append(solo);
      return;
    }
    const cats = categoriesWithContent();
    const cols = el("div", "models-cols");
    const side = el("div", "models-side");
    cats.forEach((c) => {
      side.append(window.SettingsWindow.navItem({
        label: CATEGORY_LABELS[c] || c, active: category === c,
        onClick: () => { category = c; render(); },
      }));
    });
    const pane = el("div", "models-pane eng-pane");
    renderPane(pane);
    cols.append(side, pane);
    container.append(cols);
  }

  function render() {
    if (!_mounted) return;
    const { content } = _mounted;
    const prevPane = content.querySelector(".models-pane");
    const scrollTop = prevPane ? prevPane.scrollTop : 0;
    clear(content);
    renderContent(content);
    const pane = content.querySelector(".models-pane");
    if (pane) pane.scrollTop = scrollTop;
  }

  function mount(body) {
    const content = el("div", "models-mount eng-mount");
    body.append(content);
    _mounted = { content };
    render();
    // Only on the FIRST open this session: the server holds no pipeline state
    // of its own (GET /api/pipeline always returns the bare default -- see
    // valuesAlreadyPlaced()'s comment), so a fetch on every reopen would
    // throw away whatever this session already placed and never recover it.
    // The in-memory `slots`/`values` this module already holds ARE the
    // session's pipeline; closing and reopening the panel must not re-fetch
    // over them.
    if (slots === null) ensureLoaded().then(render);
    unsub = S.subscribe(() => {}); // kept: other sections rely on the same subscribe/unsub shape
    return () => {
      if (unsub) { unsub(); unsub = null; }
      _mounted = null;
    };
  }

  window.SettingsWindow.register({
    id: "engine", group: "Generation", order: 1, title: "Engine",
    subtitle: "Settings every installed module has volunteered, grouped by what they're for.",
    keywords: "engine module settings continuity guidance conditioning sampling post system",
    iconBg: "linear-gradient(180deg,#ffb64d,#e07f1f)",
    icon: '<svg viewBox="0 0 16 16" width="13" height="13"><path d="M9.2 1.3 3 9h4.1l-1 5.7L12.9 7H8.5l.7-5.7z" fill="#fff"/></svg>',
    mount,
    pinTarget: () => (category
      ? { kind: "section", id: "engine", sub: category, label: `Engine ▸ ${CATEGORY_LABELS[category] || category}` }
      : null),
  });

  window.EngineSettingsModal = {
    open: () => window.SettingsWindow.open("engine"),
    close: () => window.SettingsWindow.close(),
  };
})();
