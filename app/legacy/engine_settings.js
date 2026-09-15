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

  const PS = window.PipelineState;
  let _mounted = null;
  let unsub = null;
  let category = null;      // selected sidebar category, chosen once modules load
  let seeded = false;        // whether `values` has been seeded from PS's slots yet
  let values = {};           // moduleId -> {settingName: value}, this panel's own draft

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

  // `values` is seeded once PipelineState has actually loaded -- from each
  // setting's own default, then whatever the graph already holds wins (see
  // PipelineState.valuesAlreadyPlaced's own comment for why that recovery
  // step exists at all). Runs at most once per session: PipelineState itself
  // only ever loads once, and reseeding on a second call would blow away
  // whatever the user already changed in THIS panel.
  function ensureSeeded() {
    if (seeded || PS.loading() || PS.loadError()) return;
    seeded = true;
    values = {};
    Object.values(PS.modulesById()).forEach((m) => {
      const own = {};
      Object.entries(m.settings || {}).forEach(([name, spec]) => { own[name] = spec.default; });
      if (Object.keys(own).length) values[m.id] = own;
    });
    const already = PS.valuesAlreadyPlaced();
    Object.entries(already).forEach(([moduleId, own]) => {
      values[moduleId] = { ...(values[moduleId] || {}), ...own };
    });
  }

  function commit() { PS.save({ values }).then(render); }

  function categoriesWithContent() {
    const found = new Set();
    Object.values(PS.modulesById()).forEach((m) => {
      if (Object.keys(m.settings || {}).length) found.add(m.category || "");
    });
    return CATEGORY_ORDER.filter((c) => found.has(c));
  }

  function renderPane(pane) {
    if (PS.loading()) { pane.append(hintEl("Loading…")); return; }
    if (PS.loadError()) {
      pane.append(hintEl(`Could not load Engine settings: ${PS.loadError()}`));
      return;
    }
    ensureSeeded();
    const cats = categoriesWithContent();
    if (!cats.length) {
      pane.append(hintEl(
        "No installed module exposes a setting yet. This fills in as model "
        + "and effect modules are built -- see the port plan."));
      return;
    }
    if (!category || !cats.includes(category)) category = cats[0];

    const modulesInCategory = Object.values(PS.modulesById())
      .filter((m) => (m.category || "") === category && Object.keys(m.settings || {}).length);

    const notes = PS.saveNotes();
    if (notes.length) pane.append(hintEl(notes.join(" ")));

    modulesInCategory.forEach((m) => {
      const g = group(pane, m.title || m.id);
      Object.entries(m.settings || {}).forEach(([name, spec]) => {
        if (!whenSatisfied(m.id, spec.when)) return;
        g.append(controlFor(m.id, name, spec));
      });
    });
  }

  function renderContent(container) {
    if (PS.loading() || PS.loadError() || !categoriesWithContent().length) {
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
    // PipelineState loads at most once per session and is shared with every
    // other panel that edits the pipeline -- closing and reopening this one
    // must not re-fetch over whatever is already held (see pipeline_state.js).
    PS.ensureLoaded().then(render);
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
