// Engine settings: a schema-driven renderer over v5's real module manifest
// (GET /funpack/api/modules) instead of v4's ~84 hardcoded Studio/Chain
// Sampler `kind:` declarations, which describe nodes that no longer exist.
// Same outer shell as before (SettingsWindow section, "engine"/"Generation"/
// order 1) so every existing entry point -- the inspector's "Engine settings
// →" button, the wheel pin, the tour -- keeps working unchanged; only the
// CONTENT is new. A category with nothing to show is simply absent from the
// sidebar: this tab looking emptier than v4's screenshot is it reflecting
// real backend state, not a bug (see the port plan).
//
// The field-widget builder and the module-values tree it reads/writes both
// live outside this file now (settings_field.js, PipelineState.currentValues/
// setModuleValue) -- the sampler quick-access bar (sampler_quickbar.js)
// edits the SAME tree through the SAME functions, so there is exactly one
// place that knows how to render a setting and exactly one place that knows
// how to save one.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const API = window.MovieEditorAPI;
  const SF = window.SettingsField;

  const CATEGORY_LABELS = {
    continuity: "Continuity", guidance: "Guidance", conditioning: "Conditioning",
    sampling: "Sampling", post: "Post", system: "System", "": "Other",
  };
  const CATEGORY_ORDER = ["continuity", "guidance", "conditioning", "sampling", "post", "system", ""];

  const PS = window.PipelineState;
  let _mounted = null;
  let unsub = null;
  let category = null;      // selected sidebar category, chosen once modules load

  function onChange(moduleId, name, value) {
    PS.setModuleValue(moduleId, name, value).then(render);
  }

  function categoriesWithContent() {
    const found = new Set();
    Object.values(PS.modulesById()).forEach((m) => {
      if (Object.keys(m.settings || {}).length) found.add(m.category || "");
    });
    return CATEGORY_ORDER.filter((c) => found.has(c));
  }

  function renderPane(pane) {
    if (PS.loading()) { pane.append(SF.hintEl("Loading…")); return; }
    if (PS.loadError()) {
      pane.append(SF.hintEl(`Could not load Engine settings: ${PS.loadError()}`));
      return;
    }
    const cats = categoriesWithContent();
    if (!cats.length) {
      pane.append(SF.hintEl(
        "No installed module exposes a setting yet. This fills in as model "
        + "and effect modules are built -- see the port plan."));
      return;
    }
    if (!category || !cats.includes(category)) category = cats[0];

    const modulesInCategory = Object.values(PS.modulesById())
      .filter((m) => (m.category || "") === category && Object.keys(m.settings || {}).length);

    const notes = PS.saveNotes();
    if (notes.length) pane.append(SF.hintEl(notes.join(" ")));

    const values = PS.currentValues();
    modulesInCategory.forEach((m) => {
      const g = SF.group(pane, m.title || m.id);
      Object.entries(m.settings || {}).forEach(([name, spec]) => {
        if (!SF.whenSatisfied(values, m.id, spec.when)) return;
        g.append(SF.controlFor(m.id, name, spec, values, onChange));
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
