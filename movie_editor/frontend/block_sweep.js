// H3 block-repeat sweep: one scene, one fixed seed, several repeat configs run back to back
// and captured into a labelled gallery — "what option produced what output" at a glance.
// Each run is a one-off node_overrides on the built graph (store.runBlockRepeatSweep); the
// project's own Engine Settings are never touched. A section of the unified Settings window,
// sibling to Refinement & Taste — this IS the research tooling that rule asks for.
(function () {
  const { el, clear } = window.dom;
  const API = window.MovieEditorAPI;

  function _activeScenes() {
    const st = window.Store.get();
    return (st.project?.scenes || []).filter((s) => !s.excluded);
  }

  function _sceneLabel(s, i) {
    return `Scene ${i + 1}` + (s.text ? ": " + s.text.substring(0, 30) : "");
  }

  function _card(item) {
    const card = el("div", "media-card tmp-card");
    const thumb = el("div", "media-thumb");
    if (item.media) {
      const vid = document.createElement("video");
      vid.src = API.resultUrl(window.Store.get().project.id, item.media);
      vid.controls = true; vid.muted = true; vid.playsInline = true; vid.preload = "metadata";
      thumb.append(vid);
    } else {
      thumb.append(el("span", "media-icon", "✕"));
    }
    card.append(thumb);
    const nameEl = el("div", "media-name", item.label);
    nameEl.title = item.label;
    card.append(nameEl);
    return card;
  }

  function mount(container, ctx) {
    const S = window.Store;
    let sceneId = (_activeScenes()[0] || {}).id || null;
    let configText = "";

    container.append(el("div", "es-hint",
      "Runs ONE scene through several H3 block-repeat configs back to back, same frame and "
      + "seed throughout, so results are directly comparable. Nothing here touches your Engine "
      + "Settings — every run is a one-off override on that run only."));

    const sceneRow = el("div", "sw-stack");
    const sceneSel = el("select");
    _activeScenes().forEach((s, i) => sceneSel.append(new Option(_sceneLabel(s, i), s.id)));
    if (sceneId) sceneSel.value = sceneId;
    sceneSel.onchange = () => { sceneId = sceneSel.value; };
    sceneRow.append(el("label", "sw-label", "Scene to sweep"), sceneSel);
    container.append(sceneRow);

    const expCb = el("input"); expCb.type = "checkbox";
    const expLabel = el("label", "sw-hint");
    expLabel.append(expCb, document.createTextNode(
      " Show experimental config — unvalidated, results will not match any prior sweep 1:1 "
      + "(different checkpoint/steps/sampler port nothing over, see the block-repeat research notes)"));
    container.append(expLabel);

    const cfgWrap = el("div", "sw-stack");
    cfgWrap.hidden = true;
    const cfgHint = el("div", "es-hint",
      "One config per line: blocks;seam|noseam;times — e.g. 10,11,13;seam;5  or  40-41;noseam;1. "
      + "Blocks accept a single number, a range (31-40), or a comma list; times is clamped to 1-4.");
    const cfgArea = document.createElement("textarea");
    cfgArea.className = "sw-textarea";
    cfgArea.rows = 6;
    cfgArea.placeholder = "10,11,13;seam;5\n40-41;noseam;1";
    cfgArea.oninput = () => { configText = cfgArea.value; };
    const runBtn = el("button", "btn primary tiny", "▶ Run sweep");
    cfgWrap.append(cfgHint, cfgArea, runBtn);
    container.append(cfgWrap);
    expCb.onchange = () => { cfgWrap.hidden = !expCb.checked; };

    const statusEl = el("div", "pj-meta");
    container.append(statusEl);
    const grid = el("div", "media-grid tmp-grid");
    container.append(grid);

    function render() {
      const bs = S.get().blockSweep || {};
      runBtn.disabled = !!bs.running;
      runBtn.textContent = bs.running ? `Running ${bs.current || 0}/${bs.total || 0}…` : "▶ Run sweep";
      statusEl.textContent = bs.running
        ? `Sweeping (seed ${bs.seed}): ${bs.label || ""}`
        : bs.error
          ? "Stopped: " + bs.error
          : (bs.results && bs.results.length ? `${bs.results.length} result(s), seed ${bs.seed}` : "");
      clear(grid);
      (bs.results || []).forEach((item) => grid.append(_card(item)));
    }

    runBtn.onclick = () => {
      if (!sceneId) { alert("No scene to sweep."); return; }
      S.runBlockRepeatSweep(sceneId, configText);
    };

    const unsub = S.subscribe(render);
    render();
    return unsub;
  }

  window.SettingsWindow.register({
    id: "block_sweep", group: "Learning", order: 1, title: "H3 Block Repeat Sweep",
    subtitle: "Run one scene through several block-repeat configs, compare results side by side.",
    keywords: "h3 block repeat sweep span loop seam experimental research gallery batch",
    iconBg: "linear-gradient(180deg,#b48bff,#6d3fd8)",
    icon: '<svg viewBox="0 0 16 16" width="13" height="13"><path d="M2 3h4v4H2zM10 3h4v4h-4zM2 9h4v4H2zM10 9h4v4h-4z" fill="#fff"/></svg>',
    mount,
  });
})();
