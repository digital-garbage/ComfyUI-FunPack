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

  // "10,11,13;seam;5" -> "sweep_10-11-13_seam_5_<ts>.mp4" — the raw config line, filesystem-safe,
  // timestamped so re-running the same line twice doesn't collide in the Media bin.
  function _sweepFilename(label) {
    const safe = String(label).replace(/[^A-Za-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
    return `sweep_${safe}_${Date.now()}.mp4`;
  }

  function _card(item) {
    // A real viewing size, not the media-bin's thumbnail height — the whole point of this
    // gallery is comparing results without popping each one into a separate tab.
    const card = el("div", "sw-sweep-card");
    if (item.media) {
      const vid = document.createElement("video");
      vid.src = API.resultUrl(window.Store.get().project.id, item.media);
      vid.controls = true; vid.muted = true; vid.playsInline = true; vid.preload = "metadata";
      card.append(vid);
    } else {
      card.append(el("div", "sw-sweep-fail", "✕ run failed"));
    }
    const nameEl = el("div", "media-name", item.label);
    nameEl.title = item.label;
    card.append(nameEl);
    if (item.media) {
      const saveBtn = el("button", "btn ghost tiny sw-sweep-save", "💾 Save to Media bin");
      saveBtn.onclick = async () => {
        saveBtn.disabled = true; saveBtn.textContent = "Saving…";
        try {
          await API.importClipToMediaBin({
            filename: item.media.filename,
            subfolder: item.media.subfolder || "",
            type: item.media.type || "output",
          }, _sweepFilename(item.label));
          await window.Store.loadMedia();
          saveBtn.textContent = "✓ Saved";
        } catch (e) {
          alert("Save failed: " + (e.message || e));
          saveBtn.textContent = "💾 Save to Media bin";
        } finally {
          saveBtn.disabled = false;
        }
      };
      card.append(saveBtn);
    }
    return card;
  }

  // Module-level, not per-mount: this section is its own settings entry (not something an
  // "experimental" checkbox needs to gate), so the config you typed should survive closing
  // and reopening the panel, not just live inside one mount() closure.
  let _configText = "";

  function mount(container, ctx) {
    const S = window.Store;
    let sceneId = (_activeScenes()[0] || {}).id || null;

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

    const cfgWrap = el("div", "sw-stack");
    const cfgHint = el("div", "es-hint",
      "One config per line: blocks;seam|noseam;times[;laststeps] — e.g. 10,11,13;seam;5  or  "
      + "40-41;noseam;1;2. Blocks accept a single number, a range (31-40), or a comma list; "
      + "times is clamped to 1-4. laststeps confines the repeat to the final N denoise steps "
      + "(0 or omitted = every step); mixing bare and laststeps-tagged lines in one sweep is "
      + "fine but they are not comparable to each other. Unvalidated — results will not match "
      + "any prior sweep 1:1 (different checkpoint/steps/sampler port nothing over, see the "
      + "block-repeat research notes).");
    const cfgArea = document.createElement("textarea");
    cfgArea.className = "sw-textarea";
    cfgArea.rows = 6;
    cfgArea.placeholder = "10,11,13;seam;5\n40-41;noseam;1;2";
    cfgArea.value = _configText;
    cfgArea.oninput = () => { _configText = cfgArea.value; };
    const runBtn = el("button", "btn primary tiny", "▶ Run sweep");
    cfgWrap.append(cfgHint, cfgArea, runBtn);
    container.append(cfgWrap);

    const statusEl = el("div", "pj-meta");
    container.append(statusEl);
    const grid = el("div", "sw-sweep-grid");
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
      S.runBlockRepeatSweep(sceneId, _configText);
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
