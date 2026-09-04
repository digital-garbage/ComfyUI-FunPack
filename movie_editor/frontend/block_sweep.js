// H3 combo sweep: one scene, several REINS-steering / block-repeat configs run back to back
// and captured into a labelled gallery — "what option produced what output" at a glance.
// Each run is a one-off node_overrides on the built graph (store.runComboSweep); the
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

  // REINS reads a rating by SIGN only (direction() splits liked/disliked purely on
  // weight > 0 vs < 0, not magnitude) — the full nuanced label taxonomy is overkill for
  // this panel's purpose. These are the plain "H3 scale" digit strings the rating pipeline
  // already accepts (see conditioning.py's normalize_refiner_v2_rating: bare "1".."10" maps
  // through a linear reward, sign flips around ~5.6), picked well clear of that midpoint so
  // there is no ambiguity about which side of liked/disliked either button lands on.
  const _LIKE_RATING = "9";
  const _DISLIKE_RATING = "2";

  // Cards the user has already rated in THIS mount, so the row shows the outcome instead of
  // resetting to the buttons on the next re-render (S.subscribe re-renders the whole grid).
  const _rated = new WeakMap(); // item -> "liked" | "disliked"
  // Cards explicitly skipped, so the row doesn't keep asking. Local/cosmetic only — skipping
  // costs nothing and commits nothing, it just stops offering to rate that card.
  const _skipped = new WeakSet();

  function _card(item, onRated) {
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
    const metaEl = el("div", "pj-meta sw-sweep-seed", `seed ${item.seed}`);
    card.append(metaEl);
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

      // Each result with a REINS half saves its capture under its OWN slot (server-side,
      // h3_repr_capture_slot) — independent of every other result's, so EVERY card here is
      // rateable, in any order, any subset, same as the real batch training this panel
      // replaced. Liking/disliking/skipping is a direct commit against data already on
      // disk — no extra generation, unlike the old single-"pending" design this used to have.
      const outcome = _rated.get(item);
      if (outcome) {
        card.append(el("div", "es-hint sw-sweep-rate-note",
          outcome === "liked" ? "✓ Liked" : "✓ Disliked"));
      } else if (_skipped.has(item)) {
        card.append(el("div", "es-hint sw-sweep-rate-note", "Skipped"));
      } else if (item.slotId) {
        const rateRow = el("div", "sw-sweep-rate");
        const mkBtn = (cls, txt, title) => {
          const b = el("button", cls, txt); b.title = title; return b;
        };
        const likeBtn = mkBtn("btn ghost tiny", "👍 Like", "Commit a liked rating for this result.");
        const dislikeBtn = mkBtn("btn ghost tiny", "👎 Dislike", "Commit a disliked rating for this result.");
        const skipBtn = mkBtn("btn ghost tiny", "Skip", "Don't rate this one — discards its capture.");
        const setBusy = (busy) => { [likeBtn, dislikeBtn, skipBtn].forEach((b) => { b.disabled = busy; }); };
        const rate = async (label, outcomeLabel, btn) => {
          setBusy(true); btn.textContent = "…";
          try {
            const ok = await window.Store.rateComboResult(item, label);
            if (ok) { _rated.set(item, outcomeLabel); onRated && onRated(); }
            else { alert("Rating failed — the capture may have already been rated or discarded."); }
          } finally {
            setBusy(false);
            btn.textContent = btn === likeBtn ? "👍 Like" : "👎 Dislike";
          }
        };
        likeBtn.onclick = () => rate(_LIKE_RATING, "liked", likeBtn);
        dislikeBtn.onclick = () => rate(_DISLIKE_RATING, "disliked", dislikeBtn);
        skipBtn.onclick = async () => {
          setBusy(true);
          await window.Store.discardComboResult(item);
          _skipped.add(item); onRated && onRated();
        };
        rateRow.append(likeBtn, dislikeBtn, skipBtn);
        card.append(rateRow);
      }
    }
    return card;
  }

  // Module-level, not per-mount: this section is its own settings entry, so the config you
  // typed should survive closing and reopening the panel, not just live inside one mount()
  // closure.
  let _configText = "";
  let _sameSeed = false;

  function mount(container, ctx) {
    const S = window.Store;
    let sceneId = (_activeScenes()[0] || {}).id || null;

    container.append(el("div", "es-hint",
      "Runs ONE scene through several REINS / block-repeat configs back to back, so results "
      + "are directly comparable. Nothing here touches your Engine Settings — every run is a "
      + "one-off override on that run only."));

    const sceneRow = el("div", "sw-stack");
    const sceneSel = el("select");
    _activeScenes().forEach((s, i) => sceneSel.append(new Option(_sceneLabel(s, i), s.id)));
    if (sceneId) sceneSel.value = sceneId;
    sceneSel.onchange = () => { sceneId = sceneSel.value; };
    sceneRow.append(el("label", "sw-label", "Scene to sweep"), sceneSel);
    container.append(sceneRow);

    const seedCb = el("input"); seedCb.type = "checkbox"; seedCb.style.width = "auto";
    seedCb.checked = _sameSeed;
    seedCb.onchange = () => { _sameSeed = seedCb.checked; };
    const seedLabel = el("label", "sw-hint sw-check-label");
    seedLabel.append(seedCb, document.createTextNode(
      " Same seed for every line (off = each line gets its own fresh random seed, regardless "
      + "of your Engine Settings seed)"));
    container.append(seedLabel);

    const cfgWrap = el("div", "sw-stack");
    const cfgHint = el("div", "es-hint",
      "One config per line, combining either or both with \"|\": reins:strength;block and/or "
      + "sweep:blocks;seam|noseam;times[;laststeps] — e.g. reins:0.15;49|sweep:40-41;noseam;1;0 "
      + "or just reins:0.1;49 alone. Whichever half you omit is explicitly OFF for that line, "
      + "not \"whatever Engine Settings has\". sweep's blocks accept a single number, a range "
      + "(31-40), or a comma list; times is clamped to 1-4; laststeps confines the repeat to "
      + "part of the schedule (0/omitted = every step, positive = final N steps, negative = "
      + "first |N| steps then stop). Unvalidated — results will not match any prior sweep 1:1, "
      + "see the block-repeat research notes.");
    const cfgArea = document.createElement("textarea");
    cfgArea.className = "sw-textarea";
    cfgArea.rows = 6;
    cfgArea.placeholder = "reins:0.15;49|sweep:40-41;noseam;1;0\nreins:0.1;49\nsweep:31-40;noseam;1;-3";
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
      const cs = S.get().comboSweep || {};
      runBtn.disabled = !!cs.running;
      runBtn.textContent = cs.running ? `Running ${cs.current || 0}/${cs.total || 0}…` : "▶ Run sweep";
      const seedNote = cs.sameSeed ? `seed ${cs.seed} (fixed)` : "each line its own seed";
      statusEl.textContent = cs.running
        ? `Sweeping (${seedNote}): ${cs.label || ""}`
        : cs.error
          ? "Stopped: " + cs.error
          : (cs.results && cs.results.length ? `${cs.results.length} result(s), ${seedNote}` : "");
      clear(grid);
      const results = cs.results || [];
      results.forEach((item) => grid.append(_card(item, render)));
    }

    runBtn.onclick = () => {
      if (!sceneId) { alert("No scene to sweep."); return; }
      S.runComboSweep(sceneId, _configText, _sameSeed);
    };

    const unsub = S.subscribe(render);
    render();
    return unsub;
  }

  window.SettingsWindow.register({
    id: "combo_sweep", group: "Learning", order: 1, title: "H3 Combo Sweep",
    subtitle: "Run one scene through several REINS / block-repeat configs, compare results side by side.",
    keywords: "h3 reins representation steering block repeat sweep span loop seam experimental research gallery batch rate",
    iconBg: "linear-gradient(180deg,#b48bff,#6d3fd8)",
    icon: '<svg viewBox="0 0 16 16" width="13" height="13"><path d="M2 3h4v4H2zM10 3h4v4h-4zM2 9h4v4H2zM10 9h4v4h-4z" fill="#fff"/></svg>',
    mount,
  });
})();
