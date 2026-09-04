// Unified Settings window, macOS System Settings style: a sidebar of grouped,
// searchable sections (colored icon chips) + one content pane. Sections self-register
// via SettingsWindow.register(); the legacy per-modal facades (EditorSettingsModal,
// EngineSettingsModal, ModelsModal, TempBrowserModal) deep-link into their section,
// so every existing entry point (menus, inspector, workflow import) keeps working.
(function () {
  const { el, clear } = window.dom;

  const GROUPS = ["", "Generation", "Learning", "System"];
  const registry = [];
  let overlay = null, activeId = null, cleanup = null, query = "";
  let navEl, bodyEl, titleEl, subEl, actionsEl;

  function register(spec) {
    registry.push(spec);
    if (overlay) renderNav();
  }

  // Sidebar order: group order, then explicit spec.order, then registration order.
  function orderedSections() {
    const out = [];
    GROUPS.forEach((g) => {
      out.push(...registry
        .filter((s) => (s.group || "") === g)
        .sort((a, b) => (a.order ?? 99) - (b.order ?? 99)));
    });
    return out;
  }

  function matches(spec) {
    if (!query) return true;
    return (spec.title + " " + (spec.subtitle || "") + " " + (spec.keywords || ""))
      .toLowerCase().includes(query);
  }

  function iconEl(spec) {
    const w = el("span", "sw-icon");
    if (spec.iconBg) w.style.background = spec.iconBg;
    w.innerHTML = spec.icon || "";
    return w;
  }

  function renderNav() {
    clear(navEl);
    const ordered = orderedSections();
    GROUPS.forEach((g) => {
      const items = ordered.filter((s) => (s.group || "") === g && matches(s));
      if (!items.length) return;
      if (g) navEl.append(el("div", "sw-group-title", g));
      items.forEach((s) => {
        const it = el("div", "sw-item" + (s.id === activeId ? " active" : ""));
        it.append(iconEl(s), el("span", "sw-item-label", s.title));
        it.onclick = () => show(s.id);
        navEl.append(it);
      });
    });
    if (!navEl.children.length) navEl.append(el("div", "sw-nav-empty", "No matches"));
  }

  function teardownSection() {
    if (typeof cleanup === "function") { try { cleanup(); } catch (_) {} }
    cleanup = null;
  }

  // `sub` names a place INSIDE a section (an Engine category, say). Sections that have
  // inner views read it from ctx on mount; the rest ignore it harmlessly.
  function show(id, subView) {
    const spec = registry.find((s) => s.id === id) || orderedSections()[0];
    if (!spec || !overlay) return;
    teardownSection();
    activeId = spec.id;
    titleEl.textContent = spec.title;
    subEl.textContent = spec.subtitle || "";
    subEl.hidden = !spec.subtitle;
    clear(actionsEl);
    clear(bodyEl);
    // flush sections manage their own inner layout/scrolling (e.g. Models' sidebar)
    bodyEl.classList.toggle("sw-flush", !!spec.flush);
    bodyEl.scrollTop = 0;
    const ctx = {
      setActions: (nodes) => { clear(actionsEl); (nodes || []).forEach((n) => actionsEl.append(n)); },
      openSection: (sid) => show(sid),
      sub: subView || null,
    };
    cleanup = spec.mount(bodyEl, ctx) || null;
    renderNav();
  }

  function close() {
    const wasOpen = !!overlay;
    teardownSection();
    if (overlay) overlay.remove();
    overlay = null; activeId = null;
    // One save on the way out, with the values on screen. A save landing mid-edit used
    // to come back and overwrite the knob under the cursor.
    if (wasOpen) window.Store?.resumeSave?.();
  }

  function open(id, subView) {
    // Recover if something removed our overlay from the DOM directly
    // (e.g. the tour's closeModalOverlay sweep) without calling close().
    if (overlay && !overlay.isConnected) { close(); }
    if (overlay) { show(id || activeId, subView); return; }
    query = "";
    // Held until close(). Paired with the overlay's lifetime, so only on this branch.
    window.Store?.suspendSave?.();

    overlay = el("div", "modal-overlay sw-overlay");
    const win = el("div", "settings-win");

    const side = el("div", "sw-side");
    const search = el("input", "sw-search");
    search.type = "search"; search.placeholder = "Search"; search.autocomplete = "off";
    search.oninput = () => { query = search.value.trim().toLowerCase(); renderNav(); };
    navEl = el("div", "sw-nav");
    side.append(search, navEl);

    const main = el("div", "sw-main");
    const head = el("div", "sw-head");
    const ht = el("div", "sw-head-text");
    titleEl = el("div", "sw-head-title");
    subEl = el("div", "sw-head-sub");
    ht.append(titleEl, subEl);
    actionsEl = el("div", "sw-head-actions");
    const pin = el("button", "btn ghost tiny sw-pin", "📌 Pin to a button");
    pin.type = "button";
    pin.title = "Put a shortcut to whatever is open here on the timeline toolbar";
    pin.onclick = () => window.PinnedButtons?.pinCurrent();
    const x = el("button", "btn ghost tiny sw-close", "✕");
    x.onclick = close;
    head.append(ht, actionsEl, pin, x);
    if (!window.PinnedButtons) pin.hidden = true;
    bodyEl = el("div", "sw-body");
    main.append(head, bodyEl);

    win.append(side, main);
    overlay.append(win);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.body.append(overlay);
    show(id, subView);
  }

  window.addEventListener("keydown", (e) => {
    if (overlay && e.key === "Escape") { close(); return; }
    if ((e.metaKey || e.ctrlKey) && !e.altKey && e.key === ",") {
      e.preventDefault();
      overlay ? close() : open();
    }
  });

  // Shared builder for inner-sidebar nav items (Models node list, Engine categories):
  // { icon | dot, label, sub, badge, active, onClick } → .mn-item element.
  function navItem(opts) {
    const it = el("div", "mn-item" + (opts.active ? " active" : ""));
    if (opts.dot) it.append(el("span", "mn-dot " + opts.dot));
    else if (opts.icon) it.append(el("span", "mn-ico", opts.icon));
    const lab = el("span", "mn-label");
    lab.append(el("span", "mn-name", opts.label));
    if (opts.sub) lab.append(el("span", "mn-sub", opts.sub));
    it.append(lab);
    if (opts.badge != null) it.append(el("span", "mn-badge", String(opts.badge)));
    it.onclick = opts.onClick;
    return it;
  }

  // Mount ONE registered section into an arbitrary container — no sidebar, no search, no
  // window chrome. Used by the setup wizard to show a section as a screen of its own.
  // Returns { spec, cleanup }, or null when nothing has registered that id.
  function mountSection(id, host, opts = {}) {
    const spec = registry.find((s) => s.id === id);
    if (!spec) return null;
    host.classList.toggle("sw-flush", !!spec.flush);
    const cleanup = spec.mount(host, {
      setActions: opts.setActions || (() => {}),
      openSection: opts.openSection || (() => {}),
    });
    return { spec, cleanup: () => { if (typeof cleanup === "function") { try { cleanup(); } catch (_) {} } } };
  }

  function hasSection(id) { return registry.some((s) => s.id === id); }

  /** What a pinned button should reopen to get back to what is on screen now.
   *
   * A section can override this (Models points at the open NODE rather than at itself),
   * which is the whole value of the feature: the deep places are the slow ones to reach.
   */
  function currentTarget() {
    const spec = registry.find((s) => s.id === activeId);
    if (!spec) return null;
    let custom = null;
    try { custom = typeof spec.pinTarget === "function" ? spec.pinTarget() : null; } catch (_) {}
    return custom || { kind: "section", id: spec.id, label: spec.title };
  }

  /** Every section, for the pin dialog's "somewhere else" list. */
  function sectionList() {
    return orderedSections().map((s) => ({ id: s.id, title: s.title, group: s.group || "" }));
  }

  window.SettingsWindow = {
    open, close, register, navItem, mountSection, hasSection, currentTarget, sectionList,
  };

  // ── shared row builders for settings panels ────────────────────────────
  function actionRow(title, hint, btnLabel, onClick, opts = {}) {
    const row = el("div", "sw-row");
    const main = el("div", "sw-row-main");
    main.append(el("div", "sw-row-title", title));
    if (hint) main.append(el("div", "sw-row-hint", hint));
    row.append(main);
    const btn = el("button", "btn ghost tiny" + (opts.danger ? " danger" : ""), btnLabel);
    btn.type = "button";
    if (opts.disabled) btn.disabled = true;
    btn.onclick = onClick;
    row.append(btn);
    return row;
  }

  // ── Trajectory probe ───────────────────────────────────────────────────
  // Every rating-driven mechanism only acts over the last half of a generation, so a
  // rating about MOTION arrives after motion was decided. The probe records what the model
  // was predicting at several points through each generation and then says whether your
  // good and bad ratings actually look different EARLY on — which is what would have to be
  // true for steering to be worth extending back there.
  //
  // Recording costs nothing measurable and steers nothing. It needs a refinement key, since
  // the rating has to pair with the run that earned it.
  let probeState = null, probeReport = null, probeBusy = false, probeError = "", probeNote = "";

  function probeToggleRow(enabled, onChange) {
    const row = el("div", "sw-row");
    const main = el("div", "sw-row-main");
    main.append(el("div", "sw-row-title", "Record what each generation predicts"));
    main.append(el("div", "sw-row-hint",
      "Off by default. Recording only — nothing about your generations changes. "
      + "Set a refinement key and rate as usual; each rated generation adds one run."));
    row.append(main);
    const lbl = el("label", "chk es-toggle");
    const cb = el("input");
    cb.type = "checkbox";
    cb.checked = !!enabled;
    cb.onchange = () => onChange(cb.checked);
    lbl.append(cb, el("span", null, ""));
    row.append(lbl);
    return row;
  }

  function probeVerdict(report) {
    const box = el("div", "sw-hint");
    if (!report || !report.buckets || !report.buckets.length) {
      box.textContent = "Nothing recorded yet.";
      return box;
    }
    box.textContent = report.verdict === "early"
      ? "Ratings separate EARLY — before anything currently steers. Extending steering back "
        + "there has something to learn."
      : "No early separation beyond what shuffling the ratings produces. Check the late rows "
        + "first: if those are blank too, this is too few runs (or every run rated alike), "
        + "not an answer about the model.";
    return box;
  }

  function probeTable(report) {
    const tbl = el("div", "sw-rows");
    report.buckets.forEach((b) => {
      const p = b.pooled;
      const when = b.early ? "Early — nothing steers here yet" : "Late — already steered";
      const verdict = !p ? "not enough runs"
        : (p.p_value <= report.threshold ? "SEPARATES" : "no signal");
      const detail = p
        ? `${verdict} · p ${p.p_value.toFixed(4)} · ${b.good} good / ${b.bad} bad`
        : `${verdict} · ${b.good} good / ${b.bad} bad`;
      tbl.append(infoRow(`Part ${b.bucket + 1} of ${report.buckets.length} — ${when}`,
                         detail, p ? p.p_value <= report.threshold : null));
    });
    return tbl;
  }

  function probeRows() {
    const box = el("div", "sw-stack");
    // A plain file input, clicked by the row's button. The browser's own picker is
    // the whole feature; a drop zone would be more code for the same result. Built
    // once, outside paint(), so a redraw cannot discard a pick in progress.
    const probeFile = el("input");
    probeFile.type = "file";
    probeFile.accept = ".pt";
    probeFile.style.display = "none";
    probeFile.onchange = async () => {
      const file = probeFile.files && probeFile.files[0];
      probeFile.value = "";                  // so the same file can be picked twice
      if (!file) return;
      probeError = "";
      try {
        const res = await window.MovieEditorAPI.probeImport(file);
        probeState = res;
        probeReport = null;                  // the old reading predates these runs
        probeNote = `${res.added} run(s) added; ${res.runs} recorded now.`;
      } catch (e) { probeError = String(e.message || e); }
      paint();
    };
    const paint = () => {
      clear(box);
      const rows = el("div", "sw-rows");
      rows.append(probeToggleRow(probeState && probeState.enabled, async (on) => {
        probeError = "";
        try {
          probeState = await window.MovieEditorAPI.probeSetEnabled(on);
        } catch (e) { probeError = String(e.message || e); }
        paint();
      }));
      const runs = probeState ? probeState.runs : 0;
      rows.append(infoRow("Rated generations recorded", String(runs), runs > 0 ? true : null));
      rows.append(actionRow("Save the measurement",
        "Downloads every recorded run as one file. A rental is replaced when something "
        + "breaks and this data is not in git, so without this the count restarts on the "
        + "next box and the reading is never reached.",
        "Download", async () => {
          probeError = ""; 
          try { await window.MovieEditorAPI.probeExport(); }
          catch (e) { probeError = String(e.message || e); paint(); }
        }, { disabled: !runs }));
      rows.append(actionRow("Load a saved measurement",
        "Adds runs from a file onto this box. Runs already here are skipped, so importing "
        + "the same file twice cannot inflate the count.",
        "Load…", () => probeFile.click()));
      rows.append(actionRow("Start a fresh measurement",
        "Throws away every recorded run AND what was learned from them, so the next test "
        + "starts from nothing. Recording stays on. Cannot be undone \u2014 download first "
        + "if you want to keep it.",
        "Clear\u2026", async () => {
          if (!window.confirm(`Throw away ${runs} recorded run(s) and everything learned `
              + "from them?\n\nThis cannot be undone. Download the measurement first if you "
              + "want to keep it.")) return;
          probeError = ""; probeNote = "";
          try {
            const res = await window.MovieEditorAPI.probeClear();
            probeState = res; probeReport = null;
            probeNote = `Cleared ${res.runs_removed ?? res.runs ?? 0} run(s). `
              + "Nothing is steering from ratings until new ones are recorded.";
          } catch (e) { probeError = String(e.message || e); }
          paint();
        }, { disabled: !runs, danger: true }));
      rows.append(actionRow("Read the result",
        "Asks whether your good and bad ratings look different early on, and checks the "
        + "answer against shuffled ratings so a handful of runs cannot fake one.",
        probeBusy ? "Reading…" : "Read", async () => {
          probeBusy = true; probeError = ""; paint();
          try {
            probeReport = await window.MovieEditorAPI.probeAnalyse(2000);
          } catch (e) { probeError = String(e.message || e); }
          probeBusy = false; paint();
        }, { disabled: probeBusy || !runs }));
      box.append(rows);
      box.append(probeFile);
      if (probeNote) box.append(el("div", "sw-hint", probeNote));
      if (probeError) box.append(el("div", "sw-hint", probeError));
      // A reading is of the runs that existed when it was taken. Once more have been rated,
      // the table below and the count above are describing different things, and saying so
      // beats letting the two numbers quietly disagree.
      if (probeReport && probeState && probeReport.runs !== probeState.runs) {
        box.append(el("div", "sw-hint",
          `This reading covers ${probeReport.runs} run(s); ${runs} are recorded now. `
          + "Read again to include the rest."));
      }
      if (probeReport && probeReport.buckets && probeReport.buckets.length) {
        if (probeReport.multipass) {
          box.append(el("div", "sw-hint",
            `${probeReport.multipass} run(s) used a second pass. Only the first pass is read — `
            + "the second runs its own schedule, so its parts cover different ground."));
        }
        if (probeReport.unbound_steps) {
          box.append(el("div", "sw-hint",
            `${probeReport.unbound_steps} step(s) could not be measured, so those runs are `
            + "read from fewer steps than they took."));
        }
        box.append(probeTable(probeReport));
        box.append(probeVerdict(probeReport));
      }
    };
    paint();
    // Re-fetched on EVERY mount, not only the first. The section is re-mounted whenever you
    // navigate back to it, and the cached count is the one indicator of whether your recent
    // ratings are being recorded — an indicator that answers with a stale number is worse
    // than one that is not there.
    if (window.MovieEditorAPI) {
      window.MovieEditorAPI.probeStatus()
        .then((s) => { probeState = s; paint(); })
        .catch(() => {});
    }
    return box;
  }

  // ── H3 representation steering ──────────────────────────────────────────
  // Per-KEY (unlike the probe above, which pools across every key): one sidecar per
  // refinement key, so status/export/clear all act on whichever key this project is using.
  //
  // The block sweep is a 9-block permutation test -- heavy enough on CPU that running it on
  // every panel mount (status was doing this) stalled the event loop ComfyUI's own server
  // shares with this one, confirmed live as a generation freezing while the panel was open.
  // Split the same way the probe already is: cheap status auto-fetched, the actual sweep
  // only on explicit request (the "Read" button below).
  let reinsState = null, reinsSweep = null, reinsSweepBusy = false, reinsError = "";
  let biState = null, biError = "";
  let dpRows = null, dpError = "";

  // Reads a comparison the way the user would judge it by eye, but from three numbers that
  // can disagree with each other -- that disagreement is the whole point. Detail up on
  // existing edges with the picture intact is sharpening; detail up spread evenly is grain;
  // structure down means it is simply a different generation.
  function dpVerdict(r) {
    // A low `structure` has two very different causes and the numbers cannot separate them,
    // so the seed decides. Different seeds = two different generations and the comparison
    // never was an A/B (a reference pins the subject, not the sample). Same seed = the
    // change moved the shot rather than its detail, which is a RESULT, not a bad setup.
    // A structural change (different token count) is a different prompt. A small drift is
    // something rating-driven moving the conditioning under you — worth naming, but it does
    // not invalidate the comparison the way a new prompt does.
    if (r.changed && r.changed.conditioning && (r.cond_shift ?? 1) >= 1.0) {
      return "the prompt or scenes changed — a different generation, not an A/B";
    }
    const onlySeed = r.changed && !Object.keys(r.changed).filter((k) => k !== "seed").length;
    if (onlySeed && r.same_seed) {
      // Nothing differed at all: this is the instrument's noise floor for this setup.
      return r.structure > 0.98 && Math.abs(r.detail - 1) < 0.02
        ? `stable — two identical runs match (structure ${r.structure.toFixed(3)}), so a real `
          + "A/B can be trusted"
        : `two identical runs already differ this much (structure ${r.structure.toFixed(2)}, `
          + `detail x${r.detail.toFixed(2)}) — generation is not reproducible here, so no A/B `
          + "on this setup means anything";
    }
    if (r.structure < 0.85) {
      return r.same_seed
        ? `the shot itself moved (structure ${r.structure.toFixed(2)}) — this changed the `
          + "picture, not just its detail"
        : "different seed — two separate generations, so detail cannot be compared. Rerun "
          + "the same seed with the change off, then on.";
    }
    const gain = (r.detail - 1) * 100;
    const caveat = r.same_seed === false ? " (different seed — treat as indicative only)" : "";
    if (Math.abs(gain) < 2) return "no real change in detail" + caveat;
    if (gain < 0) return `detail DOWN ${Math.abs(gain).toFixed(0)}% — softer, not sharper` + caveat;
    return (r.edge_aligned > 0.35
      ? `sharper — +${gain.toFixed(0)}% detail, landing on existing edges`
      : `+${gain.toFixed(0)}% detail but spread evenly — reads as grain, not sharpening`) + caveat;
  }

  // What this pair actually was. The probe cannot know which knob you consider the
  // variable, so it diffs every scalar setting between the two runs and names the result.
  // No difference is not a wasted row: two identical runs measure how much the generation
  // varies on its own, and without that number none of the others mean anything.
  function dpChangeSummary(r) {
    const ch = r.changed;
    if (!ch) return `${r.label_before || "?"} → ${r.label_after || "?"}`;  // pre-diff rows
    const keys = Object.keys(ch).filter((k) => k !== "seed");
    if (!keys.length) {
      return r.same_seed ? "nothing changed — repeatability check"
                         : "only the seed changed — a different sample";
    }
    const rest = keys.filter((k) => k !== "conditioning");
    const shift = r.cond_shift;
    const promptMoved = !keys.includes("conditioning") ? null
      : shift >= 1.0 ? "prompt/scenes changed"
      : `conditioning drifted ${(shift * 100).toFixed(1)}%`;
    const parts = rest.slice(0, 3).map((k) => {
      const show = (v) => (v === "" || v == null ? "—" : String(v));
      return `${k}: ${show(ch[k][0])} → ${show(ch[k][1])}`;
    });
    if (promptMoved) parts.unshift(promptMoved);
    return parts.join(" · ") + (rest.length > 3 ? ` · +${rest.length - 3} more` : "");
  }

  function detailProbeRows(key) {
    const box = el("div", "sw-stack");
    const paint = () => {
      clear(box);
      const rows = el("div", "sw-rows");
      const n = dpRows ? dpRows.length : 0;
      rows.append(actionRow("Start fresh",
        "Throws away every recorded comparison for this key.",
        "Clear…", async () => {
          dpError = "";
          try { const r = await window.MovieEditorAPI.detailProbeClear(key); dpRows = r.rows; }
          catch (e) { dpError = String(e.message || e); }
          paint();
        }, { disabled: !n, danger: true }));
      box.append(rows);
      if (dpError) box.append(el("div", "sw-hint", dpError));
      if (!n) {
        box.append(el("div", "sw-hint",
          "Nothing compared yet. With recording on, generate the SAME SEED twice — once "
          + "with the change off, once with it on — and each pair is scored here. Two runs "
          + "on different seeds are different generations, not an A/B, and are flagged as "
          + "such."));
        return;
      }
      const tbl = el("div", "sw-rows");
      // Newest first: the run you just made is the one you are asking about.
      dpRows.slice().reverse().forEach((r) => {
        tbl.append(infoRow(
          dpChangeSummary(r),
          dpVerdict(r),
          r.same_seed !== false && r.structure >= 0.85 && r.detail > 1.02
            && r.edge_aligned > 0.35));
        const seeds = r.seed_before == null && r.seed_after == null ? ""
          : ` · seed ${r.seed_before ?? "?"}${r.same_seed ? "" : " → " + (r.seed_after ?? "?")}`;
        tbl.append(el("div", "sw-hint",
          `detail ×${r.detail.toFixed(3)} · structure kept ${r.structure.toFixed(3)} · `
          + `edge-aligned ${r.edge_aligned >= 0 ? "+" : ""}${r.edge_aligned.toFixed(3)}${seeds}`));
      });
      box.append(tbl);
    };
    paint();
    if (window.MovieEditorAPI) {
      window.MovieEditorAPI.detailProbeStatus(key)
        .then((r) => { dpRows = r.rows || []; paint(); })
        .catch(() => {});
    }
    return box;
  }

  // The whole point of the probe in one number. Flatness is the spread of the per-block
  // profile as a fraction of its own mean, so it is comparable across models and runs: near
  // zero means every block moves the picture by the same amount and there is nothing for
  // rating-driven block weighting to aim at.
  function biVerdict(state) {
    if (!state || state.flatness == null) return "Nothing recorded yet.";
    const f = state.flatness;
    const shape = f < 0.05 ? "flat — every block contributes about the same, so weighting "
                             + "blocks by rating has nothing to grip"
      : f < 0.20 ? "slightly uneven — some structure, not much"
      : "structured — blocks differ enough to be worth aiming at";
    return `Flatness ${f.toFixed(3)} · ${shape}`;
  }

  // How much of each block's push is NEW, rather than more of what the previous block was
  // already doing. Magnitude cannot tell those apart; this is the number that speaks to
  // whether 50 blocks are doing 50 things or one thing 50 times.
  function biNoveltyVerdict(state) {
    const n = state && state.mean_novelty;
    if (n == null) return null;
    const shape = n > 0.6 ? "each block mostly amplifies the one before it — little new per "
                            + "block"
      : n > 0.2 ? "partly new, partly a continuation of the previous block"
      : n > -0.2 ? "each block adds something the one before it did not — near-independent "
                   + "contributions"
      : "blocks partly undo each other, so the useful work is smaller than the sizes suggest";
    return `Novelty ${n >= 0 ? "+" : ""}${n.toFixed(3)} · ${shape}`;
  }

  function biProfileTable(state) {
    const overall = state && state.overall;
    if (!overall || !Object.keys(overall).length) return null;
    const diff = (state && state.difference) || null;
    // Ranked by how much the block moves the picture, not by block number: the question is
    // which blocks do the work, and a 50-row list in index order buries that.
    // Rank by SHARE, not by the ratio: ||f_i||/||x_i|| shrinks with depth by construction
    // (the residual stream grows as it goes), so ranking on it just sorts the earliest
    // blocks to the top regardless of how much they actually move the picture.
    const share = (state && state.share) || {};
    const rankOn = Object.keys(share).length ? share : overall;
    const entries = Object.keys(rankOn)
      .map((b) => [Number(b), overall[b], share[b]])
      .sort((a, b) => (b[2] ?? b[1]) - (a[2] ?? a[1]))
      .slice(0, 12);
    const tbl = el("div", "sw-rows");
    entries.forEach(([block, v, sh]) => {
      const d = diff ? diff[String(block)] : null;
      const nov = state.novelty ? state.novelty[String(block)] : null;
      let value = sh != null ? `${(sh * 100).toFixed(1)}% of the movement`
                             : (v == null ? "—" : v.toPrecision(3));
      if (nov != null) value += ` · new ${nov >= 0 ? "+" : ""}${nov.toFixed(2)}`;
      if (d != null) value += ` · liked ${d >= 0 ? "+" : ""}${d.toFixed(4)}`;
      tbl.append(infoRow(`Block ${block}`, value, d == null ? null : d > 0));
    });
    return tbl;
  }

  function blockInfluenceRows(key) {
    const box = el("div", "sw-stack");
    const paint = () => {
      clear(box);
      const rows = el("div", "sw-rows");
      const runs = biState ? biState.runs : 0;

      const row = el("div", "sw-row");
      const main = el("div", "sw-row-main");
      main.append(el("div", "sw-row-title", "Record which blocks move the picture"));
      main.append(el("div", "sw-row-hint",
        "Off by default. Recording only — nothing about your generations changes. Each "
        + "rated generation adds one profile."));
      row.append(main);
      const lbl = el("label", "chk es-toggle");
      const cb = el("input");
      cb.type = "checkbox";
      cb.checked = !!(biState && biState.enabled);
      cb.onchange = async () => {
        biError = "";
        try { biState = await window.MovieEditorAPI.blockInfluenceSetEnabled(key, cb.checked); }
        catch (e) { biError = String(e.message || e); }
        paint();
      };
      lbl.append(cb, el("span", null, ""));
      row.append(lbl);
      rows.append(row);

      rows.append(infoRow("Rated generations recorded (this key)",
        biState ? `${runs} (${biState.n_liked} liked / ${biState.n_disliked} disliked)`
                : "0", runs > 0 ? true : null));
      rows.append(actionRow("Save this key's measurement",
        "Downloads every recorded profile as one file. A rental gets replaced and "
        + "refinements/ is not in git, so without this the count restarts on the next box.",
        "Download", async () => {
          biError = "";
          try { await window.MovieEditorAPI.blockInfluenceExport(key); }
          catch (e) { biError = String(e.message || e); paint(); }
        }, { disabled: !runs }));
      rows.append(actionRow("Start fresh",
        "Throws away every recorded profile for this key. Recording stays on. Cannot be "
        + "undone — download first if you want to keep it.",
        "Clear…", async () => {
          if (!window.confirm(`Throw away ${runs} recorded profile(s) for '${key}'?\n\nThis `
              + "cannot be undone. Download it first if you want to keep it.")) return;
          biError = "";
          try { biState = await window.MovieEditorAPI.blockInfluenceClear(key); }
          catch (e) { biError = String(e.message || e); }
          paint();
        }, { disabled: !runs, danger: true }));
      box.append(rows);

      box.append(el("div", "sw-hint", biVerdict(biState)));
      const nv = biNoveltyVerdict(biState);
      if (nv) box.append(el("div", "sw-hint", nv));
      if (biState && biState.difference == null && runs > 0) {
        box.append(el("div", "sw-hint",
          `Needs ${biState.min_per_group}+ liked and ${biState.min_per_group}+ disliked `
          + "before it can say which blocks run hotter on the ones you liked."));
      }
      if (biError) box.append(el("div", "sw-hint", biError));
      const tbl = biProfileTable(biState);
      if (tbl) {
        box.append(el("div", "sw-rows-label",
          "Busiest blocks — each block's share of all the movement in the stack"));
        box.append(tbl);
      }
    };
    paint();
    if (window.MovieEditorAPI) {
      window.MovieEditorAPI.blockInfluenceStatus(key)
        .then((s) => { biState = s; paint(); })
        .catch(() => {});
    }
    return box;
  }

  function reinsSweepTable(sweep) {
    const entries = Object.entries(sweep || {}).sort((a, b) => a[1].p_value - b[1].p_value);
    if (!entries.length) return null;
    const tbl = el("div", "sw-rows");
    entries.forEach(([block, r]) => {
      const sig = r.p_value <= 0.05;
      tbl.append(infoRow(`Block ${block}`,
        `sep ${r.separation >= 0 ? "+" : ""}${r.separation.toFixed(3)} · p ${r.p_value.toFixed(3)} · n=${r.n}`,
        sig));
    });
    return tbl;
  }

  function reinsRows(key) {
    const box = el("div", "sw-stack");
    const paint = () => {
      clear(box);
      const rows = el("div", "sw-rows");
      const nLiked = reinsState ? reinsState.n_liked : 0;
      const nDisliked = reinsState ? reinsState.n_disliked : 0;
      const ready = !!(reinsState && reinsState.ready);
      rows.append(infoRow("Rated generations (this key)",
        `${nLiked} liked / ${nDisliked} disliked`, reinsState ? ready : null));
      if (reinsState && !ready) {
        rows.append(el("div", "sw-hint",
          `Needs ${reinsState.min_per_group}+ of each before it steers anything — still `
          + "capturing every generation either way."));
      }
      rows.append(actionRow("Save this key's REINS data",
        "Downloads every captured run (all candidate blocks, not just the one that steers) "
        + "as one file — same reason the probe has this: a rental gets replaced and "
        + "refinements/ is not in git.",
        "Download", async () => {
          reinsError = "";
          try { await window.MovieEditorAPI.reinsExport(key); }
          catch (e) { reinsError = String(e.message || e); paint(); }
        }, { disabled: !nLiked && !nDisliked }));
      rows.append(actionRow("Start fresh",
        "Throws away every captured run for this key. Cannot be undone — download first if "
        + "you want to keep it.",
        "Clear…", async () => {
          if (!window.confirm(`Throw away all REINS data for '${key}'?\n\nThis cannot be `
              + "undone. Download it first if you want to keep it.")) return;
          reinsError = "";
          try { reinsState = await window.MovieEditorAPI.reinsClear(key); reinsSweep = null; }
          catch (e) { reinsError = String(e.message || e); }
          paint();
        }, { disabled: !nLiked && !nDisliked, danger: true }));
      rows.append(actionRow("Read the block sweep",
        "Runs a permutation test at every candidate block to see which one actually "
        + "separates liked from disliked. Heavy enough on CPU to notice — not run "
        + "automatically, only when you ask for it.",
        reinsSweepBusy ? "Reading…" : "Read", async () => {
          reinsSweepBusy = true; reinsError = ""; paint();
          try { reinsSweep = await window.MovieEditorAPI.reinsSweep(key, 2000); }
          catch (e) { reinsError = String(e.message || e); }
          reinsSweepBusy = false; paint();
        }, { disabled: reinsSweepBusy || (!nLiked && !nDisliked) }));
      box.append(rows);
      if (reinsError) box.append(el("div", "sw-hint", reinsError));
      const sweepTbl = reinsSweep && reinsSweepTable(reinsSweep.sweep);
      if (sweepTbl) {
        box.append(el("div", "sw-rows-label",
          `Block sweep — which of H3's blocks separates liked from disliked (block `
          + `${reinsState ? reinsState.default_block : "?"} is the one that actually steers)`));
        box.append(sweepTbl);
      }
    };
    paint();
    // Re-fetched on every mount, same reasoning as the probe: a stale count is worse than
    // none, and this section is remounted every time you navigate back to it.
    if (window.MovieEditorAPI) {
      window.MovieEditorAPI.reinsStatus(key)
        .then((s) => { reinsState = s; paint(); })
        .catch(() => {});
    }
    return box;
  }

  function infoRow(title, value, ok) {
    const row = el("div", "sw-row");
    const main = el("div", "sw-row-main");
    main.append(el("div", "sw-row-title", title));
    row.append(main);
    const val = el("span", "sw-row-value", value);
    if (ok != null) {
      const led = el("span", "sw-led " + (ok ? "ok" : "bad"));
      row.append(led);
    }
    row.append(val);
    return row;
  }

  // ── built-in section: About FunPack ────────────────────────────────────
  register({
    id: "about", group: "", order: 0, title: "About FunPack",
    subtitle: "",
    keywords: "about version commit branch copyright info funpack cutting room "
      + "cpu chip memory ram gpu graphics vram disk storage system os python torch cuda specs hardware",
    iconBg: "linear-gradient(180deg,#ffc36b,#e0891f)",
    icon: '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="#fff" stroke-width="1.5"><circle cx="8" cy="8" r="6.2"/><circle cx="8" cy="8" r="2.4" fill="#fff" stroke="none"/></svg>',
    mount(body, ctx) {
      const G = window.FunPackGit;
      const wrap = el("div", "sw-about");
      body.append(wrap);

      // Host facts describe the machine ComfyUI runs on, which on a rental is NOT the
      // machine showing this window. Fetched once per mount; nothing here changes while
      // the window is open. Null until it arrives (or forever, if the backend is older).
      let sys = null;

      const gb = (n) => (n == null ? null : `${n} GB`);

      function hardwareFacts(fact) {
        if (!sys) return;
        const cpu = sys.cpu || {};
        const cores = cpu.cores && cpu.threads && cpu.cores !== cpu.threads
          ? `${cpu.cores}C/${cpu.threads}T`
          : (cpu.threads ? `${cpu.threads}C` : null);
        fact("Chip", [cpu.name, cores].filter(Boolean).join(" · ") || cpu.arch);

        const mem = sys.memory || {};
        fact("Memory", mem.available_gb != null && mem.total_gb != null
          ? `${mem.available_gb} GB free of ${mem.total_gb} GB`
          : gb(mem.total_gb));

        const gpus = sys.gpus || [];
        if (gpus.length) {
          gpus.forEach((g, i) => {
            const label = gpus.length > 1 ? `Graphics ${i}` : "Graphics";
            fact(label, [g.name, gb(g.vram_gb), g.capability].filter(Boolean).join(" · "));
          });
        } else {
          // No CUDA device: say which kind of nothing, since "—" reads like a bug.
          fact("Graphics", sys.mps ? "Apple GPU (MPS)" : "CPU only (no CUDA device)");
        }

        const disk = sys.disk || {};
        fact("Storage", disk.free_gb != null && disk.total_gb != null
          ? `${disk.free_gb} GB available of ${disk.total_gb} GB`
          : gb(disk.total_gb));
      }

      function softwareFacts(fact) {
        if (!sys) return;
        fact("System", sys.os);
        fact("ComfyUI", sys.comfyui);
        fact("Python", sys.python);
        const t = sys.torch || {};
        fact("Torch", [t.version, t.cuda ? `CUDA ${t.cuda}` : null].filter(Boolean).join(" · "));
        if (t.attention) fact("Attention", t.attention);
        if (sys.host) fact("Host", sys.host);
      }

      function render() {
        clear(wrap);
        const git = G ? G.get() : null;
        wrap.append(el("div", "sw-about-mark", "◉"));
        const major = String(git?.version || "").split(".")[0];
        wrap.append(el("div", "sw-about-name", "FunPack" + (major ? " " + major : "")));
        // Omitted when absent — empty quotes would look like a bug.
        if (git?.codename) {
          wrap.append(el("div", "sw-about-codename", "“" + git.codename + "”"));
        }
        // window.FunPackAppName lets a different frontend sharing this file (e.g. Easy
        // Gen) relabel the app name without forking the whole section; Editor leaves it
        // unset and keeps "Cutting Room".
        wrap.append(el("div", "sw-about-sub", window.FunPackAppName || "Cutting Room"));

        const facts = el("div", "sw-about-facts");
        const fact = (k, v) => {
          const r = el("div", "sw-about-fact");
          r.append(el("span", "sw-about-k", k), el("span", "sw-about-v", v || "—"));
          facts.append(r);
        };
        const group = (label) => facts.append(el("div", "sw-about-group", label));

        fact("Version", git?.version);
        fact("Commit", git?.ok ? git.commit + (git.dirty ? " (local changes)" : "") : null);
        fact("Branch", git?.ok ? git.branch : null);

        if (sys) {
          group("Hardware");
          hardwareFacts(fact);
          group("Software");
          softwareFacts(fact);
        }
        wrap.append(facts);

        if (sys) {
          wrap.append(el("div", "sw-about-hint",
            "The machine ComfyUI runs on — not this browser."));
        }

        const upd = el("button", "btn ghost tiny", "Software Update…");
        upd.onclick = () => ctx.openSection("system");
        wrap.append(upd);

        wrap.append(el("div", "sw-about-copy", "© 2025–2026 DigitalGarbage"));
      }

      render();
      if (G?.refresh) G.refresh().then(() => { if (wrap.isConnected) render(); }).catch(() => {});
      // Optional-chained: a frontend sharing this file without the endpoint (or an older
      // backend) simply keeps the version-only About instead of erroring.
      window.MovieEditorAPI?.systemInfo?.()
        .then((info) => { sys = info; if (wrap.isConnected) render(); })
        .catch(() => {});
    },
  });

  // ── built-in section: Refinement & Taste (Learning) ────────────────────
  // Buttons reuse the single implementations exported by menubar.js
  // (window.FunPackMaintenance) — no duplicated logic, just a second door.
  register({
    id: "refinement", group: "Learning", title: "Refinement & Taste",
    subtitle: "Learned-taste state: refinement keys and the Absolute global-taste store.",
    keywords: "refinement key export import delete clear absolute taste store learning ratings "
            + "trajectory probe record measure early separation motion action steering",
    iconBg: "linear-gradient(180deg,#ff8bc2,#d84f92)",
    icon: '<svg viewBox="0 0 16 16" width="13" height="13"><path d="M8 1.5 9.6 6.4 14.5 8 9.6 9.6 8 14.5 6.4 9.6 1.5 8 6.4 6.4 8 1.5z" fill="#fff"/></svg>',
    mount(body) {
      const M = window.FunPackMaintenance || {};
      const S = window.Store;
      const wrap = el("div", "sw-stack");
      body.append(wrap);

      function render() {
        clear(wrap);
        wrap.append(el("div", "sw-hint",
          "Refinement keys hold what FunPack has learned from your ratings for a named session. "
          + "They are stored on this ComfyUI instance as <key>.json plus sidecar banks."));

        const st = S ? S.get() : {};
        const studioOn = !!window.PipelineCaps?.usesFunpackStudio?.(st);
        const armed = !!st.resetSessionArmed;
        wrap.append(el("div", "sw-rows-label", "Studio session"));
        const sess = el("div", "sw-rows");
        sess.append(actionRow("Reset Studio session",
          armed
            ? "Armed — the session's learned keys are wiped on the FIRST run of the next generation. Click to cancel."
            : "Wipe this session's learned keys on the next generation (asks which keys first).",
          armed ? "✓ Armed — cancel" : "Reset…",
          () => S?.resetStudioSession?.(),
          { disabled: !st.project || !studioOn, danger: !armed }));
        wrap.append(sess);

        wrap.append(el("div", "sw-rows-label", "Refinement keys"));
        const rows = el("div", "sw-rows");
        rows.append(actionRow("Export refinement key", "Download a key as <key>.json.", "⬇ Export…", () => M.exportRefinementKey?.()));
        rows.append(actionRow("Import refinement key", "Load a previously exported <key>.json onto this instance.", "Import…", () => M.importRefinementKeyFile?.()));
        wrap.append(rows);

        wrap.append(el("div", "sw-rows-label", "Trajectory probe"));
        wrap.append(probeRows());

        wrap.append(el("div", "sw-rows-label", "H3 representation steering (REINS)"));
        wrap.append(reinsRows(st.project?.refinement_key || "default"));

        wrap.append(el("div", "sw-rows-label", "Block influence (measurement only)"));
        wrap.append(blockInfluenceRows(st.project?.refinement_key || "default"));

        wrap.append(el("div", "sw-rows-label",
          "Detail check — did a change sharpen the picture, or just alter it?"));
        wrap.append(detailProbeRows(st.project?.refinement_key || "default"));

        wrap.append(el("div", "sw-rows-label", "Danger zone"));
        const danger = el("div", "sw-rows");
        danger.append(actionRow("Delete refinement key",
          "Removes a key AND all its sidecars (value function, blessed banks, creativity latent, velocity memory).",
          "Delete…", () => M.deleteRefinementKey?.(), { danger: true }));
        danger.append(actionRow("Clear global taste store",
          "Wipes the Absolute store pooled from every rated run (applied in absolute/both steer mode).",
          "Clear…", () => M.clearGlobalTaste?.(), { danger: true }));
        wrap.append(danger);
      }

      const unsub = S ? S.subscribe(render) : null;
      render();
      return () => { if (unsub) unsub(); };
    },
  });

  // ── built-in section: Updates & ComfyUI (System) ───────────────────────
  register({
    id: "system", group: "System", order: 1, title: "Updates & ComfyUI",
    subtitle: "Server connection, FunPack code updates, pipeline health.",
    keywords: "update git branch restart comfyui server connection health template version",
    iconBg: "linear-gradient(180deg,#a7adb8,#6b7280)",
    icon: '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="#fff" stroke-width="1.5" stroke-linecap="round"><circle cx="8" cy="8" r="2.2"/><path d="M8 1.6v2M8 12.4v2M1.6 8h2M12.4 8h2M3.5 3.5l1.4 1.4M11.1 11.1l1.4 1.4M12.5 3.5l-1.4 1.4M4.9 11.1l-1.4 1.4"/></svg>',
    mount(body) {
      const S = window.Store, G = window.FunPackGit, M = window.FunPackMaintenance || {};
      const wrap = el("div", "sw-stack");
      body.append(wrap);

      function render() {
        clear(wrap);
        const st = S ? S.get() : {};
        const git = G ? G.get() : null;
        const gitOk = !!(git && git.ok);

        wrap.append(el("div", "sw-rows-label", "ComfyUI server"));
        const srv = el("div", "sw-rows");
        srv.append(infoRow("Connection", st.health?.ok ? "Connected · " + window.location.host : "Not reachable", !!st.health?.ok));
        srv.append(actionRow("Open ComfyUI", "The node-graph UI, in a new tab.", "↗ Open", () => window.open("/", "_blank")));
        srv.append(actionRow("Restart ComfyUI", "Down for ~10–40s; any running generation is lost. Reloads automatically.", "⟳ Restart…", () => M.restartComfy?.(), { danger: true }));
        wrap.append(srv);

        wrap.append(el("div", "sw-rows-label", "FunPack code"));
        const code = el("div", "sw-rows");
        code.append(infoRow("Branch", gitOk
          ? git.branch + (git.commit ? " @ " + git.commit : "") + (git.dirty ? " (local changes)" : "")
          : (git?.detail || "unavailable")));
        code.append(actionRow("Switch branch", "Check out another FunPack branch and reload.", "Switch…",
          () => G?.switchBranch?.(), { disabled: !gitOk || !!git?.dirty }));
        code.append(actionRow("Update FunPack",
          gitOk && git.behind > 0 ? git.behind + " new commit(s) available." : "Pull the latest code and reload.",
          gitOk && git.behind > 0 ? git.behind + "↓ Update" : "⬇⟳ Update",
          () => G?.update?.(), { disabled: !gitOk || !!git?.dirty }));
        wrap.append(code);

        wrap.append(el("div", "sw-rows-label", "Pipeline health"));
        const health = el("div", "sw-rows");
        health.append(infoRow("Workflow template", st.health?.reference_loaded ? "loaded" : "missing", !!st.health?.reference_loaded));
        health.append(infoRow("Configured nodes", String(st.health?.configured_slots ?? 0)));
        wrap.append(health);
      }

      const unsub = S ? S.subscribe(render) : null;
      render();
      if (G?.refresh) G.refresh().then(() => { if (wrap.isConnected) render(); }).catch(() => {});
      return () => { if (unsub) unsub(); };
    },
  });
})();
