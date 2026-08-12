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

  function show(id) {
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
    };
    cleanup = spec.mount(bodyEl, ctx) || null;
    renderNav();
  }

  function close() {
    const wasOpen = !!overlay;
    teardownSection();
    if (overlay) overlay.remove();
    overlay = null; activeId = null;
    // One save for the whole session, on the way out, with the values that are on screen.
    // Nothing in here is worth a network round-trip per keystroke, and a save landing
    // mid-edit used to come back and overwrite the knob still under the cursor.
    if (wasOpen) window.Store?.resumeSave?.();
  }

  function open(id) {
    // Recover if something removed our overlay from the DOM directly
    // (e.g. the tour's closeModalOverlay sweep) without calling close().
    if (overlay && !overlay.isConnected) { close(); }
    if (overlay) { show(id || activeId); return; }
    query = "";
    // Held until close() — see the note there. Paired strictly with the overlay's
    // lifetime, and only on the branch that actually creates one.
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
    const x = el("button", "btn ghost tiny sw-close", "✕");
    x.onclick = close;
    head.append(ht, actionsEl, x);
    bodyEl = el("div", "sw-body");
    main.append(head, bodyEl);

    win.append(side, main);
    overlay.append(win);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.body.append(overlay);
    show(id);
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

  // Mount ONE registered section into an arbitrary container, with no sidebar, no search
  // and no window chrome. The setup wizard shows Models (and, on Easy Gen, Shortcuts) as
  // full screens of its own — asking someone to set up models should not drop the whole
  // Settings window on top of the flow they are halfway through.
  //
  // Returns { spec, cleanup } or null when nothing has registered that id.
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

  window.SettingsWindow = { open, close, register, navItem, mountSection, hasSection };

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
    keywords: "refinement key export import delete clear absolute taste store learning ratings",
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
