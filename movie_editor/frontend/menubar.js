// Top menu bar: File / Edit / View / Settings / FunPack + status chips.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const menusEl = document.getElementById("menus");
  const veil = document.getElementById("menu-veil");
  let openName = null;

  function sel() { return S.get().selectedSceneId; }
  function hasProject() { return !!S.get().project; }

  // hidden file input for importing a project file
  const importFileInput = (function () {
    const inp = document.createElement("input");
    inp.type = "file"; inp.accept = ".json"; inp.style.display = "none";
    inp.onchange = () => { if (inp.files[0]) { S.importProject(inp.files[0]); inp.value = ""; } };
    document.body.appendChild(inp);
    return inp;
  })();

  const refinementKeyImportInput = (function () {
    const inp = document.createElement("input");
    inp.type = "file"; inp.accept = ".json,application/json"; inp.style.display = "none";
    inp.onchange = async () => {
      const file = inp.files?.[0];
      inp.value = "";
      if (!file) return;
      try {
        const data = JSON.parse(await file.text());
        let res;
        try {
          res = await window.MovieEditorAPI.importRefinementKey(data);
        } catch (e) {
          if (!e.exists) throw e;
          // Name collision: keys are stored by name (<key>.json), so importing
          // replaces the existing one. Confirm before overwriting.
          if (!confirm(`A refinement key named "${e.key}" already exists.\n\nOverwrite it with the imported file?`)) {
            return;
          }
          res = await window.MovieEditorAPI.importRefinementKey(data, { overwrite: true });
        }
        alert(`Imported refinement key "${res.imported || file.name}".`);
      } catch (e) {
        alert("Refinement key import failed: " + (e.message || e));
      }
    };
    document.body.appendChild(inp);
    return inp;
  })();

  function _projectRefinementKey() {
    const st = S.get();
    const raw = st.project?.studio_inputs?.studio_settings;
    if (!raw) return "";
    try {
      const parsed = JSON.parse(raw);
      return String(parsed?.refinement_key || "").trim();
    } catch (_) {
      return "";
    }
  }

  async function exportRefinementKey() {
    closeAll();
    let payload;
    try { payload = await window.MovieEditorAPI.refinementKeys(); }
    catch (e) { alert("Could not list refinement keys: " + (e.message || e)); return; }
    const names = (payload.keys || []).filter((k) => k && k !== "-None-");
    if (!names.length) { alert("No refinement keys on disk yet."); return; }
    const cur = _projectRefinementKey();
    if (!window.SlotPicker) return;
    window.SlotPicker.open({
      title: "Export refinement key",
      options: names.map((k) => ({ value: k, label: k, hint: k === cur ? "project session" : "" })),
      onPick: async (key) => {
        try {
          await window.MovieEditorAPI.exportRefinementKeyFile(key);
        } catch (e) {
          alert("Refinement key export failed: " + (e.message || e));
        }
      },
    });
  }

  async function deleteRefinementKey() {
    closeAll();
    let payload;
    try { payload = await window.MovieEditorAPI.refinementKeys(); }
    catch (e) { alert("Could not list refinement keys: " + (e.message || e)); return; }
    const names = (payload.keys || []).filter((k) => k && k !== "-None-");
    if (!names.length) { alert("No refinement keys on disk yet."); return; }
    const cur = _projectRefinementKey();
    if (!window.SlotPicker) return;
    window.SlotPicker.open({
      title: "Delete refinement key",
      options: names.map((k) => ({ value: k, label: k, hint: k === cur ? "project session" : "" })),
      onPick: async (key) => {
        // Atomic: removes the key AND its sidecars (value function, blessed
        // attention/K-V banks, creativity latent, velocity memory). Deleting only
        // <key>.json by hand orphans those and they keep steering future runs.
        if (!confirm(`Delete refinement key "${key}"?\n\nThis removes its learned state AND all sidecars (value function, blessed attention/K-V banks, creativity latent, velocity memory). This cannot be undone.`)) {
          return;
        }
        try {
          const res = await window.MovieEditorAPI.deleteRefinementKey(key);
          alert(`Deleted refinement key "${res.deleted || key}" (${res.removed || 0} file(s)).`);
        } catch (e) {
          alert("Refinement key delete failed: " + (e.message || e));
        }
      },
    });
  }

  async function clearGlobalTaste() {
    closeAll();
    let info;
    try { info = await window.MovieEditorAPI.absoluteStoreInfo(); }
    catch (e) { alert("Could not read the global-taste store: " + (e.message || e)); return; }
    if (!info.exists) { alert("The Absolute global-taste store is already empty."); return; }
    if (!confirm(`Clear the Absolute global-taste store?\n\nIt has pooled ${info.total_iterations} rated generation(s) (${info.liked_count} liked / ${info.bad_count} disliked directions) across all prompts. It learns from every rated run and is applied only in absolute/both steer mode. This cannot be undone.`)) {
      return;
    }
    try {
      await window.MovieEditorAPI.clearAbsoluteStore();
      alert("Cleared the Absolute global-taste store.");
    } catch (e) {
      alert("Clear global taste failed: " + (e.message || e));
    }
  }

  function _restartOverlay(message) {
    return window.FunPackRestart?.showOverlay?.(message);
  }

  function _waitForComfyReload(msgEl, startMs) {
    window.FunPackRestart?.waitForReload?.(msgEl, startMs);
  }

  async function restartComfy() {
    if (!confirm("Restart ComfyUI now?\n\nThe server will be down for ~10-40s and any running generation will be lost. This page reloads automatically when it's back.")) return;
    const msg = _restartOverlay("Restarting ComfyUI…\nThis page will reload when it's back.");
    try { await window.MovieEditorAPI.restart(); } catch (_) { /* connection drops as it execv's - expected */ }
    _waitForComfyReload(msg, Date.now());
  }

  function promptNewProject() {
    if (window.SlotPicker?.openPrompt) {
      window.SlotPicker.openPrompt({
        title: "New project",
        value: "Untitled montage",
        placeholder: "Project name",
        onPick: (name) => S.newProject(name),
      });
      return;
    }
    S.newProject(prompt("Project name:", "Untitled montage"));
  }

  function menuSpec() {
    const st = S.get();
    const recent = (st.projects || []).slice(0, 8).map((p) => ({
      label: p.name, hint: `${p.scene_count}▦`, action: () => S.loadProject(p.id),
    }));
    return {
      File: [
        { label: "New Project", hint: "⌘N", action: promptNewProject },
        { label: "Project Setup Wizard…", hint: "theme · model · tour",
          action: () => window.Onboarding?.reopen?.() },
        { sep: true },
        { menulabel: "Open recent" },
        ...(recent.length ? recent : [{ label: "No projects", disabled: true }]),
        { sep: true },
        { label: "Save Project File…", disabled: !hasProject(), hint: "⬇", action: () => S.downloadProject() },
        { label: "Load Project File…", action: () => importFileInput.click() },
        { sep: true },
        { label: "Import Media…", soon: true, disabled: true },
        { label: "Delete Current Project", danger: true, disabled: !hasProject(),
          action: () => { const p = S.get().project; if (p && confirm(`Delete "${p.name}"?`)) S.deleteProject(p.id); } },
      ],
      Edit: [
        { label: "Undo", hint: "⌘Z", disabled: !window.EditorHistory?.canUndo(), action: () => S.undo() },
        { label: "Redo", hint: "⇧⌘Z", disabled: !window.EditorHistory?.canRedo(), action: () => S.redo() },
        { sep: true },
        { label: "Add Scene", hint: "+", disabled: !hasProject(), action: () => S.addScene() },
        { label: "Delete Scene", disabled: !(st.selectedSceneIds?.length || st.selectedSceneId), action: () => S.removeSelectedScenes() },
        { sep: true },
        { label: "Move Clip Left", hint: "timeline cut", disabled: !sel(), action: () => S.moveTimelineClip(sel(), -1) },
        { label: "Move Clip Right", hint: "timeline cut", disabled: !sel(), action: () => S.moveTimelineClip(sel(), 1) },
        { sep: true },
        { label: "Toggle Exclude", disabled: !sel(), action: () => { const s = S.scene(sel()); if (s) S.patchScene(s.id, { excluded: !s.excluded }); } },
      ],
      View: [
        { label: "Refresh Preview", action: () => S.refreshPreview() },
        { sep: true },
        { label: "Reset Layout", action: () => { const r = document.documentElement; r.style.removeProperty("--media-w"); r.style.removeProperty("--timeline-h"); } },
      ],
      Help: window.__FUNPACK_TOUR__ ? [
        { label: "Restart tour", action: () => window.TourGuide?.jump?.(0) },
        { label: "Skip to FAQ", action: () => window.TourGuide?.jump?.(17) },
        { sep: true },
        { label: "Exit tour", action: () => window.TourGuide?.exit?.(false) },
      ] : [
        { label: "Welcome tour…", hint: "?", action: () => {
          const u = window.TourSandbox?.tourUrl?.() || (window.location.pathname + "?mode=tour");
          window.location.href = u;
        } },
      ],
    };
  }

  function closeAll() { openName = null; veil.hidden = true; render(); }

  function openMenu(name) { openName = name; veil.hidden = false; render(); }

  // Simple ⇄ Editor, beside the wordmark. Both drive the same project and pipeline; the
  // mode only decides how much of the app is on screen.
  function renderModeSwitch() {
    const M = window.FunPackMode;
    const host = document.getElementById("mode-switch");
    if (!M || !host) return;
    clear(host);
    [["simple", "Simple", "Prompt, Generate, result — nothing else"],
     ["editor", "Editor", "Timeline, inspector, ratings, every setting"]].forEach(([key, label, title]) => {
      const b = el("button", "mode-btn" + (M.is(key) ? " on" : ""), label);
      b.type = "button";
      b.title = title;
      b.onclick = () => {
        // Once, the first time: Simple genuinely changes what a run does, and finding
        // that out from a scene report is too late.
        if (key === "simple" && !M.warned()) {
          const ok = confirm(
            "Simple mode generates what you asked for and nothing else.\n\n"
            + "Rating-driven steering, cross-shot memory and experimental sampling are "
            + "switched OFF while you are in it — none of them can do anything without "
            + "ratings or a second shot. The second pass still works.\n\n"
            + "Your project settings are kept — switch back to Editor to use them again.");
          if (!ok) return;
          M.markWarned();
        }
        M.set(key);
      };
      host.append(b);
    });
  }

  function render() {
    renderModeSwitch();
    const spec = menuSpec();
    clear(menusEl);
    Object.keys(spec).forEach((name) => {
      const wrap = el("div", "menu" + (openName === name ? " open" : ""));
      const btn = el("button", "menu-btn", name);
      btn.dataset.menu = name;
      btn.onclick = (e) => { e.stopPropagation(); openName === name ? closeAll() : openMenu(name); };
      btn.onmouseenter = () => { if (openName && openName !== name) openMenu(name); };
      wrap.append(btn);
      if (openName === name) {
        const pop = el("div", "menu-pop");
        spec[name].forEach((item) => {
          if (item.sep) return pop.append(el("div", "menu-sep"));
          if (item.menulabel) return pop.append(el("div", "menu-label", item.menulabel));
          const mi = el("div", "menu-item" + (item.disabled ? " disabled" : "") + (item.danger ? " danger" : ""));
          mi.append(el("span", null, item.label + (item.soon ? "  ⋯" : "")));
          if (item.hint) mi.append(el("span", "hint", item.hint));
          if (!item.disabled && item.action) mi.onclick = () => { closeAll(); item.action(); };
          pop.append(mi);
        });
        wrap.append(pop);
      }
      menusEl.append(wrap);
    });
    // Single door to ALL settings — a plain button, not a menu (no duplicated entries).
    // Keeps .menu-btn[data-menu="Settings"] so the tour can still point at it.
    const sw = el("div", "menu");
    const sbtn = el("button", "menu-btn", "Settings");
    sbtn.dataset.menu = "Settings";
    sbtn.title = "All settings (⌘,)";
    sbtn.onclick = (e) => { e.stopPropagation(); closeAll(); window.SettingsWindow.open(); };
    sbtn.onmouseenter = () => { if (openName) closeAll(); };
    sw.append(sbtn);
    menusEl.append(sw);
  }

  function renderSaveChip(st, detail) {
    const sc = document.getElementById("save-chip");
    if (!sc) return;
    const saving = detail?.saving ?? st.saving;
    const unsaved = detail?.unsaved ?? st.unsaved;
    sc.className = "save-chip" + (saving || unsaved ? " dirty" : "");
    sc.textContent = window.__FUNPACK_TOUR__
      ? "demo"
      : (saving ? "saving…" : (unsaved ? "unsaved" : (st.project ? "saved" : "")));
  }

  function renderChips(st) {
    const hc = document.getElementById("health-chip");
    const ok = st.health?.ok;
    hc.className = "health-chip " + (ok ? "ok" : "bad");
    clear(hc);
    hc.append(el("span", "led"));
    hc.append(el("span", null, window.__FUNPACK_TOUR__ ? "Tour demo" : (ok ? "ComfyUI live" : "offline")));
    renderSaveChip(st);
  }

  veil.addEventListener("click", closeAll);
  window.addEventListener("keydown", (e) => { if (e.key === "Escape") closeAll(); });

  window.addEventListener("funpack-save-status", (e) => renderSaveChip(S.get(), e.detail));
  window.addEventListener("funpack-history-state", () => render());
  window.addEventListener("funpack-ui-mode", () => render());
  window.addEventListener("keydown", (e) => {
    if (!(e.metaKey || e.ctrlKey) || e.altKey) return;
    const t = document.activeElement;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
    if (e.key === "n" || e.key === "N") {
      e.preventDefault();
      promptNewProject();
      return;
    }
    if (e.key === "z" || e.key === "Z") {
      e.preventDefault();
      if (e.shiftKey) S.redo(); else S.undo();
      render();
    }
  });
  S.subscribe((st) => { render(); renderChips(st); });
  render();

  // Single implementations for maintenance actions, shared with the Settings window
  // (Refinement & Taste + Updates & ComfyUI sections) — menus stay as shortcuts.
  window.FunPackMaintenance = {
    exportRefinementKey,
    importRefinementKeyFile: () => refinementKeyImportInput.click(),
    deleteRefinementKey,
    clearGlobalTaste,
    restartComfy,
  };
})();
