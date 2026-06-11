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

  async function restartComfy() {
    if (!confirm("Restart ComfyUI now?\n\nThe server will be down for ~10–40s and any running generation will be lost. This page reloads automatically when it's back.")) return;
    const ov = el("div", "restart-overlay");
    const card = el("div", "restart-card");
    card.append(el("div", "restart-spin"));
    const msg = el("div", "restart-msg", "Restarting ComfyUI…\nThis page will reload when it's back.");
    card.append(msg); ov.append(card); document.body.append(ov);
    try { await window.MovieEditorAPI.restart(); } catch (_) { /* connection drops as it execv's — expected */ }
    const start = Date.now();
    const tick = async () => {
      try {
        const h = await window.MovieEditorAPI.health();
        if (h && h.ok) { location.reload(); return; }
      } catch (_) { /* still down */ }
      if (Date.now() - start > 90000) msg.textContent = "Still waiting on ComfyUI…\nIt may need a manual restart — check the console.";
      setTimeout(tick, 2000);
    };
    setTimeout(tick, 3500);  // give it a moment to actually go down first
  }

  function menuSpec() {
    const st = S.get();
    const recent = (st.projects || []).slice(0, 8).map((p) => ({
      label: p.name, hint: `${p.scene_count}▦`, action: () => S.loadProject(p.id),
    }));
    return {
      File: [
        { label: "New Project", hint: "⌘N", action: () => S.newProject(prompt("Project name:", "Untitled montage")) },
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
        { label: "Delete Scene", disabled: !sel(), action: () => S.removeScene(sel()) },
        { sep: true },
        { label: "Move Scene Left", disabled: !sel(), action: () => S.moveScene(sel(), -1) },
        { label: "Move Scene Right", disabled: !sel(), action: () => S.moveScene(sel(), 1) },
        { sep: true },
        { label: "Toggle Exclude", disabled: !sel(), action: () => { const s = S.scene(sel()); if (s) S.patchScene(s.id, { excluded: !s.excluded }); } },
      ],
      View: [
        { label: "Refresh Preview", action: () => S.refreshPreview() },
        { sep: true },
        { label: "Reset Layout", action: () => { const r = document.documentElement; r.style.removeProperty("--media-w"); r.style.removeProperty("--timeline-h"); } },
      ],
      Settings: [
        { label: "Engine settings…", disabled: !hasProject(), action: () => window.EngineSettingsModal.open() },
        { sep: true },
        { label: "Models…", action: () => window.ModelsModal.open() },
        { label: "Refresh model list", hint: "R", action: async () => { try { await window.MovieEditorAPI.refreshModels(); } catch (_) {} } },
        { sep: true },
        { label: `Conditioning: ${_roleLabel(st.project?.conditioning_slot, "FunPack Studio")}`, disabled: !hasProject(),
          action: () => _pickRole("conditioning_slot", "Conditioning node", st) },
        { label: `Sampler: ${_roleLabel(st.project?.sampler_slot, "FunPack Chain Sampler")}`, disabled: !hasProject(),
          action: () => _pickRole("sampler_slot", "Sampler node", st) },
      ],
      FunPack: [
        { label: st.resetSessionArmed ? "Reset Studio session ✓ armed — click to cancel" : "Reset Studio session",
          disabled: !hasProject(), action: () => S.resetStudioSession() },
        { sep: true },
        { menulabel: "Libraries (in ComfyUI Studio)" },
        { label: "Open ComfyUI", hint: "↗", action: () => window.open("/", "_blank") },
        { label: "Restart ComfyUI", hint: "⟳", danger: true, action: restartComfy },
        { sep: true },
        { label: st.health?.reference_loaded ? "Pipeline reference: loaded" : "Pipeline reference: missing", disabled: true },
        { label: `Configured nodes: ${st.health?.configured_slots ?? 0}`, disabled: true },
        { label: st.health?.ok ? `Connected · ${window.location.host}` : "ComfyUI not reachable", disabled: true },
      ],
    };
  }

  function _roleLabel(slotId, defaultLabel) {
    if (!slotId || slotId === "funpack") return defaultLabel;
    const slot = (S.get().models?.slots || []).find((s) => s.id === slotId);
    return slot ? (slot.label || slot.node_class || slotId) : slotId;
  }

  function _pickRole(field, title, st) {
    closeAll();
    const slots = (st.models?.slots || []);
    const cur = st.project?.[field] || "funpack";
    const opts = [{ id: "funpack", label: field === "conditioning_slot" ? "FunPack Studio (built-in)" : "FunPack Chain Sampler (built-in)" },
      ...slots.map((s) => ({ id: s.id, label: s.label || s.node_class || s.id }))];
    if (!window.SlotPicker) return;
    window.SlotPicker.open({
      title,
      options: opts.map((o) => ({ value: o.id, label: o.label, hint: o.id === cur ? "current" : "" })),
      onPick: (id) => {
        if (field === "conditioning_slot") S.setConditioningSlot(id);
        else S.setSamplerSlot(id);
      },
    });
  }

  function closeAll() { openName = null; veil.hidden = true; render(); }

  function openMenu(name) { openName = name; veil.hidden = false; render(); }

  function render() {
    const spec = menuSpec();
    clear(menusEl);
    Object.keys(spec).forEach((name) => {
      const wrap = el("div", "menu" + (openName === name ? " open" : ""));
      const btn = el("button", "menu-btn", name);
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
  }

  function renderSaveChip(st, detail) {
    const sc = document.getElementById("save-chip");
    if (!sc) return;
    const saving = detail?.saving ?? st.saving;
    const unsaved = detail?.unsaved ?? st.unsaved;
    sc.className = "save-chip" + (saving || unsaved ? " dirty" : "");
    sc.textContent = saving ? "saving…" : (unsaved ? "unsaved" : (st.project ? "saved" : ""));
  }

  function renderChips(st) {
    const hc = document.getElementById("health-chip");
    const ok = st.health?.ok;
    hc.className = "health-chip " + (ok ? "ok" : "bad");
    clear(hc);
    hc.append(el("span", "led"));
    hc.append(el("span", null, ok ? "ComfyUI live" : "offline"));
    renderSaveChip(st);
  }

  veil.addEventListener("click", closeAll);
  window.addEventListener("keydown", (e) => { if (e.key === "Escape") closeAll(); });

  window.addEventListener("funpack-save-status", (e) => renderSaveChip(S.get(), e.detail));
  window.addEventListener("funpack-history-state", () => render());
  window.addEventListener("keydown", (e) => {
    if (!(e.metaKey || e.ctrlKey) || e.altKey) return;
    const t = document.activeElement;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
    if (e.key === "z" || e.key === "Z") {
      e.preventDefault();
      if (e.shiftKey) S.redo(); else S.undo();
      render();
    }
  });
  S.subscribe((st) => { render(); renderChips(st); });
  render();
})();
