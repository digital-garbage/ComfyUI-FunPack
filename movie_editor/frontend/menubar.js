// Top menu bar: File / Edit / View / Generate / FunPack + status chips.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const menusEl = document.getElementById("menus");
  const veil = document.getElementById("menu-veil");
  let openName = null;

  function sel() { return S.get().selectedSceneId; }
  function hasProject() { return !!S.get().project; }

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
        { label: "Import Media…", soon: true, disabled: true },
        { label: "Delete Current Project", danger: true, disabled: !hasProject(),
          action: () => { const p = S.get().project; if (p && confirm(`Delete "${p.name}"?`)) S.deleteProject(p.id); } },
      ],
      Edit: [
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
      Generate: [
        { label: "Generate Whole Montage", hint: "▶", disabled: !hasProject(), action: () => S.generate(null) },
        { label: "Generate Selected Scene", disabled: !sel(), action: () => S.generate(sel()) },
        { sep: true },
        { label: "Project Settings", disabled: !hasProject(), action: () => S.selectScene(null) },
        { label: "Render / Export…", soon: true, disabled: true },
      ],
      Models: [
        { label: "Settings…", action: () => window.ModelsModal.open() },
        { label: "Refresh model list", hint: "R", action: async () => { try { await window.MovieEditorAPI.refreshModels(); } catch (_) {} } },
      ],
      FunPack: [
        { menulabel: "Libraries (in ComfyUI Studio)" },
        { label: "Open ComfyUI", hint: "↗", action: () => window.open("/", "_blank") },
        { sep: true },
        { label: st.health?.reference_loaded ? "Pipeline reference: loaded" : "Pipeline reference: missing", disabled: true },
        { label: `Configured nodes: ${st.health?.configured_slots ?? 0}`, disabled: true },
        { label: st.health?.ok ? `Connected · ${st.health.comfy_url || ""}` : "ComfyUI not reachable", disabled: true },
      ],
    };
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

  function renderChips(st) {
    const hc = document.getElementById("health-chip");
    const ok = st.health?.ok;
    hc.className = "health-chip " + (ok ? "ok" : "bad");
    clear(hc);
    hc.append(el("span", "led"));
    hc.append(el("span", null, ok ? "ComfyUI live" : "offline"));
    const sc = document.getElementById("save-chip");
    sc.className = "save-chip" + (st.saving ? " dirty" : "");
    sc.textContent = st.saving ? "saving…" : (st.project ? "saved" : "");
  }

  veil.addEventListener("click", closeAll);
  window.addEventListener("keydown", (e) => { if (e.key === "Escape") closeAll(); });

  S.subscribe((st) => { render(); renderChips(st); });
  render();
})();
