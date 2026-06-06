// Top menu bar: File / Edit / View / Generate / FunPack + status chips.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const menusEl = document.getElementById("menus");
  const veil = document.getElementById("menu-veil");
  let openName = null;

  function sel() { return S.get().selectedSceneId; }
  function hasProject() { return !!S.get().project; }

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
        { label: "Restart ComfyUI", hint: "⟳", danger: true, action: restartComfy },
        { sep: true },
        { label: st.health?.reference_loaded ? "Pipeline reference: loaded" : "Pipeline reference: missing", disabled: true },
        { label: `Configured nodes: ${st.health?.configured_slots ?? 0}`, disabled: true },
        { label: st.health?.ok ? `Connected · ${window.location.host}` : "ComfyUI not reachable", disabled: true },
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
